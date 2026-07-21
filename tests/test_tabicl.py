"""Numerical validation tests for the JAX nanoTabICL against the PyTorch reference."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from model_tabicl import (
    InducedTransformerBlock as JAXInducedBlock,
    NanoTabICLClassifier,
    NanoTabICLModel,
    QASSMax as JAXQASSMax,
    TransformerBlock as JAXTransformerBlock,
    _apply_rope,
)
from torch_impl.model_tabicl import (
    InducedTransformerBlock as TorchInducedBlock,
    NanoTabICLv2,
    QASSMax as TorchQASSMax,
    Rope as TorchRope,
    TransformerBlock as TorchTransformerBlock,
)


# --------------------------------------------------------------------------- #
# Weight-copying helpers (PyTorch -> JAX/Equinox)                             #
# --------------------------------------------------------------------------- #
def _copy_linear(jax_linear, torch_linear):
    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, jnp.array(torch_linear.weight.detach().numpy()))
    jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, jnp.array(torch_linear.bias.detach().numpy()))
    return jax_linear


def _copy_layernorm(jax_ln, torch_ln):
    jax_ln = eqx.tree_at(lambda m: m.weight, jax_ln, jnp.array(torch_ln.weight.detach().numpy()))
    jax_ln = eqx.tree_at(lambda m: m.bias, jax_ln, jnp.array(torch_ln.bias.detach().numpy()))
    return jax_ln


def _copy_mlp(jax_mlp, torch_seq):
    """Copy a JAX MLP from a torch ``Sequential(Linear, GELU, Linear)``."""
    jax_mlp = eqx.tree_at(lambda m: m.linear1, jax_mlp, _copy_linear(jax_mlp.linear1, torch_seq[0]))
    jax_mlp = eqx.tree_at(lambda m: m.linear2, jax_mlp, _copy_linear(jax_mlp.linear2, torch_seq[2]))
    return jax_mlp


def _copy_qassmax(jax_q, torch_q):
    jax_q = eqx.tree_at(lambda m: m.base_mlp, jax_q, _copy_mlp(jax_q.base_mlp, torch_q.base_mlp))
    jax_q = eqx.tree_at(lambda m: m.query_mlp, jax_q, _copy_mlp(jax_q.query_mlp, torch_q.query_mlp))
    return jax_q


def _copy_transformer_block(jax_b, torch_b):
    jax_b = eqx.tree_at(lambda m: m.in_proj_weight, jax_b, jnp.array(torch_b.in_proj_weight.detach().numpy()))
    jax_b = eqx.tree_at(lambda m: m.in_proj_bias, jax_b, jnp.array(torch_b.in_proj_bias.detach().numpy()))
    jax_b = eqx.tree_at(lambda m: m.out_proj, jax_b, _copy_linear(jax_b.out_proj, torch_b.out_proj))
    jax_b = eqx.tree_at(lambda m: m.mlp, jax_b, _copy_mlp(jax_b.mlp, torch_b.mlp))
    jax_b = eqx.tree_at(lambda m: m.ln_attn, jax_b, _copy_layernorm(jax_b.ln_attn, torch_b.ln_attn))
    jax_b = eqx.tree_at(lambda m: m.ln_mlp, jax_b, _copy_layernorm(jax_b.ln_mlp, torch_b.ln_mlp))
    if jax_b.ssmax_layer is not None:
        jax_b = eqx.tree_at(lambda m: m.ssmax_layer, jax_b, _copy_qassmax(jax_b.ssmax_layer, torch_b.ssmax_layer))
    return jax_b


def _copy_induced_block(jax_b, torch_b):
    jax_b = eqx.tree_at(lambda m: m.tfm1, jax_b, _copy_transformer_block(jax_b.tfm1, torch_b.tfm1))
    jax_b = eqx.tree_at(lambda m: m.tfm2, jax_b, _copy_transformer_block(jax_b.tfm2, torch_b.tfm2))
    jax_b = eqx.tree_at(
        lambda m: m.inducing_vectors, jax_b, jnp.array(torch_b.inducing_vectors.detach().numpy()).squeeze(0)
    )
    return jax_b


def _copy_model(jax_m, torch_m):
    jax_m = eqx.tree_at(lambda m: m.x_embed, jax_m, _copy_linear(jax_m.x_embed, torch_m.x_embed))
    jax_m = eqx.tree_at(lambda m: m.y_embed_in.weight, jax_m, jnp.array(torch_m.y_embed_in.weight.detach().numpy()))
    jax_m = eqx.tree_at(lambda m: m.y_embed_icl.weight, jax_m, jnp.array(torch_m.y_embed_icl.weight.detach().numpy()))
    for i in range(len(torch_m.col_blocks)):
        jax_m = eqx.tree_at(
            lambda m, i=i: m.col_blocks[i], jax_m, _copy_induced_block(jax_m.col_blocks[i], torch_m.col_blocks[i])
        )
    for i in range(len(torch_m.row_blocks)):
        jax_m = eqx.tree_at(
            lambda m, i=i: m.row_blocks[i], jax_m, _copy_transformer_block(jax_m.row_blocks[i], torch_m.row_blocks[i])
        )
    for i in range(len(torch_m.icl_blocks)):
        jax_m = eqx.tree_at(
            lambda m, i=i: m.icl_blocks[i], jax_m, _copy_transformer_block(jax_m.icl_blocks[i], torch_m.icl_blocks[i])
        )
    jax_m = eqx.tree_at(
        lambda m: m.row_cls_tokens, jax_m, jnp.array(torch_m.row_cls_tokens.detach().numpy()).squeeze(0).squeeze(0)
    )
    jax_m = eqx.tree_at(lambda m: m.row_ln, jax_m, _copy_layernorm(jax_m.row_ln, torch_m.row_ln))
    jax_m = eqx.tree_at(lambda m: m.out_ln, jax_m, _copy_layernorm(jax_m.out_ln, torch_m.out_ln))
    jax_m = eqx.tree_at(lambda m: m.out_mlp, jax_m, _copy_mlp(jax_m.out_mlp, torch_m.out_mlp))
    return jax_m


CONFIG = dict(
    max_classes=10,
    out_dim=10,
    embed_dim=16,
    col_num_blocks=2,
    row_num_blocks=2,
    icl_num_blocks=2,
    col_nhead=4,
    row_nhead=4,
    icl_nhead=4,
    feature_group_size=3,
    n_cls_cols=4,
    n_cls_rows=8,
)


@pytest.fixture
def model_setup() -> dict:
    """Create matched JAX and PyTorch nanoTabICL models with test data."""
    torch.manual_seed(0)
    torch_model = NanoTabICLv2(**CONFIG)
    torch_model.eval()

    jax_model = NanoTabICLModel(**CONFIG, key=jax.random.PRNGKey(0))
    jax_model = _copy_model(jax_model, torch_model)

    n_train, n_test, n_cols = 12, 6, 5
    np.random.seed(42)
    x_np = np.random.randn(n_train + n_test, n_cols).astype(np.float32)
    y_np = np.random.randint(0, 3, size=(n_train,)).astype(np.int64)

    return {
        "torch_model": torch_model,
        "jax_model": jax_model,
        "x_np": x_np,
        "y_np": y_np,
        "n_train": n_train,
        "n_test": n_test,
        "n_cols": n_cols,
    }


def _torch_forward(setup: dict) -> np.ndarray:
    with torch.no_grad():
        out = setup["torch_model"](
            torch.from_numpy(setup["x_np"]).unsqueeze(0), torch.from_numpy(setup["y_np"]).unsqueeze(0)
        ).squeeze(0)
    return out.numpy()


def _jax_forward(setup: dict) -> np.ndarray:
    n_train, n_test = setup["n_train"], setup["n_test"]
    y_full = np.concatenate([setup["y_np"], np.zeros(n_test, dtype=np.int64)])
    train_mask = jnp.arange(n_train + n_test) < n_train
    out = setup["jax_model"](jnp.array(setup["x_np"]), jnp.array(y_full), train_mask)
    return np.array(out)[n_train:]  # keep only the meaningful test-row logits


# --------------------------------------------------------------------------- #
# Component tests                                                             #
# --------------------------------------------------------------------------- #
def test_rope_matches() -> None:
    """RoPE application matches the torch Rope module."""
    num_heads, seq, head_dim = 4, 7, 8
    np.random.seed(0)
    x_np = np.random.randn(1, num_heads, seq, head_dim).astype(np.float32)

    torch_rope = TorchRope(head_dim=head_dim, theta=100_000.0)
    with torch.no_grad():
        out_torch = torch_rope(torch.from_numpy(x_np)).squeeze(0).numpy()

    out_jax = np.array(_apply_rope(jnp.array(x_np[0]), head_dim // 2, 100_000.0))
    np.testing.assert_allclose(out_jax, out_torch, atol=1e-5)


def test_qassmax_matches() -> None:
    """QASSMax modulation matches the torch implementation."""
    num_heads, seq, head_dim = 4, 7, 8
    torch.manual_seed(1)
    torch_q = TorchQASSMax(num_heads=num_heads, head_dim=head_dim)
    jax_q = JAXQASSMax(num_heads=num_heads, head_dim=head_dim, key=jax.random.PRNGKey(1))
    jax_q = _copy_qassmax(jax_q, torch_q)

    np.random.seed(0)
    q_np = np.random.randn(1, num_heads, seq, head_dim).astype(np.float32)
    n = 13

    with torch.no_grad():
        out_torch = torch_q(torch.from_numpy(q_np), n=n).squeeze(0).numpy()
    out_jax = np.array(jax_q(jnp.array(q_np[0]), n))
    np.testing.assert_allclose(out_jax, out_torch, atol=1e-5)


@pytest.mark.parametrize(("use_rope", "ssmax"), [(False, False), (True, False), (False, True)])
def test_transformer_block_matches(use_rope: bool, ssmax: bool) -> None:
    """TransformerBlock self-attention matches torch for each attention variant."""
    embed_dim, num_heads, seq = 16, 4, 6
    torch.manual_seed(2)
    torch_b = TorchTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, use_rope=use_rope, ssmax=ssmax)
    torch_b.eval()
    jax_b = JAXTransformerBlock(embed_dim, num_heads, use_rope=use_rope, ssmax=ssmax, key=jax.random.PRNGKey(2))
    jax_b = _copy_transformer_block(jax_b, torch_b)

    np.random.seed(0)
    src_np = np.random.randn(seq, embed_dim).astype(np.float32)

    with torch.no_grad():
        out_torch = torch_b(torch.from_numpy(src_np).unsqueeze(0)).squeeze(0).numpy()
    out_jax = np.array(jax_b(jnp.array(src_np)))
    np.testing.assert_allclose(out_jax, out_torch, atol=1e-4)


def test_transformer_block_kv_max_idx() -> None:
    """TransformerBlock with a restricted key/value window matches torch."""
    embed_dim, num_heads, seq, kv_max = 16, 4, 6, 4
    torch.manual_seed(3)
    torch_b = TorchTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ssmax=True)
    torch_b.eval()
    jax_b = JAXTransformerBlock(embed_dim, num_heads, ssmax=True, key=jax.random.PRNGKey(3))
    jax_b = _copy_transformer_block(jax_b, torch_b)

    np.random.seed(0)
    src_np = np.random.randn(seq, embed_dim).astype(np.float32)

    key_mask = jnp.arange(seq) < kv_max
    with torch.no_grad():
        out_torch = torch_b(torch.from_numpy(src_np).unsqueeze(0), kv_max_idx=kv_max).squeeze(0).numpy()
    out_jax = np.array(jax_b(jnp.array(src_np), key_mask=key_mask))
    np.testing.assert_allclose(out_jax, out_torch, atol=1e-4)


def test_induced_block_matches() -> None:
    """InducedTransformerBlock (column attention) matches torch."""
    embed_dim, num_heads, n_inducing, seq = 16, 4, 8, 6
    torch.manual_seed(4)
    torch_b = TorchInducedBlock(embed_dim=embed_dim, num_heads=num_heads, n_inducing=n_inducing, ssmax=True)
    torch_b.eval()
    jax_b = JAXInducedBlock(embed_dim, num_heads, n_inducing=n_inducing, ssmax=True, key=jax.random.PRNGKey(4))
    jax_b = _copy_induced_block(jax_b, torch_b)

    np.random.seed(0)
    src_np = np.random.randn(seq, embed_dim).astype(np.float32)
    kv_max = 4
    key_mask = jnp.arange(seq) < kv_max

    with torch.no_grad():
        out_torch = torch_b(torch.from_numpy(src_np).unsqueeze(0), kv_max_idx=kv_max).squeeze(0).numpy()
    out_jax = np.array(jax_b(jnp.array(src_np), key_mask=key_mask))
    np.testing.assert_allclose(out_jax, out_torch, atol=1e-4)


# --------------------------------------------------------------------------- #
# Full-model tests                                                           #
# --------------------------------------------------------------------------- #
def test_full_model_shape(model_setup: dict) -> None:
    """Full model returns logits of shape (n_test, out_dim)."""
    out_jax = _jax_forward(model_setup)
    assert out_jax.shape == (model_setup["n_test"], CONFIG["out_dim"])


def test_full_model_logits_match(model_setup: dict) -> None:
    """Full model logits match torch."""
    np.testing.assert_allclose(_jax_forward(model_setup), _torch_forward(model_setup), atol=1e-3)


def test_full_model_predictions_match(model_setup: dict) -> None:
    """Full model argmax predictions match torch."""
    pred_jax = np.argmax(_jax_forward(model_setup), axis=-1)
    pred_torch = np.argmax(_torch_forward(model_setup), axis=-1)
    np.testing.assert_array_equal(pred_jax, pred_torch)


# --------------------------------------------------------------------------- #
# Classifier tests                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def classifier_setup(model_setup: dict) -> dict:
    """Fit a NanoTabICLClassifier on the matched model with train/test split."""
    n_train = model_setup["n_train"]
    x_np, y_np = model_setup["x_np"], model_setup["y_np"]

    classifier = NanoTabICLClassifier(model_setup["jax_model"])
    classifier.fit(x_np[:n_train], y_np.astype(np.float32))

    return {
        "classifier": classifier,
        "torch_model": model_setup["torch_model"],
        "X_train": x_np[:n_train],
        "y_train": y_np,
        "X_test": x_np[n_train:],
        "n_train": n_train,
        "n_test": model_setup["n_test"],
        "num_classes": int(y_np.max()) + 1,
    }


def test_classifier_predict_proba_shape(classifier_setup: dict) -> None:
    """predict_proba returns (n_test, num_classes)."""
    proba = classifier_setup["classifier"].predict_proba(classifier_setup["X_test"])
    assert proba.shape == (classifier_setup["n_test"], classifier_setup["num_classes"])


def test_classifier_predict_proba_sums_to_one(classifier_setup: dict) -> None:
    """predict_proba rows sum to one."""
    proba = classifier_setup["classifier"].predict_proba(classifier_setup["X_test"])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_predict_valid_classes(classifier_setup: dict) -> None:
    """predict returns valid class indices."""
    preds = classifier_setup["classifier"].predict(classifier_setup["X_test"])
    assert preds.shape == (classifier_setup["n_test"],)
    assert all(0 <= p < classifier_setup["num_classes"] for p in preds)


def test_classifier_num_classes_detection(classifier_setup: dict) -> None:
    """Classifier infers the number of classes from the labels."""
    assert classifier_setup["classifier"].num_classes == classifier_setup["num_classes"]


def test_classifier_matches_torch(classifier_setup: dict) -> None:
    """Classifier probabilities and predictions match the torch reference."""
    setup = classifier_setup
    num_classes = setup["num_classes"]

    x = np.concatenate([setup["X_train"], setup["X_test"]], axis=0)
    with torch.no_grad():
        out_torch = setup["torch_model"](
            torch.from_numpy(x).unsqueeze(0), torch.from_numpy(setup["y_train"]).unsqueeze(0)
        ).squeeze(0)
    out_torch_np = out_torch.numpy()[:, :num_classes]
    proba_torch = torch.nn.functional.softmax(torch.from_numpy(out_torch_np), dim=-1).numpy()
    pred_torch = np.argmax(out_torch_np, axis=-1)

    proba_jax = setup["classifier"].predict_proba(setup["X_test"])
    pred_jax = setup["classifier"].predict(setup["X_test"])

    np.testing.assert_allclose(proba_jax, proba_torch, atol=1e-3)
    np.testing.assert_array_equal(pred_jax, pred_torch)

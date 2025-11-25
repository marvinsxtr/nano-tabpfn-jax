"""Numerical validation tests for NanoTabPFNClassifier: JAX vs PyTorch."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from model import NanoTabPFNClassifier, NanoTabPFNModel as JAXNanoTabPFN
from torch_impl.model import NanoTabPFNModel as TorchNanoTabPFN


def _copy_torch_linear_to_jax(jax_linear: eqx.nn.Linear, torch_linear: torch.nn.Linear) -> eqx.nn.Linear:
    """Copy weights and biases from PyTorch Linear to JAX Linear."""
    weight = torch_linear.weight.detach().cpu().numpy()
    bias = torch_linear.bias.detach().cpu().numpy()
    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, jnp.array(weight))
    jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, jnp.array(bias))
    return jax_linear


def _copy_torch_layernorm_to_jax(jax_ln: eqx.nn.LayerNorm, torch_ln: torch.nn.LayerNorm) -> eqx.nn.LayerNorm:
    """Copy weights and biases from PyTorch LayerNorm to JAX LayerNorm."""
    weight = torch_ln.weight.detach().cpu().numpy()
    bias = torch_ln.bias.detach().cpu().numpy()
    jax_ln = eqx.tree_at(lambda m: m.weight, jax_ln, jnp.array(weight))
    jax_ln = eqx.tree_at(lambda m: m.bias, jax_ln, jnp.array(bias))
    return jax_ln


def _copy_torch_mha_to_eqx_mha(
    jax_mha: eqx.nn.MultiheadAttention,
    torch_mha: torch.nn.MultiheadAttention,
) -> eqx.nn.MultiheadAttention:
    """Copy PyTorch MultiheadAttention weights to equinox MultiheadAttention."""
    embed_dim = torch_mha.embed_dim

    in_proj_weight = torch_mha.in_proj_weight.detach().cpu().numpy()
    in_proj_bias = torch_mha.in_proj_bias.detach().cpu().numpy()
    out_proj_weight = torch_mha.out_proj.weight.detach().cpu().numpy()
    out_proj_bias = torch_mha.out_proj.bias.detach().cpu().numpy()

    w_q = in_proj_weight[:embed_dim, :]
    w_k = in_proj_weight[embed_dim : 2 * embed_dim, :]
    w_v = in_proj_weight[2 * embed_dim :, :]

    b_q = in_proj_bias[:embed_dim]
    b_k = in_proj_bias[embed_dim : 2 * embed_dim]
    b_v = in_proj_bias[2 * embed_dim :]

    jax_mha = eqx.tree_at(lambda m: m.query_proj.weight, jax_mha, jnp.array(w_q))
    jax_mha = eqx.tree_at(lambda m: m.query_proj.bias, jax_mha, jnp.array(b_q))
    jax_mha = eqx.tree_at(lambda m: m.key_proj.weight, jax_mha, jnp.array(w_k))
    jax_mha = eqx.tree_at(lambda m: m.key_proj.bias, jax_mha, jnp.array(b_k))
    jax_mha = eqx.tree_at(lambda m: m.value_proj.weight, jax_mha, jnp.array(w_v))
    jax_mha = eqx.tree_at(lambda m: m.value_proj.bias, jax_mha, jnp.array(b_v))
    jax_mha = eqx.tree_at(lambda m: m.output_proj.weight, jax_mha, jnp.array(out_proj_weight))
    jax_mha = eqx.tree_at(lambda m: m.output_proj.bias, jax_mha, jnp.array(out_proj_bias))

    return jax_mha


def _copy_model_weights(jax_model: JAXNanoTabPFN, torch_model: TorchNanoTabPFN) -> JAXNanoTabPFN:
    """Copy all weights from PyTorch model to JAX model.

    Args:
        jax_model: JAX model to copy weights into.
        torch_model: PyTorch model to copy weights from.

    Returns:
        JAX model with copied weights.
    """
    jax_model = eqx.tree_at(
        lambda m: m.feature_encoder.linear_layer,
        jax_model,
        _copy_torch_linear_to_jax(jax_model.feature_encoder.linear_layer, torch_model.feature_encoder.linear_layer),
    )

    jax_model = eqx.tree_at(
        lambda m: m.target_encoder.linear_layer,
        jax_model,
        _copy_torch_linear_to_jax(jax_model.target_encoder.linear_layer, torch_model.target_encoder.linear_layer),
    )

    for i, torch_block in enumerate(torch_model.transformer_blocks):
        jax_block = jax_model.transformer_blocks[i]

        jax_block = eqx.tree_at(
            lambda m: m.self_attn_features,
            jax_block,
            _copy_torch_mha_to_eqx_mha(jax_block.self_attn_features, torch_block.self_attention_between_features),
        )
        jax_block = eqx.tree_at(
            lambda m: m.self_attn_datapoints,
            jax_block,
            _copy_torch_mha_to_eqx_mha(jax_block.self_attn_datapoints, torch_block.self_attention_between_datapoints),
        )

        jax_block = eqx.tree_at(
            lambda m: m.linear1,
            jax_block,
            _copy_torch_linear_to_jax(jax_block.linear1, torch_block.linear1),
        )
        jax_block = eqx.tree_at(
            lambda m: m.linear2,
            jax_block,
            _copy_torch_linear_to_jax(jax_block.linear2, torch_block.linear2),
        )

        jax_block = eqx.tree_at(
            lambda m: m.norm1,
            jax_block,
            _copy_torch_layernorm_to_jax(jax_block.norm1, torch_block.norm1),
        )
        jax_block = eqx.tree_at(
            lambda m: m.norm2,
            jax_block,
            _copy_torch_layernorm_to_jax(jax_block.norm2, torch_block.norm2),
        )
        jax_block = eqx.tree_at(
            lambda m: m.norm3,
            jax_block,
            _copy_torch_layernorm_to_jax(jax_block.norm3, torch_block.norm3),
        )

        jax_model = eqx.tree_at(lambda m, idx=i: m.transformer_blocks[idx], jax_model, jax_block)

    jax_model = eqx.tree_at(
        lambda m: m.decoder.linear1,
        jax_model,
        _copy_torch_linear_to_jax(jax_model.decoder.linear1, torch_model.decoder.linear1),
    )
    jax_model = eqx.tree_at(
        lambda m: m.decoder.linear2,
        jax_model,
        _copy_torch_linear_to_jax(jax_model.decoder.linear2, torch_model.decoder.linear2),
    )

    return jax_model


@pytest.fixture
def classifier_setup() -> dict:
    """Create matched JAX classifier and PyTorch model with test data.

    Returns:
        Dictionary containing classifier, torch model, test data, and configuration.
    """
    embedding_size = 16
    num_attention_heads = 4
    mlp_hidden_size = 32
    num_layers = 2
    num_outputs = 10

    num_train = 20
    num_test = 5
    num_features = 3
    num_classes = 3

    np.random.seed(42)
    X_train = np.random.randn(num_train, num_features).astype(np.float32)
    y_train = np.random.randint(0, num_classes, size=(num_train,)).astype(np.float32)
    X_test = np.random.randn(num_test, num_features).astype(np.float32)

    torch_model = TorchNanoTabPFN(embedding_size, num_attention_heads, mlp_hidden_size, num_layers, num_outputs)
    torch_model.eval()

    key = jax.random.PRNGKey(0)
    jax_model = JAXNanoTabPFN(embedding_size, num_attention_heads, mlp_hidden_size, num_layers, num_outputs, key=key)
    jax_model = _copy_model_weights(jax_model, torch_model)

    classifier = NanoTabPFNClassifier(jax_model)
    classifier.fit(X_train, y_train)

    return {
        "classifier": classifier,
        "torch_model": torch_model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "num_train": num_train,
        "num_test": num_test,
        "num_classes": num_classes,
        "num_outputs": num_outputs,
    }


def test_classifier_predict_proba_shape(classifier_setup: dict) -> None:
    """Test that predict_proba returns correct shape."""
    setup = classifier_setup
    proba = setup["classifier"].predict_proba(setup["X_test"])

    assert proba.shape == (setup["num_test"], setup["num_classes"])


def test_classifier_predict_proba_sums_to_one(classifier_setup: dict) -> None:
    """Test that probabilities sum to 1."""
    setup = classifier_setup
    proba = setup["classifier"].predict_proba(setup["X_test"])

    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_predict_shape(classifier_setup: dict) -> None:
    """Test that predict returns correct shape."""
    setup = classifier_setup
    predictions = setup["classifier"].predict(setup["X_test"])

    assert predictions.shape == (setup["num_test"],)


def test_classifier_predict_valid_classes(classifier_setup: dict) -> None:
    """Test that predictions are valid class indices."""
    setup = classifier_setup
    predictions = setup["classifier"].predict(setup["X_test"])

    assert all(0 <= p < setup["num_classes"] for p in predictions)


def test_classifier_matches_torch_logits(classifier_setup: dict) -> None:
    """Test that classifier logits match PyTorch model output."""
    setup = classifier_setup

    X_combined = np.concatenate([setup["X_train"], setup["X_test"]], axis=0)

    num_features = X_combined.shape[1]
    if num_features < 10:
        padding = np.zeros((X_combined.shape[0], 10 - num_features), dtype=np.float32)
        X_padded = np.concatenate([X_combined, padding], axis=1)
    else:
        X_padded = X_combined

    x_torch = torch.from_numpy(X_padded).unsqueeze(0)
    y_torch = torch.from_numpy(setup["y_train"]).unsqueeze(0)

    with torch.no_grad():
        out_torch = setup["torch_model"]((x_torch, y_torch), setup["num_train"]).squeeze(0)
    out_torch_np = out_torch.cpu().numpy()[:, : setup["num_classes"]]

    x_jax = jnp.array(X_padded)
    y_jax = jnp.concatenate([setup["y_train"], jnp.full(setup["num_test"], setup["y_train"].mean())])
    train_mask = jnp.arange(len(X_padded)) < setup["num_train"]

    jax_model = setup["classifier"].model
    out_jax = jax_model(x_jax, y_jax, train_mask=train_mask)
    out_jax_test = np.array(out_jax)[setup["num_train"] :, : setup["num_classes"]]

    np.testing.assert_allclose(out_jax_test, out_torch_np, atol=1e-3)


def test_classifier_matches_torch_predictions(classifier_setup: dict) -> None:
    """Test that classifier predictions match PyTorch model predictions."""
    setup = classifier_setup

    X_combined = np.concatenate([setup["X_train"], setup["X_test"]], axis=0)

    num_features = X_combined.shape[1]
    if num_features < 10:
        padding = np.zeros((X_combined.shape[0], 10 - num_features), dtype=np.float32)
        X_padded = np.concatenate([X_combined, padding], axis=1)
    else:
        X_padded = X_combined

    x_torch = torch.from_numpy(X_padded).unsqueeze(0)
    y_torch = torch.from_numpy(setup["y_train"]).unsqueeze(0)

    with torch.no_grad():
        out_torch = setup["torch_model"]((x_torch, y_torch), setup["num_train"]).squeeze(0)
    out_torch_np = out_torch.cpu().numpy()[:, : setup["num_classes"]]
    pred_torch = np.argmax(out_torch_np, axis=-1)

    pred_jax = setup["classifier"].predict(setup["X_test"])

    np.testing.assert_array_equal(pred_jax, pred_torch)


def test_classifier_matches_torch_probabilities(classifier_setup: dict) -> None:
    """Test that classifier probabilities match PyTorch model probabilities."""
    setup = classifier_setup

    X_combined = np.concatenate([setup["X_train"], setup["X_test"]], axis=0)

    num_features = X_combined.shape[1]
    if num_features < 10:
        padding = np.zeros((X_combined.shape[0], 10 - num_features), dtype=np.float32)
        X_padded = np.concatenate([X_combined, padding], axis=1)
    else:
        X_padded = X_combined

    x_torch = torch.from_numpy(X_padded).unsqueeze(0)
    y_torch = torch.from_numpy(setup["y_train"]).unsqueeze(0)

    with torch.no_grad():
        out_torch = setup["torch_model"]((x_torch, y_torch), setup["num_train"]).squeeze(0)
    out_torch_np = out_torch.cpu().numpy()[:, : setup["num_classes"]]
    proba_torch = torch.nn.functional.softmax(torch.from_numpy(out_torch_np), dim=-1).numpy()

    proba_jax = setup["classifier"].predict_proba(setup["X_test"])

    np.testing.assert_allclose(proba_jax, proba_torch, atol=1e-3)


def test_classifier_num_classes_detection(classifier_setup: dict) -> None:
    """Test that classifier correctly detects number of classes."""
    setup = classifier_setup
    assert setup["classifier"].num_classes == setup["num_classes"]


def test_classifier_stores_training_data(classifier_setup: dict) -> None:
    """Test that classifier stores training data correctly."""
    setup = classifier_setup
    np.testing.assert_array_equal(setup["classifier"].X_train, setup["X_train"])
    np.testing.assert_array_equal(setup["classifier"].y_train, setup["y_train"])
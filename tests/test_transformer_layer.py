"""Step-by-step numerical validation tests for TransformerEncoderLayer."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as f

from model import TransformerEncoderLayer as JAXTransformerLayer
from torch_impl.model import TransformerEncoderLayer as TorchTransformerLayer


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


@pytest.fixture
def transformer_setup() -> dict:
    """Create matched JAX and PyTorch transformer layers with test data.

    Returns:
        Dictionary containing layers, test data, and configuration.
    """
    embedding_size = 8
    nhead = 2
    mlp_hidden_size = 16
    num_rows = 6
    num_features_plus_target = 3
    train_test_split_index = 4

    np.random.seed(42)
    src_np = np.random.randn(num_rows, num_features_plus_target, embedding_size).astype(np.float32)

    torch_layer = TorchTransformerLayer(embedding_size, nhead, mlp_hidden_size)
    torch_layer.eval()

    key = jax.random.PRNGKey(0)
    jax_layer = JAXTransformerLayer(embedding_size, nhead, mlp_hidden_size, key=key)

    jax_layer = eqx.tree_at(
        lambda m: m.self_attn_features,
        jax_layer,
        _copy_torch_mha_to_eqx_mha(jax_layer.self_attn_features, torch_layer.self_attention_between_features),
    )
    jax_layer = eqx.tree_at(
        lambda m: m.self_attn_datapoints,
        jax_layer,
        _copy_torch_mha_to_eqx_mha(jax_layer.self_attn_datapoints, torch_layer.self_attention_between_datapoints),
    )
    jax_layer = eqx.tree_at(
        lambda m: m.linear1,
        jax_layer,
        _copy_torch_linear_to_jax(jax_layer.linear1, torch_layer.linear1),
    )
    jax_layer = eqx.tree_at(
        lambda m: m.linear2,
        jax_layer,
        _copy_torch_linear_to_jax(jax_layer.linear2, torch_layer.linear2),
    )
    jax_layer = eqx.tree_at(
        lambda m: m.norm1,
        jax_layer,
        _copy_torch_layernorm_to_jax(jax_layer.norm1, torch_layer.norm1),
    )
    jax_layer = eqx.tree_at(
        lambda m: m.norm2,
        jax_layer,
        _copy_torch_layernorm_to_jax(jax_layer.norm2, torch_layer.norm2),
    )
    jax_layer = eqx.tree_at(
        lambda m: m.norm3,
        jax_layer,
        _copy_torch_layernorm_to_jax(jax_layer.norm3, torch_layer.norm3),
    )

    return {
        "torch_layer": torch_layer,
        "jax_layer": jax_layer,
        "src_np": src_np,
        "num_rows": num_rows,
        "num_features_plus_target": num_features_plus_target,
        "embedding_size": embedding_size,
        "train_test_split_index": train_test_split_index,
    }


def test_feature_attention(transformer_setup: dict) -> None:
    """Test attention between features matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])

    batch_size = 1
    rows_size = setup["num_rows"]
    col_size = setup["num_features_plus_target"]
    emb_size = setup["embedding_size"]

    src_torch_reshaped = src_torch.unsqueeze(0).reshape(batch_size * rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_feat = setup["torch_layer"].self_attention_between_features(
            src_torch_reshaped, src_torch_reshaped, src_torch_reshaped
        )[0]
    src_torch_feat = src_torch_feat + src_torch_reshaped
    src_torch_feat_np = src_torch_feat.cpu().numpy()

    src_jax_feat = jax.vmap(setup["jax_layer"].self_attn_features)(src_jax, src_jax, src_jax) + src_jax
    src_jax_feat_np = np.array(src_jax_feat)

    np.testing.assert_allclose(src_jax_feat_np, src_torch_feat_np, atol=1e-2)


def test_layernorm1(transformer_setup: dict) -> None:
    """Test first LayerNorm matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])

    batch_size = 1
    rows_size = setup["num_rows"]
    col_size = setup["num_features_plus_target"]
    emb_size = setup["embedding_size"]

    src_torch_reshaped = src_torch.unsqueeze(0).reshape(batch_size * rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_feat = setup["torch_layer"].self_attention_between_features(
            src_torch_reshaped, src_torch_reshaped, src_torch_reshaped
        )[0]
    src_torch_feat = src_torch_feat + src_torch_reshaped
    src_torch_feat_reshaped = src_torch_feat.reshape(batch_size, rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_norm1 = setup["torch_layer"].norm1(src_torch_feat_reshaped)
    src_torch_norm1_np = src_torch_norm1.cpu().numpy()

    src_jax_feat = jax.vmap(setup["jax_layer"].self_attn_features)(src_jax, src_jax, src_jax) + src_jax
    src_jax_norm1 = jax.vmap(jax.vmap(setup["jax_layer"].norm1))(src_jax_feat)
    src_jax_norm1_np = np.array(src_jax_norm1)

    np.testing.assert_allclose(src_jax_norm1_np, src_torch_norm1_np.squeeze(0), atol=1e-2)


def test_datapoint_attention(transformer_setup: dict) -> None:
    """Test attention between datapoints matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])
    train_test_split_index = setup["train_test_split_index"]

    batch_size = 1
    rows_size = setup["num_rows"]
    col_size = setup["num_features_plus_target"]
    emb_size = setup["embedding_size"]

    src_torch_reshaped = src_torch.unsqueeze(0).reshape(batch_size * rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_feat = setup["torch_layer"].self_attention_between_features(
            src_torch_reshaped, src_torch_reshaped, src_torch_reshaped
        )[0]
    src_torch_feat = src_torch_feat + src_torch_reshaped
    src_torch_feat_reshaped = src_torch_feat.reshape(batch_size, rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_norm1 = setup["torch_layer"].norm1(src_torch_feat_reshaped)

    src_torch_transposed = src_torch_norm1.transpose(1, 2).reshape(batch_size * col_size, rows_size, emb_size)
    with torch.no_grad():
        src_left = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_right = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, train_test_split_index:],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_torch_dp = torch.cat([src_left, src_right], dim=1) + src_torch_transposed
    src_torch_dp = src_torch_dp.reshape(batch_size, col_size, rows_size, emb_size).transpose(2, 1)
    src_torch_dp_np = src_torch_dp.cpu().numpy()

    src_jax_feat = jax.vmap(setup["jax_layer"].self_attn_features)(src_jax, src_jax, src_jax) + src_jax
    src_jax_norm1 = jax.vmap(jax.vmap(setup["jax_layer"].norm1))(src_jax_feat)
    src_jax_transposed = jnp.transpose(src_jax_norm1, (1, 0, 2))
    train_mask_jax = jnp.arange(rows_size) < train_test_split_index
    mask = jnp.broadcast_to(train_mask_jax, (rows_size, rows_size))
    mha_fn = lambda q, k, v: setup["jax_layer"].self_attn_datapoints(q, k, v, mask=mask)
    src_jax_dp = jax.vmap(mha_fn)(src_jax_transposed, src_jax_transposed, src_jax_transposed) + src_jax_transposed
    src_jax_dp = jnp.transpose(src_jax_dp, (1, 0, 2))
    src_jax_dp_np = np.array(src_jax_dp)

    np.testing.assert_allclose(src_jax_dp_np, src_torch_dp_np.squeeze(0), atol=1e-2)


def test_layernorm2(transformer_setup: dict) -> None:
    """Test second LayerNorm matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])
    train_test_split_index = setup["train_test_split_index"]

    batch_size = 1
    rows_size = setup["num_rows"]
    col_size = setup["num_features_plus_target"]
    emb_size = setup["embedding_size"]

    src_torch_reshaped = src_torch.unsqueeze(0).reshape(batch_size * rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_feat = setup["torch_layer"].self_attention_between_features(
            src_torch_reshaped, src_torch_reshaped, src_torch_reshaped
        )[0]
    src_torch_feat = src_torch_feat + src_torch_reshaped
    src_torch_feat_reshaped = src_torch_feat.reshape(batch_size, rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_norm1 = setup["torch_layer"].norm1(src_torch_feat_reshaped)
    src_torch_transposed = src_torch_norm1.transpose(1, 2).reshape(batch_size * col_size, rows_size, emb_size)
    with torch.no_grad():
        src_left = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_right = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, train_test_split_index:],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_torch_dp = torch.cat([src_left, src_right], dim=1) + src_torch_transposed
    src_torch_dp = src_torch_dp.reshape(batch_size, col_size, rows_size, emb_size).transpose(2, 1)
    with torch.no_grad():
        src_torch_norm2 = setup["torch_layer"].norm2(src_torch_dp)
    src_torch_norm2_np = src_torch_norm2.cpu().numpy()

    src_jax_feat = jax.vmap(setup["jax_layer"].self_attn_features)(src_jax, src_jax, src_jax) + src_jax
    src_jax_norm1 = jax.vmap(jax.vmap(setup["jax_layer"].norm1))(src_jax_feat)
    src_jax_transposed = jnp.transpose(src_jax_norm1, (1, 0, 2))
    train_mask_jax = jnp.arange(rows_size) < train_test_split_index
    mask = jnp.broadcast_to(train_mask_jax, (rows_size, rows_size))
    mha_fn = lambda q, k, v: setup["jax_layer"].self_attn_datapoints(q, k, v, mask=mask)
    src_jax_dp = jax.vmap(mha_fn)(src_jax_transposed, src_jax_transposed, src_jax_transposed) + src_jax_transposed
    src_jax_dp = jnp.transpose(src_jax_dp, (1, 0, 2))
    src_jax_norm2 = jax.vmap(jax.vmap(setup["jax_layer"].norm2))(src_jax_dp)
    src_jax_norm2_np = np.array(src_jax_norm2)

    np.testing.assert_allclose(src_jax_norm2_np, src_torch_norm2_np.squeeze(0), atol=1e-2)


def test_mlp(transformer_setup: dict) -> None:
    """Test MLP block matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])
    train_test_split_index = setup["train_test_split_index"]

    batch_size = 1
    rows_size = setup["num_rows"]
    col_size = setup["num_features_plus_target"]
    emb_size = setup["embedding_size"]

    src_torch_reshaped = src_torch.unsqueeze(0).reshape(batch_size * rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_feat = setup["torch_layer"].self_attention_between_features(
            src_torch_reshaped, src_torch_reshaped, src_torch_reshaped
        )[0]
    src_torch_feat = src_torch_feat + src_torch_reshaped
    src_torch_feat_reshaped = src_torch_feat.reshape(batch_size, rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_norm1 = setup["torch_layer"].norm1(src_torch_feat_reshaped)
    src_torch_transposed = src_torch_norm1.transpose(1, 2).reshape(batch_size * col_size, rows_size, emb_size)
    with torch.no_grad():
        src_left = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_right = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, train_test_split_index:],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_torch_dp = torch.cat([src_left, src_right], dim=1) + src_torch_transposed
    src_torch_dp = src_torch_dp.reshape(batch_size, col_size, rows_size, emb_size).transpose(2, 1)
    with torch.no_grad():
        src_torch_norm2 = setup["torch_layer"].norm2(src_torch_dp)
        src_torch_mlp = (
            setup["torch_layer"].linear2(f.gelu(setup["torch_layer"].linear1(src_torch_norm2))) + src_torch_norm2
        )
    src_torch_mlp_np = src_torch_mlp.cpu().numpy()

    src_jax_feat = jax.vmap(setup["jax_layer"].self_attn_features)(src_jax, src_jax, src_jax) + src_jax
    src_jax_norm1 = jax.vmap(jax.vmap(setup["jax_layer"].norm1))(src_jax_feat)
    src_jax_transposed = jnp.transpose(src_jax_norm1, (1, 0, 2))
    train_mask_jax = jnp.arange(rows_size) < train_test_split_index
    mask = jnp.broadcast_to(train_mask_jax, (rows_size, rows_size))
    mha_fn = lambda q, k, v: setup["jax_layer"].self_attn_datapoints(q, k, v, mask=mask)
    src_jax_dp = jax.vmap(mha_fn)(src_jax_transposed, src_jax_transposed, src_jax_transposed) + src_jax_transposed
    src_jax_dp = jnp.transpose(src_jax_dp, (1, 0, 2))
    src_jax_norm2 = jax.vmap(jax.vmap(setup["jax_layer"].norm2))(src_jax_dp)
    jax_layer = setup["jax_layer"]
    src_jax_mlp = (
        jax.vmap(jax.vmap(lambda x: jax_layer.linear2(jax.nn.gelu(jax_layer.linear1(x)))))(src_jax_norm2)
        + src_jax_norm2
    )
    src_jax_mlp_np = np.array(src_jax_mlp)

    np.testing.assert_allclose(src_jax_mlp_np, src_torch_mlp_np.squeeze(0), atol=1e-2)


def test_layernorm3_final(transformer_setup: dict) -> None:
    """Test final LayerNorm matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])
    train_test_split_index = setup["train_test_split_index"]

    batch_size = 1
    rows_size = setup["num_rows"]
    col_size = setup["num_features_plus_target"]
    emb_size = setup["embedding_size"]

    src_torch_reshaped = src_torch.unsqueeze(0).reshape(batch_size * rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_feat = setup["torch_layer"].self_attention_between_features(
            src_torch_reshaped, src_torch_reshaped, src_torch_reshaped
        )[0]
    src_torch_feat = src_torch_feat + src_torch_reshaped
    src_torch_feat_reshaped = src_torch_feat.reshape(batch_size, rows_size, col_size, emb_size)
    with torch.no_grad():
        src_torch_norm1 = setup["torch_layer"].norm1(src_torch_feat_reshaped)
    src_torch_transposed = src_torch_norm1.transpose(1, 2).reshape(batch_size * col_size, rows_size, emb_size)
    with torch.no_grad():
        src_left = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_right = setup["torch_layer"].self_attention_between_datapoints(
            src_torch_transposed[:, train_test_split_index:],
            src_torch_transposed[:, :train_test_split_index],
            src_torch_transposed[:, :train_test_split_index],
        )[0]
        src_torch_dp = torch.cat([src_left, src_right], dim=1) + src_torch_transposed
    src_torch_dp = src_torch_dp.reshape(batch_size, col_size, rows_size, emb_size).transpose(2, 1)
    with torch.no_grad():
        src_torch_norm2 = setup["torch_layer"].norm2(src_torch_dp)
        src_torch_mlp = (
            setup["torch_layer"].linear2(f.gelu(setup["torch_layer"].linear1(src_torch_norm2))) + src_torch_norm2
        )
        src_torch_final = setup["torch_layer"].norm3(src_torch_mlp)
    src_torch_final_np = src_torch_final.squeeze(0).cpu().numpy()

    src_jax_feat = jax.vmap(setup["jax_layer"].self_attn_features)(src_jax, src_jax, src_jax) + src_jax
    src_jax_norm1 = jax.vmap(jax.vmap(setup["jax_layer"].norm1))(src_jax_feat)
    src_jax_transposed = jnp.transpose(src_jax_norm1, (1, 0, 2))
    train_mask_jax = jnp.arange(rows_size) < train_test_split_index
    mask = jnp.broadcast_to(train_mask_jax, (rows_size, rows_size))
    mha_fn = lambda q, k, v: setup["jax_layer"].self_attn_datapoints(q, k, v, mask=mask)
    src_jax_dp = jax.vmap(mha_fn)(src_jax_transposed, src_jax_transposed, src_jax_transposed) + src_jax_transposed
    src_jax_dp = jnp.transpose(src_jax_dp, (1, 0, 2))
    src_jax_norm2 = jax.vmap(jax.vmap(setup["jax_layer"].norm2))(src_jax_dp)
    jax_layer = setup["jax_layer"]
    src_jax_mlp = (
        jax.vmap(jax.vmap(lambda x: jax_layer.linear2(jax.nn.gelu(jax_layer.linear1(x)))))(src_jax_norm2)
        + src_jax_norm2
    )
    src_jax_final = jax.vmap(jax.vmap(setup["jax_layer"].norm3))(src_jax_mlp)
    src_jax_final_np = np.array(src_jax_final)

    np.testing.assert_allclose(src_jax_final_np, src_torch_final_np, atol=1e-2)


def test_end_to_end_forward_pass(transformer_setup: dict) -> None:
    """Test full forward pass matches between JAX and PyTorch."""
    setup = transformer_setup
    src_torch = torch.from_numpy(setup["src_np"])
    src_jax = jnp.array(setup["src_np"])
    train_test_split_index = setup["train_test_split_index"]
    rows_size = setup["num_rows"]

    src_torch_input = src_torch.unsqueeze(0)
    with torch.no_grad():
        out_torch_e2e = setup["torch_layer"](src_torch_input, train_test_split_index).squeeze(0)
    out_torch_e2e_np = out_torch_e2e.cpu().numpy()

    train_mask_jax = jnp.arange(rows_size) < train_test_split_index
    out_jax_e2e = setup["jax_layer"](src_jax, train_mask_jax)
    out_jax_e2e_np = np.array(out_jax_e2e)

    np.testing.assert_allclose(out_jax_e2e_np, out_torch_e2e_np, atol=1e-2)
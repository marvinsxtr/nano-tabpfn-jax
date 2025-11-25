"""Numerical validation tests for full NanoTabPFNModel: JAX vs PyTorch."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from model import NanoTabPFNModel as JAXNanoTabPFN
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


def _copy_torch_mha_to_jax(jax_layer, torch_mha, is_features: bool = True):
    """Copy PyTorch MultiheadAttention weights to JAX attention layers."""
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

    if is_features:
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_q.weight, jax_layer, jnp.array(w_q))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_q.bias, jax_layer, jnp.array(b_q))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_k.weight, jax_layer, jnp.array(w_k))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_k.bias, jax_layer, jnp.array(b_k))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_v.weight, jax_layer, jnp.array(w_v))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_v.bias, jax_layer, jnp.array(b_v))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_out.weight, jax_layer, jnp.array(out_proj_weight))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_features_out.bias, jax_layer, jnp.array(out_proj_bias))
    else:
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_q.weight, jax_layer, jnp.array(w_q))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_q.bias, jax_layer, jnp.array(b_q))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_k.weight, jax_layer, jnp.array(w_k))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_k.bias, jax_layer, jnp.array(b_k))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_v.weight, jax_layer, jnp.array(w_v))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_v.bias, jax_layer, jnp.array(b_v))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_out.weight, jax_layer, jnp.array(out_proj_weight))
        jax_layer = eqx.tree_at(lambda m: m.self_attn_datapoints_out.bias, jax_layer, jnp.array(out_proj_bias))

    return jax_layer


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

        jax_block = _copy_torch_mha_to_jax(jax_block, torch_block.self_attention_between_features, is_features=True)
        jax_block = _copy_torch_mha_to_jax(jax_block, torch_block.self_attention_between_datapoints, is_features=False)

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

        jax_model = eqx.tree_at(lambda m: m.transformer_blocks[i], jax_model, jax_block)

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
def full_model_setup() -> dict:
    """Create matched JAX and PyTorch full models with test data.

    Returns:
        Dictionary containing models, test data, and configuration.
    """
    embedding_size = 16
    num_attention_heads = 4
    mlp_hidden_size = 32
    num_layers = 2
    num_outputs = 2

    num_rows = 8
    num_features = 3
    train_test_split_index = 5

    np.random.seed(42)
    x_np = np.random.randn(num_rows, num_features).astype(np.float32)
    y_np = np.random.randint(0, num_outputs, size=(num_rows,)).astype(np.float32)

    torch_model = TorchNanoTabPFN(embedding_size, num_attention_heads, mlp_hidden_size, num_layers, num_outputs)
    torch_model.eval()

    key = jax.random.PRNGKey(0)
    jax_model = JAXNanoTabPFN(embedding_size, num_attention_heads, mlp_hidden_size, num_layers, num_outputs, key=key)
    jax_model = _copy_model_weights(jax_model, torch_model)

    return {
        "torch_model": torch_model,
        "jax_model": jax_model,
        "x_np": x_np,
        "y_np": y_np,
        "num_rows": num_rows,
        "num_outputs": num_outputs,
        "train_test_split_index": train_test_split_index,
    }


def test_full_model_output_shape(full_model_setup: dict) -> None:
    """Test that JAX and PyTorch full models produce expected output shapes."""
    setup = full_model_setup
    train_test_split_index = setup["train_test_split_index"]

    x_torch = torch.from_numpy(setup["x_np"])
    y_train_torch = torch.from_numpy(setup["y_np"][:train_test_split_index])

    with torch.no_grad():
        out_torch = setup["torch_model"](
            (x_torch.unsqueeze(0), y_train_torch.unsqueeze(0)), train_test_split_index
        ).squeeze(0)

    x_jax = jnp.array(setup["x_np"])
    y_jax = jnp.array(setup["y_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < train_test_split_index

    out_jax = setup["jax_model"](x_jax, y_jax, train_mask=train_mask_jax)

    num_test = setup["num_rows"] - train_test_split_index
    assert out_torch.shape == (num_test, setup["num_outputs"])
    assert out_jax.shape == (setup["num_rows"], setup["num_outputs"])


def test_full_model_test_outputs_match(full_model_setup: dict) -> None:
    """Test that test sample outputs match between JAX and PyTorch."""
    setup = full_model_setup
    train_test_split_index = setup["train_test_split_index"]

    x_torch = torch.from_numpy(setup["x_np"])
    y_train_torch = torch.from_numpy(setup["y_np"][:train_test_split_index])

    with torch.no_grad():
        out_torch = setup["torch_model"](
            (x_torch.unsqueeze(0), y_train_torch.unsqueeze(0)), train_test_split_index
        ).squeeze(0)
    out_torch_np = out_torch.cpu().numpy()

    x_jax = jnp.array(setup["x_np"])
    y_jax = jnp.array(setup["y_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < train_test_split_index

    out_jax = setup["jax_model"](x_jax, y_jax, train_mask=train_mask_jax)
    out_jax_test = np.array(out_jax)[train_test_split_index:]

    np.testing.assert_allclose(out_jax_test, out_torch_np, atol=1e-3)


def test_full_model_train_outputs_zeroed(full_model_setup: dict) -> None:
    """Test that JAX model zeros out training sample outputs."""
    setup = full_model_setup
    train_test_split_index = setup["train_test_split_index"]

    x_jax = jnp.array(setup["x_np"])
    y_jax = jnp.array(setup["y_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < train_test_split_index

    out_jax = setup["jax_model"](x_jax, y_jax, train_mask=train_mask_jax)
    out_jax_train = np.array(out_jax)[:train_test_split_index]

    np.testing.assert_allclose(out_jax_train, 0.0, atol=1e-10)


def test_full_model_predictions_match(full_model_setup: dict) -> None:
    """Test that argmax predictions match between JAX and PyTorch."""
    setup = full_model_setup
    train_test_split_index = setup["train_test_split_index"]

    x_torch = torch.from_numpy(setup["x_np"])
    y_train_torch = torch.from_numpy(setup["y_np"][:train_test_split_index])

    with torch.no_grad():
        out_torch = setup["torch_model"](
            (x_torch.unsqueeze(0), y_train_torch.unsqueeze(0)), train_test_split_index
        ).squeeze(0)
    pred_torch = np.argmax(out_torch.cpu().numpy(), axis=-1)

    x_jax = jnp.array(setup["x_np"])
    y_jax = jnp.array(setup["y_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < train_test_split_index

    out_jax = setup["jax_model"](x_jax, y_jax, train_mask=train_mask_jax)
    out_jax_test = np.array(out_jax)[train_test_split_index:]
    pred_jax = np.argmax(out_jax_test, axis=-1)

    np.testing.assert_array_equal(pred_jax, pred_torch)


def test_full_model_logits_detailed(full_model_setup: dict) -> None:
    """Test detailed logit comparison for first test sample."""
    setup = full_model_setup
    train_test_split_index = setup["train_test_split_index"]

    x_torch = torch.from_numpy(setup["x_np"])
    y_train_torch = torch.from_numpy(setup["y_np"][:train_test_split_index])

    with torch.no_grad():
        out_torch = setup["torch_model"](
            (x_torch.unsqueeze(0), y_train_torch.unsqueeze(0)), train_test_split_index
        ).squeeze(0)
    out_torch_np = out_torch.cpu().numpy()

    x_jax = jnp.array(setup["x_np"])
    y_jax = jnp.array(setup["y_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < train_test_split_index

    out_jax = setup["jax_model"](x_jax, y_jax, train_mask=train_mask_jax)
    out_jax_test = np.array(out_jax)[train_test_split_index:]

    np.testing.assert_allclose(out_jax_test[0], out_torch_np[0], atol=1e-3)

"""Numerical validation tests for FeatureEncoder: JAX vs PyTorch."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import torch

from model import FeatureEncoder as JAXFeatureEncoder
from torch_impl.model import FeatureEncoder as TorchFeatureEncoder


def _copy_torch_linear_to_jax(jax_linear: eqx.nn.Linear, torch_linear: torch.nn.Linear) -> eqx.nn.Linear:
    """Copy weights from PyTorch Linear to JAX/Equinox Linear.

    Args:
        jax_linear: Equinox Linear layer to copy weights into.
        torch_linear: PyTorch Linear layer to copy weights from.

    Returns:
        New Equinox Linear layer with copied weights.
    """
    weight = torch_linear.weight.detach().numpy()
    bias = torch_linear.bias.detach().numpy() if torch_linear.bias is not None else None

    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, weight)
    if bias is not None:
        jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, bias)
    return jax_linear


@pytest.fixture
def feature_encoder_setup() -> dict:
    """Create matched JAX and PyTorch feature encoders with test data.

    Returns:
        Dictionary containing encoders, test data, and configuration.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    num_rows = 10
    num_features = 3
    train_test_split_index = 7
    embedding_size = 16

    x_np = np.random.randn(num_rows, num_features).astype(np.float32)

    torch_enc = TorchFeatureEncoder(embedding_size)
    torch_enc.eval()

    key = jr.PRNGKey(0)
    jax_enc = JAXFeatureEncoder(embedding_size, key=key)
    jax_enc = eqx.tree_at(
        lambda m: m.linear_layer,
        jax_enc,
        _copy_torch_linear_to_jax(jax_enc.linear_layer, torch_enc.linear_layer),
    )

    return {
        "torch_enc": torch_enc,
        "jax_enc": jax_enc,
        "x_np": x_np,
        "num_rows": num_rows,
        "train_test_split_index": train_test_split_index,
    }


def test_feature_encoder_output_shape(feature_encoder_setup: dict) -> None:
    """Test that JAX and PyTorch feature encoders produce same output shapes."""
    setup = feature_encoder_setup
    x_torch = torch.from_numpy(setup["x_np"])
    x_jax = jnp.array(setup["x_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < setup["train_test_split_index"]

    with torch.no_grad():
        out_torch = setup["torch_enc"](x_torch.unsqueeze(0), setup["train_test_split_index"]).squeeze(0)

    out_jax = setup["jax_enc"](x_jax, train_mask_jax)

    assert out_torch.shape == out_jax.shape


def test_feature_encoder_numerical_match(feature_encoder_setup: dict) -> None:
    """Test that JAX and PyTorch feature encoders produce numerically matching outputs."""
    setup = feature_encoder_setup
    x_torch = torch.from_numpy(setup["x_np"])
    x_jax = jnp.array(setup["x_np"])
    train_mask_jax = jnp.arange(setup["num_rows"]) < setup["train_test_split_index"]

    with torch.no_grad():
        out_torch = setup["torch_enc"](x_torch.unsqueeze(0), setup["train_test_split_index"]).squeeze(0)

    out_jax = setup["jax_enc"](x_jax, train_mask_jax)

    out_torch_np = out_torch.numpy()
    out_jax_np = np.array(out_jax)

    np.testing.assert_allclose(out_jax_np, out_torch_np, atol=1e-5)


def test_feature_encoder_normalization_statistics(feature_encoder_setup: dict) -> None:
    """Test that normalization statistics (mean, std) match between implementations."""
    setup = feature_encoder_setup
    x_np = setup["x_np"]
    train_test_split_index = setup["train_test_split_index"]

    x_torch = torch.from_numpy(x_np)
    x_expanded_torch = x_torch.unsqueeze(0).unsqueeze(-1)
    x_train_torch = x_expanded_torch[:, :train_test_split_index]
    mean_torch = torch.mean(x_train_torch, dim=1, keepdims=True)
    std_torch = torch.std(x_train_torch, dim=1, keepdims=True) + 1e-20

    x_jax = jnp.array(x_np)
    x_expanded_jax = jnp.expand_dims(x_jax, axis=-1)
    train_mask_jax = jnp.arange(setup["num_rows"]) < train_test_split_index
    mask_3d = train_mask_jax[:, None, None]
    mean_jax = jnp.mean(x_expanded_jax, axis=0, keepdims=True, where=mask_3d)
    std_jax = jnp.std(x_expanded_jax, axis=0, keepdims=True, where=mask_3d, ddof=1) + 1e-20

    np.testing.assert_allclose(
        np.array(mean_jax).flatten()[:3],
        mean_torch.numpy().flatten()[:3],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(std_jax).flatten()[:3],
        std_torch.numpy().flatten()[:3],
        atol=1e-5,
    )

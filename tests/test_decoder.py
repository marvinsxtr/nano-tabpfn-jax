"""Numerical validation tests for Decoder: JAX vs PyTorch."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from model import Decoder as JAXDecoder
from torch_impl.model import Decoder as TorchDecoder


def _copy_torch_linear_to_jax(jax_linear: eqx.nn.Linear, torch_linear: torch.nn.Linear) -> eqx.nn.Linear:
    """Copy weights and biases from PyTorch Linear to JAX Linear.

    Args:
        jax_linear: Equinox Linear layer to copy weights into.
        torch_linear: PyTorch Linear layer to copy weights from.

    Returns:
        New Equinox Linear layer with copied weights.
    """
    weight = torch_linear.weight.detach().cpu().numpy()
    bias = torch_linear.bias.detach().cpu().numpy()

    jax_linear = eqx.tree_at(lambda m: m.weight, jax_linear, jnp.array(weight))
    jax_linear = eqx.tree_at(lambda m: m.bias, jax_linear, jnp.array(bias))
    return jax_linear


@pytest.fixture
def decoder_setup() -> dict:
    """Create matched JAX and PyTorch decoders with test data.

    Returns:
        Dictionary containing decoders, test data, and configuration.
    """
    np.random.seed(42)

    embedding_size = 16
    mlp_hidden_size = 32
    num_outputs = 2
    num_test_rows = 5

    x_np = np.random.randn(num_test_rows, embedding_size).astype(np.float32)

    torch_dec = TorchDecoder(embedding_size, mlp_hidden_size, num_outputs)
    torch_dec.eval()

    key = jax.random.PRNGKey(0)
    jax_dec = JAXDecoder(embedding_size, mlp_hidden_size, num_outputs, key=key)

    jax_dec = eqx.tree_at(
        lambda m: m.linear1,
        jax_dec,
        _copy_torch_linear_to_jax(jax_dec.linear1, torch_dec.linear1),
    )
    jax_dec = eqx.tree_at(
        lambda m: m.linear2,
        jax_dec,
        _copy_torch_linear_to_jax(jax_dec.linear2, torch_dec.linear2),
    )

    return {
        "torch_dec": torch_dec,
        "jax_dec": jax_dec,
        "x_np": x_np,
        "embedding_size": embedding_size,
        "num_outputs": num_outputs,
    }


def test_decoder_output_shape(decoder_setup: dict) -> None:
    """Test that JAX and PyTorch decoders produce same output shapes."""
    setup = decoder_setup
    x_torch = torch.from_numpy(setup["x_np"])
    x_jax = jnp.array(setup["x_np"])

    with torch.no_grad():
        out_torch = setup["torch_dec"](x_torch)

    out_jax = setup["jax_dec"](x_jax)

    assert out_torch.shape == out_jax.shape


def test_decoder_numerical_match(decoder_setup: dict) -> None:
    """Test that JAX and PyTorch decoders produce numerically matching outputs."""
    setup = decoder_setup
    x_torch = torch.from_numpy(setup["x_np"])
    x_jax = jnp.array(setup["x_np"])

    with torch.no_grad():
        out_torch = setup["torch_dec"](x_torch)

    out_jax = setup["jax_dec"](x_jax)

    out_torch_np = out_torch.cpu().numpy()
    out_jax_np = np.array(out_jax)

    np.testing.assert_allclose(out_jax_np, out_torch_np, atol=1e-3)


def test_decoder_output_dimensions(decoder_setup: dict) -> None:
    """Test that decoder outputs have correct number of output classes."""
    setup = decoder_setup
    x_jax = jnp.array(setup["x_np"])

    out_jax = setup["jax_dec"](x_jax)

    assert out_jax.shape[-1] == setup["num_outputs"]

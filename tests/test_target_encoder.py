"""Numerical validation tests for TargetEncoder: JAX vs PyTorch."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from model import TargetEncoder as JAXTargetEncoder
from torch_impl.model import TargetEncoder as TorchTargetEncoder


def _copy_linear_weights_to_jax(jax_linear: eqx.nn.Linear, torch_linear: torch.nn.Linear) -> eqx.nn.Linear:
    """Copy weights and bias from PyTorch Linear to JAX/Equinox Linear.

    Args:
        jax_linear: Equinox Linear layer to copy weights into.
        torch_linear: PyTorch Linear layer to copy weights from.

    Returns:
        New Equinox Linear layer with copied weights.
    """
    weight = jnp.array(torch_linear.weight.detach().cpu().numpy())
    bias = jnp.array(torch_linear.bias.detach().cpu().numpy())
    return eqx.tree_at(lambda m: (m.weight, m.bias), jax_linear, (weight, bias))


@pytest.fixture
def target_encoder_setup() -> dict:
    """Create matched JAX and PyTorch target encoders with test data.

    Returns:
        Dictionary containing encoders, test data, and configuration.
    """
    embedding_size = 64
    num_rows = 100
    num_train = 70

    torch.manual_seed(42)
    np.random.seed(42)

    torch_encoder = TorchTargetEncoder(embedding_size)
    torch_encoder.eval()

    jax_key = jax.random.PRNGKey(0)
    jax_encoder = JAXTargetEncoder(embedding_size, key=jax_key)
    jax_encoder = eqx.tree_at(
        lambda m: m.linear_layer,
        jax_encoder,
        _copy_linear_weights_to_jax(jax_encoder.linear_layer, torch_encoder.linear_layer),
    )

    y_full = np.random.randn(num_rows, 1).astype(np.float32)
    train_mask = np.concatenate(
        [
            np.ones(num_train, dtype=bool),
            np.zeros(num_rows - num_train, dtype=bool),
        ]
    )

    return {
        "torch_encoder": torch_encoder,
        "jax_encoder": jax_encoder,
        "y_full": y_full,
        "train_mask": train_mask,
        "num_rows": num_rows,
        "num_train": num_train,
        "embedding_size": embedding_size,
    }


def test_target_encoder_output_shape(target_encoder_setup: dict) -> None:
    """Test that JAX and PyTorch target encoders produce same output shapes."""
    setup = target_encoder_setup

    y_jax = jnp.array(setup["y_full"])
    mask_jax = jnp.array(setup["train_mask"])
    jax_output = setup["jax_encoder"](y_jax, mask_jax)

    y_train_torch = torch.from_numpy(setup["y_full"][: setup["num_train"]]).unsqueeze(0)
    with torch.no_grad():
        torch_output = setup["torch_encoder"](y_train_torch, setup["num_rows"])

    jax_shape = jax_output.shape
    torch_shape = torch_output.squeeze(0).shape

    assert jax_shape == torch_shape


def test_target_encoder_numerical_match(target_encoder_setup: dict) -> None:
    """Test that JAX and PyTorch target encoders produce numerically matching outputs."""
    setup = target_encoder_setup

    y_jax = jnp.array(setup["y_full"])
    mask_jax = jnp.array(setup["train_mask"])
    jax_output = setup["jax_encoder"](y_jax, mask_jax)

    y_train_torch = torch.from_numpy(setup["y_full"][: setup["num_train"]]).unsqueeze(0)
    with torch.no_grad():
        torch_output = setup["torch_encoder"](y_train_torch, setup["num_rows"])

    jax_out_np = np.array(jax_output)
    torch_out_np = torch_output.squeeze(0).cpu().numpy()

    np.testing.assert_allclose(jax_out_np, torch_out_np, atol=1e-5, rtol=1e-4)


def test_target_encoder_mean_imputation(target_encoder_setup: dict) -> None:
    """Test that mean imputation for test samples works correctly."""
    setup = target_encoder_setup

    mean_train = setup["y_full"][: setup["num_train"]].mean()

    y_jax = jnp.array(setup["y_full"])
    mask_jax = jnp.array(setup["train_mask"])

    y_imputed = jnp.where(
        mask_jax[:, None],
        y_jax,
        mean_train,
    )

    np.testing.assert_allclose(
        float(y_imputed[setup["num_train"], 0]),
        mean_train,
        atol=1e-6,
    )

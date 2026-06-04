from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from models.common import NanoTabPFNClassifier as BaseClassifier


class FeatureEncoder(eqx.Module):
    """Encodes features to embeddings."""

    linear_layer: eqx.nn.Linear

    def __init__(self, embedding_size: int, *, key: PRNGKeyArray) -> None:
        """Creates the linear layer that we will use to embed our features."""
        self.linear_layer = eqx.nn.Linear(1, embedding_size, key=key)

    def __call__(
        self, x: Float[Array, "num_rows num_features"], train_mask: Float[Array, " num_rows"]
    ) -> Float[Array, "num_rows num_features embedding_size"]:
        """Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: a tensor of shape (num_rows, num_features)
            train_mask: boolean mask of shape (num_rows,) indicating training rows
        Returns:
            a tensor of shape (num_rows, num_features, embedding_size), representing
            the embeddings of the features
        """
        x = jnp.expand_dims(x, axis=-1)  # (num_rows, num_features, 1)

        mask_3d = train_mask[:, None, None]  # (num_rows, 1, 1)

        mean = jnp.mean(x, axis=0, keepdims=True, where=mask_3d)
        std = jnp.std(x, axis=0, keepdims=True, where=mask_3d, ddof=1) + 1e-20

        x = (x - mean) / std
        x = jnp.clip(x, min=-100, max=100)
        return jax.vmap(jax.vmap(self.linear_layer))(x)


class TargetEncoder(eqx.Module):
    """Encodes target values using a linear embedding layer with mean imputation."""

    linear_layer: eqx.nn.Linear

    def __init__(self, embedding_size: int, *, key: PRNGKeyArray) -> None:
        """Creates the linear layer that we will use to embed our targets.

        Args:
            embedding_size: Dimension of the output embeddings.
            key: Random key for layer initialization.
        """
        self.linear_layer = eqx.nn.Linear(1, embedding_size, key=key)

    def __call__(
        self, y: Float[Array, "num_rows 1"], train_mask: Bool[Array, " num_rows"]
    ) -> Float[Array, "num_rows 1 embedding_size"]:
        """Imputes non-training targets with per-sample mean and embeds them.

        Args:
            y: Target values of shape (num_rows, 1).
            train_mask: Boolean mask of shape (num_rows,) indicating training rows.

        Returns:
            Target embeddings of shape (num_rows, 1, embedding_size).
        """
        mask_2d = train_mask[:, None]  # (num_rows, 1)
        mean = jnp.mean(y, axis=0, keepdims=True, where=mask_2d)

        y_imputed = jnp.where(mask_2d, y, mean)
        y_imputed = y_imputed[:, None, :]  # (num_rows, 1, 1)

        return jax.vmap(jax.vmap(self.linear_layer))(y_imputed)  # (num_rows, 1, embedding_size)


class TransformerEncoderLayer(eqx.Module):
    """Modified transformer encoder layer with self-attention between datapoints and features."""

    embedding_size: int
    nhead: int
    mlp_hidden_size: int

    self_attn_features: eqx.nn.MultiheadAttention
    self_attn_datapoints: eqx.nn.MultiheadAttention

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm

    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 5)

        self.embedding_size = embedding_size
        self.nhead = nhead
        self.mlp_hidden_size = mlp_hidden_size

        self.self_attn_features = eqx.nn.MultiheadAttention(
            num_heads=nhead,
            query_size=embedding_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=keys[0],
        )

        self.self_attn_datapoints = eqx.nn.MultiheadAttention(
            num_heads=nhead,
            query_size=embedding_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=keys[1],
        )

        self.linear1 = eqx.nn.Linear(embedding_size, mlp_hidden_size, key=keys[2])
        self.linear2 = eqx.nn.Linear(mlp_hidden_size, embedding_size, key=keys[3])

        self.norm1 = eqx.nn.LayerNorm(embedding_size)
        self.norm2 = eqx.nn.LayerNorm(embedding_size)
        self.norm3 = eqx.nn.LayerNorm(embedding_size)

    def __call__(
        self,
        src: Float[Array, "num_rows num_features_plus_target embedding_size"],
        train_mask: Float[Array, " num_rows"],
    ) -> Float[Array, "num_rows num_features_plus_target embedding_size"]:
        """Applies multihead attention and MLP to the input embeddings.

        The attention mechanism is special:
        - Within the training set: datapoints attend to each other
        - Within the test set: datapoints only attend to the training set (no test-to-test attention)

        Args:
            src: (num_rows, num_features+1, embedding_size)
            train_mask: boolean mask of shape (num_rows,) indicating training rows

        Returns:
            (num_rows, num_features+1, embedding_size)
        """
        src_features = jax.vmap(self.self_attn_features)(src, src, src) + src
        src = jax.vmap(jax.vmap(self.norm1))(src_features)

        src = jnp.transpose(src, (1, 0, 2))

        num_rows = src.shape[1]
        mask = jnp.broadcast_to(train_mask, (num_rows, num_rows))

        masked_mha = partial(self.self_attn_datapoints, mask=mask)
        src = jax.vmap(masked_mha)(src, src, src) + src

        src = jnp.transpose(src, (1, 0, 2))

        src = jax.vmap(jax.vmap(self.norm2))(src)

        src = jax.vmap(jax.vmap(lambda x: self.linear2(jax.nn.gelu(self.linear1(x), approximate=False))))(src) + src
        src = jax.vmap(jax.vmap(self.norm3))(src)

        return src


class Decoder(eqx.Module):
    """Decoder that converts embeddings to class logits."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int, *, key: PRNGKeyArray) -> None:
        """Initializes the linear layers for use in the forward."""
        keys = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(embedding_size, mlp_hidden_size, key=keys[0])
        self.linear2 = eqx.nn.Linear(mlp_hidden_size, num_outputs, key=keys[1])

    def __call__(self, x: Float[Array, "num_rows embedding_size"]) -> Float[Array, "num_rows num_outputs"]:
        """Applies an MLP to the embeddings to get the logits

        Args:
            x: a tensor of shape (num_rows, embedding_size)

        Returns:
            a tensor of shape (num_rows, num_outputs)
        """
        return jax.vmap(lambda x: self.linear2(jax.nn.gelu(self.linear1(x), approximate=False)))(x)


class NanoTabPFNModel(eqx.Module):
    """Main model combining feature/target encoders, transformer blocks, and decoder."""

    feature_encoder: FeatureEncoder
    target_encoder: TargetEncoder
    transformer_blocks: list[TransformerEncoderLayer]
    decoder: Decoder

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initializes the feature/target encoder, transformer stack and decoder."""
        keys = jax.random.split(key, num_layers + 3)

        self.feature_encoder = FeatureEncoder(embedding_size, key=keys[0])
        self.target_encoder = TargetEncoder(embedding_size, key=keys[1])

        self.transformer_blocks = [
            TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size, key=k)
            for k in keys[2 : 2 + num_layers]
        ]

        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs, key=keys[-1])

    def __call__(
        self,
        x_src: Float[Array, "num_rows num_features"],
        y_src: Float[Array, "num_rows 1"],
        train_mask: Float[Array, " num_rows"],
    ) -> Float[Array, "test_size num_outputs"]:
        """Forward pass through the model.

        Args:
            x_src: features of shape (num_rows, num_features)
            y_src: targets of shape (num_rows, 1)
            train_mask: boolean mask of shape (num_rows,) indicating training rows
        Returns:
            logits of shape (test_size, num_outputs) for test datapoints only
        """
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src[..., None]

        x_src = self.feature_encoder(x_src, train_mask)
        y_src = self.target_encoder(y_src, train_mask)

        src = jnp.concatenate([x_src, y_src], axis=1)

        for block in self.transformer_blocks:
            src = block(src, train_mask=train_mask)

        output = self.decoder(src[:, -1, :])

        test_mask = (~train_mask)[:, None]  # (num_rows, 1)
        output = output * test_mask

        return output


@eqx.filter_jit
def predict(
    model: NanoTabPFNModel,
    x: Float[Array, "num_rows num_features"],
    y: Float[Array, "num_rows 1"],
    train_mask: Float[Array, " num_rows"],
) -> Float[Array, "num_rows num_outputs"]:
    """JIT-compiled prediction function for the NanoTabPFNModel."""
    return model(x, y, train_mask=train_mask)


class NanoTabPFNClassifier(BaseClassifier):
    """scikit-learn-like interface for the v2 JAX model."""

    def __init__(self, model: NanoTabPFNModel) -> None:
        super().__init__(model, predict)

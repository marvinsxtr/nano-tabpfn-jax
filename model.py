from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray


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

    # Self-attention layers
    self_attn_datapoints_q: eqx.nn.Linear
    self_attn_datapoints_k: eqx.nn.Linear
    self_attn_datapoints_v: eqx.nn.Linear
    self_attn_datapoints_out: eqx.nn.Linear

    self_attn_features_q: eqx.nn.Linear
    self_attn_features_k: eqx.nn.Linear
    self_attn_features_v: eqx.nn.Linear
    self_attn_features_out: eqx.nn.Linear

    # MLP layers
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    # Layer norms
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm

    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 11)

        self.embedding_size = embedding_size
        self.nhead = nhead
        self.mlp_hidden_size = mlp_hidden_size

        # Self-attention between datapoints
        self.self_attn_datapoints_q = eqx.nn.Linear(embedding_size, embedding_size, key=keys[0])
        self.self_attn_datapoints_k = eqx.nn.Linear(embedding_size, embedding_size, key=keys[1])
        self.self_attn_datapoints_v = eqx.nn.Linear(embedding_size, embedding_size, key=keys[2])
        self.self_attn_datapoints_out = eqx.nn.Linear(embedding_size, embedding_size, key=keys[3])

        # Self-attention between features
        self.self_attn_features_q = eqx.nn.Linear(embedding_size, embedding_size, key=keys[4])
        self.self_attn_features_k = eqx.nn.Linear(embedding_size, embedding_size, key=keys[5])
        self.self_attn_features_v = eqx.nn.Linear(embedding_size, embedding_size, key=keys[6])
        self.self_attn_features_out = eqx.nn.Linear(embedding_size, embedding_size, key=keys[7])

        # MLP
        self.linear1 = eqx.nn.Linear(embedding_size, mlp_hidden_size, key=keys[8])
        self.linear2 = eqx.nn.Linear(mlp_hidden_size, embedding_size, key=keys[9])

        # Layer norms
        self.norm1 = eqx.nn.LayerNorm(embedding_size)
        self.norm2 = eqx.nn.LayerNorm(embedding_size)
        self.norm3 = eqx.nn.LayerNorm(embedding_size)

    def _multihead_attention_features(
        self,
        query: Float[Array, "seq_len embed_dim"],
        key: Float[Array, "seq_len embed_dim"],
        value: Float[Array, "seq_len embed_dim"],
    ) -> Float[Array, "seq_len embed_dim"]:
        """Compute multi-head attention for features."""
        seq_len, embed_dim = query.shape
        head_dim = embed_dim // self.nhead

        q = jax.vmap(self.self_attn_features_q)(query)
        k = jax.vmap(self.self_attn_features_k)(key)
        v = jax.vmap(self.self_attn_features_v)(value)

        # Reshape for multi-head: (seq_len, nhead, head_dim)
        q = q.reshape(seq_len, self.nhead, head_dim)
        k = k.reshape(key.shape[0], self.nhead, head_dim)
        v = v.reshape(value.shape[0], self.nhead, head_dim)

        attn_out = jax.nn.dot_product_attention(q, k, v, mask=None, implementation="xla")

        attn_out = attn_out.reshape(seq_len, embed_dim)

        return jax.vmap(self.self_attn_features_out)(attn_out)

    def _multihead_attention_datapoints(
        self,
        query: Float[Array, "seq_len embed_dim"],
        key: Float[Array, "seq_len embed_dim"],
        value: Float[Array, "seq_len embed_dim"],
        mask: Float[Array, "..."] | None = None,
    ) -> Float[Array, "seq_len embed_dim"]:
        """Compute multi-head attention for datapoints."""
        seq_len, embed_dim = query.shape
        head_dim = embed_dim // self.nhead

        q = jax.vmap(self.self_attn_datapoints_q)(query)
        k = jax.vmap(self.self_attn_datapoints_k)(key)
        v = jax.vmap(self.self_attn_datapoints_v)(value)

        # Reshape for multi-head: (seq_len, nhead, head_dim)
        q = q.reshape(seq_len, self.nhead, head_dim)
        k = k.reshape(key.shape[0], self.nhead, head_dim)
        v = v.reshape(value.shape[0], self.nhead, head_dim)

        attn_out = jax.nn.dot_product_attention(q, k, v, mask=mask, implementation="xla")

        attn_out = attn_out.reshape(seq_len, embed_dim)

        return jax.vmap(self.self_attn_datapoints_out)(attn_out)

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
        src_features = jax.vmap(self._multihead_attention_features)(src, src, src) + src
        src = jax.vmap(jax.vmap(self.norm1))(src_features)

        src = jnp.transpose(src, (1, 0, 2))

        mask = train_mask[None, :]  # (1, rows_size) - broadcasts to (nhead, rows, rows)

        mha = partial(self._multihead_attention_datapoints, mask=mask)
        src_attended = jax.vmap(mha)(src, src, src)
        src = src_attended + src

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
        # Ensure y_src has the right shape
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src[..., None]

        x_src = self.feature_encoder(x_src, train_mask)
        y_src = self.target_encoder(y_src, train_mask)

        src = jnp.concatenate([x_src, y_src], axis=1)

        for block in self.transformer_blocks:
            src = block(src, train_mask=train_mask)

        output = self.decoder(src[:, -1, :])

        # Mask out train predictions, keep only test predictions
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


class NanoTabPFNClassifier:
    """scikit-learn like interface for JAX model."""

    def __init__(self, model: NanoTabPFNModel) -> None:
        self.model = eqx.nn.inference_mode(model, value=True)
        self.X_train = None
        self.y_train = None
        self.num_classes = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Stores X_train and y_train for later use, also computes the highest class
        number occurring in num_classes.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train))) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Creates (x,y), runs it through our JAX Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities.
        """
        x = jnp.concatenate((self.X_train, X_test))

        # Pad features to fixed size (10) to avoid recompilation
        num_features = x.shape[1]
        if x.shape[1] < 10:
            padding = jnp.zeros((x.shape[0], 10 - num_features))
            x = jnp.concatenate([x, padding], axis=1)

        # Pad targets with mean imputation for test positions
        mean = self.y_train.mean()  # Scalar mean of training targets
        num_test = len(X_test)
        padding = np.full(num_test, mean)  # (num_test,) filled with mean
        y = jnp.concatenate([self.y_train, padding])  # (num_total,)

        num_train = len(self.X_train)
        train_mask = jnp.arange(len(x)) < num_train

        out = predict(self.model, x, y, train_mask=train_mask)

        # Extract only test predictions (train predictions are zeroed out)
        out = out[num_train:]

        # Slice to keep only valid classes
        out = out[:, : self.num_classes]

        probabilities = jax.nn.softmax(out, axis=1)

        return np.array(probabilities)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the test data."""
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

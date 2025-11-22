import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray


class FeatureEncoder(eqx.Module):
    """Encodes features to embeddings."""

    linear_layer: eqx.nn.Linear

    def __init__(self, embedding_size: int, *, key: PRNGKeyArray) -> None:
        """Creates the linear layer that we will use to embed our features."""
        self.linear_layer = eqx.nn.Linear(1, embedding_size, key=key)

    def __call__(
        self, x: Float[Array, "batch_size num_rows num_features"], train_test_split_index: int
    ) -> Float[Array, "batch_size num_rows num_features embedding_size"]:
        """Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: the number of datapoints in X_train
        Returns:
            a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
            the embeddings of the features
        """
        x = x[..., None]  # (batch_size, num_rows, num_features, 1)
        mean = jnp.mean(x[:, :train_test_split_index], axis=1, keepdims=True)
        std = jnp.std(x[:, :train_test_split_index], axis=1, keepdims=True) + 1e-20
        x = (x - mean) / std
        x = jnp.clip(x, min=-100, max=100)
        return jax.vmap(jax.vmap(jax.vmap(self.linear_layer)))(x)


class TargetEncoder(eqx.Module):
    """Encodes targets to embeddings."""

    linear_layer: eqx.nn.Linear

    def __init__(self, embedding_size: int, *, key: PRNGKeyArray):
        """Creates the linear layer that we will use to embed our targets."""
        self.linear_layer = eqx.nn.Linear(1, embedding_size, key=key)

    def __call__(
        self, y_train: Float[Array, "batch_size num_train_datapoints 1"], num_rows: int
    ) -> Float[Array, "batch_size num_rows 1 embedding_size"]:
        """Pads up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer

        Args:
            y_train: a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: the full length of y
        Returns:
            a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
            the embeddings of the targets
        """
        mean = jnp.mean(y_train, axis=1, keepdims=True)
        padding = jnp.repeat(mean, num_rows - y_train.shape[1], axis=1)
        y = jnp.concatenate([y_train, padding], axis=1)
        y = y[..., None]  # (batch_size, num_rows, 1, 1)
        return jax.vmap(jax.vmap(jax.vmap(self.linear_layer)))(y)


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

    def _multihead_attention(
        self,
        q_proj: eqx.nn.Linear,
        k_proj: eqx.nn.Linear,
        v_proj: eqx.nn.Linear,
        out_proj: eqx.nn.Linear,
        query: Float[Array, "seq_len embed_dim"],
        key: Float[Array, "seq_len embed_dim"],
        value: Float[Array, "seq_len embed_dim"],
    ) -> Float[Array, "seq_len embed_dim"]:
        """Compute multi-head attention."""
        seq_len, embed_dim = query.shape
        head_dim = embed_dim // self.nhead

        # Project
        q = jax.vmap(q_proj)(query)
        k = jax.vmap(k_proj)(key)
        v = jax.vmap(v_proj)(value)

        # Reshape for multi-head: (seq_len, nhead, head_dim)
        q = q.reshape(seq_len, self.nhead, head_dim)
        k = k.reshape(key.shape[0], self.nhead, head_dim)
        v = v.reshape(value.shape[0], self.nhead, head_dim)

        # Attention
        attn_out = jax.nn.dot_product_attention(q, k, v, implementation="xla")

        # Reshape back
        attn_out = attn_out.reshape(seq_len, embed_dim)

        # Output projection
        return jax.vmap(out_proj)(attn_out)

    def __call__(
        self, src: Float[Array, "batch_size num_rows num_features embedding_size"], train_test_split_index: int
    ) -> Float[Array, "batch_size num_rows num_features embedding_size"]:
        """Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                 for all the cells in the table
            train_test_split_index: the length of X_train
        Returns
            a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape

        # Attention between features
        # Reshape to (batch_size * rows_size, col_size, embedding_size)
        src_features = src.reshape(batch_size * rows_size, col_size, embedding_size)

        # Apply attention for each batch*row
        def attn_features(x):
            attn_out = self._multihead_attention(
                self.self_attn_features_q,
                self.self_attn_features_k,
                self.self_attn_features_v,
                self.self_attn_features_out,
                x,
                x,
                x,
            )
            return attn_out + x

        src_features = jax.vmap(attn_features)(src_features)
        src = src_features.reshape(batch_size, rows_size, col_size, embedding_size)
        src = jax.vmap(jax.vmap(jax.vmap(self.norm1)))(src)

        # Attention between datapoints
        # Reshape to (batch_size, col_size, rows_size, embedding_size)
        src = jnp.transpose(src, (0, 2, 1, 3))
        src_datapoints = src.reshape(batch_size * col_size, rows_size, embedding_size)

        # Training data attends to itself, test data attends to training data
        def attn_datapoints(x):
            # Training part attends to itself
            x_train = x[:train_test_split_index]
            attn_train = self._multihead_attention(
                self.self_attn_datapoints_q,
                self.self_attn_datapoints_k,
                self.self_attn_datapoints_v,
                self.self_attn_datapoints_out,
                x_train,
                x_train,
                x_train,
            )

            # Test part attends to training part
            x_test = x[train_test_split_index:]
            attn_test = self._multihead_attention(
                self.self_attn_datapoints_q,
                self.self_attn_datapoints_k,
                self.self_attn_datapoints_v,
                self.self_attn_datapoints_out,
                x_test,
                x_train,
                x_train,
            )

            return jnp.concatenate([attn_train, attn_test], axis=0) + x

        src_datapoints = jax.vmap(attn_datapoints)(src_datapoints)
        src = src_datapoints.reshape(batch_size, col_size, rows_size, embedding_size)
        src = jnp.transpose(src, (0, 2, 1, 3))
        src = jax.vmap(jax.vmap(jax.vmap(self.norm2)))(src)

        # MLP after attention
        def mlp(x):
            return jax.vmap(self.linear2)(jax.nn.gelu(jax.vmap(self.linear1)(x)))

        src = jax.vmap(jax.vmap(mlp))(src) + src
        src = jax.vmap(jax.vmap(jax.vmap(self.norm3)))(src)

        return src


class Decoder(eqx.Module):
    """Decoder that converts embeddings to class logits."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int, *, key: PRNGKeyArray):
        """Initializes the linear layers for use in the forward."""
        keys = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(embedding_size, mlp_hidden_size, key=keys[0])
        self.linear2 = eqx.nn.Linear(mlp_hidden_size, num_outputs, key=keys[1])

    def __call__(
        self, x: Float[Array, "batch_size num_rows embedding_size"]
    ) -> Float[Array, "batch_size num_rows num_outputs"]:
        """Applies an MLP to the embeddings to get the logits

        Args:
            x: a tensor of shape (batch_size, num_rows, embedding_size)

        Returns:
            a tensor of shape (batch_size, num_rows, num_outputs)
        """

        def mlp(x):
            return jax.vmap(self.linear2)(jax.nn.gelu(jax.vmap(self.linear1)(x)))

        return jax.vmap(mlp)(x)


class NanoTabPFNModel(eqx.Module):
    """Main model combining feature/target encoders, transformer blocks, and decoder."""

    feature_encoder: FeatureEncoder
    target_encoder: TargetEncoder
    transformer_blocks: list
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
    ):
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
        src: tuple[Float[Array, "batch_size num_rows num_features"], Float[Array, "batch_size num_train"]],
        train_test_split_index: int,
    ) -> Float[Array, "batch_size num_test num_outputs"]:
        """Forward pass through the model.

        Args:
            src: tuple of (x_src, y_src) where
                 x_src: features of shape (batch_size, num_rows, num_features)
                 y_src: targets of shape (batch_size, num_train_datapoints) or (batch_size, num_train_datapoints, 1)
            train_test_split_index: number of training datapoints
        Returns:
            logits of shape (batch_size, num_test_datapoints, num_outputs)
        """
        x_src, y_src = src

        # Ensure y_src has the right shape
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src[..., None]

        # Encode features: (batch, rows, features) -> (batch, rows, features, embed)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]

        # Encode targets: (batch, train_rows, 1) -> (batch, rows, 1, embed)
        y_src = self.target_encoder(y_src, num_rows)

        # Concatenate: (batch, rows, features+1, embed)
        src = jnp.concatenate([x_src, y_src], axis=2)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)

        # Select target embeddings: (batch, num_test, 1, embed)
        output = src[:, train_test_split_index:, -1, :]

        # Decode to logits: (batch, num_test, num_outputs)
        output = self.decoder(output)

        return output


class NanoTabPFNClassifier:
    """scikit-learn like interface for JAX model."""

    def __init__(self, model: NanoTabPFNModel):
        self.model = model
        self.X_train = None
        self.y_train = None
        self.num_classes = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Stores X_train and y_train for later use, also computes the highest class number occurring in num_classes."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train))) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Creates (x,y), runs it through our JAX Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities.
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train

        x = jnp.array(x)[None, ...]  # Add batch dimension
        y = jnp.array(y)[None, ...]  # Add batch dimension

        out = self.model((x, y), train_test_split_index=len(self.X_train))
        out = out[0]  # Remove batch dimension

        # Cut off classes that didn't appear in training
        out = out[:, : self.num_classes]

        # Apply softmax
        probabilities = jax.nn.softmax(out, axis=1)

        return np.array(probabilities)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

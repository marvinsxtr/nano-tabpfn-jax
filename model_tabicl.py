"""JAX/Equinox reimplementation of nanoTabICL (soda-inria/nanotabicl).

This mirrors the reference PyTorch model in ``torch_impl/model_tabicl.py`` closely
enough to be numerically validated against it after copying weights. The model
operates on a single (unbatched) dataset, following the convention used by the
JAX nanoTabPFN implementation in ``model.py``.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray


def _apply_rope(x: Float[Array, "num_heads seq head_dim"], half: int, theta: float) -> Array:
    """Applies rotary positional encoding to a per-head tensor.

    Args:
        x: Tensor of shape (num_heads, seq, head_dim).
        half: Half of the head dimension (rotation is applied on paired halves).
        theta: RoPE base frequency.

    Returns:
        The rotated tensor of the same shape as ``x``.
    """
    seq = x.shape[1]
    inv_freq = theta ** jnp.linspace(0.0, -1.0, half + 1)[:-1]  # (half,)
    pos = jnp.arange(seq)
    angles = pos[:, None] * inv_freq[None, :]  # (seq, half)
    sin, cos = jnp.sin(angles), jnp.cos(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class MLP(eqx.Module):
    """Two layer MLP with an exact GELU activation (``Linear -> GELU -> Linear``)."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, n_in: int, n_hidden: int, n_out: int, *, key: PRNGKeyArray) -> None:
        """Initializes the two linear layers.

        Args:
            n_in: Input dimension.
            n_hidden: Hidden dimension.
            n_out: Output dimension.
            key: Random key for initialization.
        """
        keys = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(n_in, n_hidden, key=keys[0])
        self.linear2 = eqx.nn.Linear(n_hidden, n_out, key=keys[1])

    def __call__(self, x: Float[Array, " n_in"]) -> Float[Array, " n_out"]:
        """Applies the MLP to a single input vector."""
        return self.linear2(jax.nn.gelu(self.linear1(x), approximate=False))


class QASSMax(eqx.Module):
    """Query-aware scalable softmax used to modulate queries before attention."""

    base_mlp: MLP
    query_mlp: MLP
    num_heads: int
    head_dim: int

    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64, *, key: PRNGKeyArray) -> None:
        """Initializes the base and query MLPs.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            n_hidden: Hidden dimension of the internal MLPs.
            key: Random key for initialization.
        """
        keys = jax.random.split(key, 2)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base_mlp = MLP(1, n_hidden, num_heads * head_dim, key=keys[0])
        self.query_mlp = MLP(head_dim, n_hidden, head_dim, key=keys[1])

    def __call__(self, q: Float[Array, "num_heads seq head_dim"], n: int) -> Array:
        """Scales the queries by a length-dependent, query-dependent factor.

        Args:
            q: Query tensor of shape (num_heads, seq, head_dim).
            n: Number of keys attended to (context length).

        Returns:
            The modulated query tensor of the same shape as ``q``.
        """
        logn = jnp.array([math.log(max(1, n))])  # (1,)
        base = self.base_mlp(logn).reshape(self.num_heads, 1, self.head_dim)
        modulation = 1.0 + jnp.tanh(jax.vmap(jax.vmap(self.query_mlp))(q))
        return base * modulation * q


class TransformerBlock(eqx.Module):
    """Pre-norm transformer block with a joint QKV projection, matching the reference.

    Optionally applies rotary positional encoding (``use_rope``) or query-aware
    scalable softmax (``ssmax``) inside the attention.
    """

    num_heads: int
    head_dim: int
    use_rope: bool
    rope_half: int
    rope_theta: float

    in_proj_weight: Array
    in_proj_bias: Array
    out_proj: eqx.nn.Linear
    mlp: MLP
    ln_attn: eqx.nn.LayerNorm
    ln_mlp: eqx.nn.LayerNorm
    ssmax_layer: QASSMax | None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        ssmax: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initializes projections, MLP, layer norms and optional attention modules.

        Args:
            embed_dim: Model / embedding dimension.
            num_heads: Number of attention heads.
            use_rope: Whether to apply rotary positional encoding.
            ssmax: Whether to apply query-aware scalable softmax.
            key: Random key for initialization.
        """
        keys = jax.random.split(key, 5)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.rope_half = self.head_dim // 2
        self.rope_theta = 100_000.0

        bound = 1.0 / math.sqrt(embed_dim)
        self.in_proj_weight = jax.random.uniform(
            keys[0], (3 * embed_dim, embed_dim), minval=-bound, maxval=bound
        )
        self.in_proj_bias = jnp.zeros((3 * embed_dim,))
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.mlp = MLP(embed_dim, embed_dim * 2, embed_dim, key=keys[2])
        self.ln_attn = eqx.nn.LayerNorm(embed_dim)
        self.ln_mlp = eqx.nn.LayerNorm(embed_dim)
        self.ssmax_layer = QASSMax(num_heads, self.head_dim, key=keys[3]) if ssmax else None

    def _attn(self, q: Float[Array, "q_len embed"], k: Float[Array, "kv_len embed"]) -> Array:
        """Computes multi-head attention where keys and values share the ``k`` input.

        Args:
            q: Query tensor of shape (q_len, embed_dim).
            k: Key/value tensor of shape (kv_len, embed_dim).

        Returns:
            Attention output of shape (q_len, embed_dim).
        """
        embed_dim = q.shape[-1]
        heads, head_dim = self.num_heads, self.head_dim

        w = self.in_proj_weight
        b = self.in_proj_bias
        q_proj = q @ w[:embed_dim].T + b[:embed_dim]
        k_proj = k @ w[embed_dim : 2 * embed_dim].T + b[embed_dim : 2 * embed_dim]
        v_proj = k @ w[2 * embed_dim :].T + b[2 * embed_dim :]

        q_h = q_proj.reshape(-1, heads, head_dim).transpose(1, 0, 2)  # (heads, q_len, head_dim)
        k_h = k_proj.reshape(-1, heads, head_dim).transpose(1, 0, 2)
        v_h = v_proj.reshape(-1, heads, head_dim).transpose(1, 0, 2)

        if self.ssmax_layer is not None:
            q_h = self.ssmax_layer(q_h, k.shape[0])
        if self.use_rope:
            q_h = _apply_rope(q_h, self.rope_half, self.rope_theta)
            k_h = _apply_rope(k_h, self.rope_half, self.rope_theta)

        scale = 1.0 / math.sqrt(head_dim)
        scores = jnp.einsum("hqd,hkd->hqk", q_h, k_h) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hqk,hkd->hqd", weights, v_h)  # (heads, q_len, head_dim)
        out = out.transpose(1, 0, 2).reshape(-1, embed_dim)
        return jax.vmap(self.out_proj)(out)

    def __call__(
        self,
        q: Float[Array, "q_len embed"],
        kv: Float[Array, "kv_len embed"] | None = None,
        q_max_idx: int | None = None,
        kv_max_idx: int | None = None,
    ) -> Array:
        """Applies pre-norm attention followed by a pre-norm MLP.

        Args:
            q: Query sequence of shape (q_len, embed_dim).
            kv: Optional key/value sequence; defaults to ``q`` (self-attention).
            q_max_idx: If set, only the first ``q_max_idx`` queries are kept.
            kv_max_idx: If set, only the first ``kv_max_idx`` keys/values are used.

        Returns:
            The updated sequence of shape (q_len or q_max_idx, embed_dim).
        """
        x = q
        q = jax.vmap(self.ln_attn)(q)
        kv = q if kv is None else jax.vmap(self.ln_attn)(kv)
        if kv_max_idx is not None:
            kv = kv[:kv_max_idx]
        if q_max_idx is not None:
            x, q = x[:q_max_idx], q[:q_max_idx]

        x = x + self._attn(q, kv)
        return x + jax.vmap(self.mlp)(jax.vmap(self.ln_mlp)(x))


class InducedTransformerBlock(eqx.Module):
    """Induced set-attention block (ISAB): rows attend to a set of inducing vectors."""

    tfm1: TransformerBlock
    tfm2: TransformerBlock
    inducing_vectors: Array

    def __init__(
        self, embed_dim: int, num_heads: int, n_inducing: int, ssmax: bool = False, *, key: PRNGKeyArray
    ) -> None:
        """Initializes the two transformer blocks and the inducing vectors.

        Args:
            embed_dim: Model / embedding dimension.
            num_heads: Number of attention heads.
            n_inducing: Number of inducing vectors.
            ssmax: Whether the first block uses query-aware scalable softmax.
            key: Random key for initialization.
        """
        keys = jax.random.split(key, 3)
        self.tfm1 = TransformerBlock(embed_dim, num_heads, ssmax=ssmax, key=keys[0])
        self.tfm2 = TransformerBlock(embed_dim, num_heads, key=keys[1])
        self.inducing_vectors = 0.02 * jax.random.normal(keys[2], (n_inducing, embed_dim))

    def __call__(
        self,
        q: Float[Array, "seq embed"],
        kv: Float[Array, "kv_len embed"] | None = None,
        q_max_idx: int | None = None,
        kv_max_idx: int | None = None,
    ) -> Array:
        """Summarizes the keys into inducing vectors, then queries attend to them.

        Args:
            q: Query sequence of shape (seq, embed_dim).
            kv: Optional key/value sequence; defaults to ``q``.
            q_max_idx: If set, only the first ``q_max_idx`` queries are kept.
            kv_max_idx: If set, only the first ``kv_max_idx`` keys/values are used.

        Returns:
            The updated query sequence.
        """
        kv_in = q if kv is None else kv
        summary = self.tfm1(self.inducing_vectors, kv_in, kv_max_idx=kv_max_idx)
        return self.tfm2(q, summary, q_max_idx=q_max_idx)


class NanoTabICLModel(eqx.Module):
    """JAX reimplementation of ``NanoTabICLv2`` for a single dataset (classification)."""

    feature_group_size: int
    n_cls_cols: int
    embed_dim: int

    x_embed: eqx.nn.Linear
    y_embed_in: eqx.nn.Embedding
    y_embed_icl: eqx.nn.Embedding
    col_blocks: list[InducedTransformerBlock]
    row_blocks: list[TransformerBlock]
    icl_blocks: list[TransformerBlock]
    row_cls_tokens: Array
    row_ln: eqx.nn.LayerNorm
    out_ln: eqx.nn.LayerNorm
    out_mlp: MLP

    def __init__(
        self,
        max_classes: int,
        out_dim: int,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        row_num_blocks: int = 3,
        icl_num_blocks: int = 12,
        col_nhead: int = 8,
        row_nhead: int = 8,
        icl_nhead: int = 8,
        feature_group_size: int = 3,
        n_cls_cols: int = 4,
        n_cls_rows: int = 128,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initializes all sub-modules following the reference architecture.

        Args:
            max_classes: Number of class-embedding entries (must exceed max label).
            out_dim: Output dimension (number of classes for classification).
            embed_dim: Per-cell embedding dimension.
            col_num_blocks: Number of column (induced) attention blocks.
            row_num_blocks: Number of row attention blocks.
            icl_num_blocks: Number of in-context-learning attention blocks.
            col_nhead: Attention heads in column blocks.
            row_nhead: Attention heads in row blocks.
            icl_nhead: Attention heads in ICL blocks.
            feature_group_size: Number of grouped features per column.
            n_cls_cols: Number of learnable CLS columns.
            n_cls_rows: Number of inducing vectors in column blocks.
            key: Random key for initialization.
        """
        keys = jax.random.split(key, 8)
        icl_dim = embed_dim * n_cls_cols

        self.feature_group_size = feature_group_size
        self.n_cls_cols = n_cls_cols
        self.embed_dim = embed_dim

        self.x_embed = eqx.nn.Linear(feature_group_size, embed_dim, key=keys[0])
        self.y_embed_in = eqx.nn.Embedding(max_classes, embed_dim, key=keys[1])
        self.y_embed_icl = eqx.nn.Embedding(max_classes, icl_dim, key=keys[2])

        col_keys = jax.random.split(keys[3], col_num_blocks)
        self.col_blocks = [
            InducedTransformerBlock(embed_dim, col_nhead, n_inducing=n_cls_rows, ssmax=True, key=k)
            for k in col_keys
        ]
        row_keys = jax.random.split(keys[4], row_num_blocks)
        self.row_blocks = [TransformerBlock(embed_dim, row_nhead, use_rope=True, key=k) for k in row_keys]
        icl_keys = jax.random.split(keys[5], icl_num_blocks)
        self.icl_blocks = [TransformerBlock(icl_dim, icl_nhead, ssmax=True, key=k) for k in icl_keys]

        self.row_cls_tokens = 0.02 * jax.random.normal(keys[6], (n_cls_cols, embed_dim))
        self.row_ln = eqx.nn.LayerNorm(embed_dim)
        self.out_ln = eqx.nn.LayerNorm(icl_dim)
        self.out_mlp = MLP(icl_dim, icl_dim * 2, out_dim, key=keys[7])

    def __call__(
        self,
        x: Float[Array, "num_rows num_cols"],
        y: Int[Array, " num_train"],
        n_train: int,
    ) -> Float[Array, "num_test out_dim"]:
        """Runs the full forward pass and returns logits for the test rows.

        Args:
            x: Feature matrix of shape (num_rows, num_cols); train rows come first.
            y: Integer class labels for the training rows, shape (num_train,).
            n_train: Number of training rows (static).

        Returns:
            Logits of shape (num_test, out_dim) for the test rows.
        """
        n_rows, n_cols = x.shape

        # ----- Embedding: standardize by train stats -> repeated feature grouping -> embed
        mean = jnp.mean(x[:n_train], axis=0, keepdims=True)
        std = jnp.std(x[:n_train], axis=0, keepdims=True) + 1e-8
        x = (x - mean) / std
        idxs = jnp.arange(n_cols)
        groups = [x[:, (idxs + (2**i - 1)) % n_cols] for i in range(self.feature_group_size)]
        x_grouped = jnp.stack(groups, axis=-1)  # (n_rows, n_cols, feature_group_size)
        emb = jax.vmap(jax.vmap(self.x_embed))(x_grouped)  # (n_rows, n_cols, embed_dim)

        y_int = y.astype(jnp.int32)
        y_in = jax.vmap(self.y_embed_in)(y_int)  # (n_train, embed_dim)
        emb = jnp.concatenate([emb[:n_train] + y_in[:, None, :], emb[n_train:]], axis=0)

        # ----- TF_col: induced self-attention within each column (rows attend to train rows)
        for block in self.col_blocks:
            emb = jax.vmap(lambda s, b=block: b(s, kv_max_idx=n_train), in_axes=1, out_axes=1)(emb)

        # ----- TF_row: prepend CLS columns -> row attention -> norm + merge CLS tokens
        cls = jnp.broadcast_to(self.row_cls_tokens, (n_rows, self.n_cls_cols, self.embed_dim))
        emb = jnp.concatenate([cls, emb], axis=1)  # (n_rows, n_cls_cols + n_cols, embed_dim)
        for block in self.row_blocks[:-1]:
            emb = jax.vmap(lambda s, b=block: b(s), in_axes=0, out_axes=0)(emb)
        last_row = self.row_blocks[-1]
        emb = jax.vmap(lambda s: last_row(s, q_max_idx=self.n_cls_cols), in_axes=0, out_axes=0)(emb)
        emb = jax.vmap(jax.vmap(self.row_ln))(emb)  # (n_rows, n_cls_cols, embed_dim)
        emb = emb.reshape(n_rows, self.n_cls_cols * self.embed_dim)  # (n_rows, icl_dim)

        # ----- TF_icl: add y embedding again -> self-attention over rows
        y_icl = jax.vmap(self.y_embed_icl)(y_int)  # (n_train, icl_dim)
        emb = jnp.concatenate([emb[:n_train] + y_icl, emb[n_train:]], axis=0)
        for block in self.icl_blocks[:-1]:
            emb = block(emb, kv_max_idx=n_train)
        emb = self.icl_blocks[-1](emb[n_train:], emb[:n_train])  # (n_test, icl_dim)

        return jax.vmap(self.out_mlp)(jax.vmap(self.out_ln)(emb))


@eqx.filter_jit
def predict(
    model: NanoTabICLModel,
    x: Float[Array, "num_rows num_cols"],
    y: Int[Array, " num_train"],
    n_train: int,
) -> Float[Array, "num_test out_dim"]:
    """JIT-compiled prediction function for :class:`NanoTabICLModel`."""
    return model(x, y, n_train)


class NanoTabICLClassifier:
    """scikit-learn like interface for the JAX nanoTabICL model."""

    def __init__(self, model: NanoTabICLModel) -> None:
        """Stores the model in inference mode.

        Args:
            model: A :class:`NanoTabICLModel` instance.
        """
        self.model = eqx.nn.inference_mode(model, value=True)
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.num_classes: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Stores the training data and infers the number of classes.

        Args:
            X_train: Training features of shape (num_train, num_features).
            y_train: Training labels of shape (num_train,).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train))) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for the test data.

        Args:
            X_test: Test features of shape (num_test, num_features).

        Returns:
            Probabilities of shape (num_test, num_classes).
        """
        x = jnp.concatenate([jnp.asarray(self.X_train), jnp.asarray(X_test)])
        y = jnp.asarray(self.y_train).astype(jnp.int32)
        n_train = len(self.X_train)

        out = predict(self.model, x, y, n_train)
        out = out[:, : self.num_classes]
        return np.array(jax.nn.softmax(out, axis=1))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predicts class labels for the test data.

        Args:
            X_test: Test features of shape (num_test, num_features).

        Returns:
            Predicted class indices of shape (num_test,).
        """
        return self.predict_proba(X_test).argmax(axis=1)

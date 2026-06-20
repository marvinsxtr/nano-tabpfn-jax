"""Nano JAX/Equinox port of TabPFN v3.

Unbatched (single dataset) + ``vmap`` style, like models/v2.py. Shape suffixes
follow full_v3.py: R rows, C columns, E embed, I inducing, T classes,
Cl CLS tokens, D=Cl*E (ICL embed), H heads, Dh head dim.

NANO scope: the simplest faithful forward pass. Intentionally omitted (to add
later): KV cache, row/col chunking, NaN indicators, and RoPE in the column
aggregator. Train-only attention is done with masks (not slicing), matching v2.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from models.common import NanoTabPFNClassifier as BaseClassifier

NUM_CLS_TOKENS = 2
NUM_INDUCING = 16
FEATURE_GROUP_SIZE = 2


def _make_mlp(emsize: int, dim_ff: int, *, key: PRNGKeyArray) -> eqx.nn.MLP:
    """Two-layer GELU MLP (E -> dim_ff -> E)."""
    return _zero_last(eqx.nn.MLP(emsize, emsize, dim_ff, depth=1, activation=jax.nn.gelu, key=key))


def _zero_last(mlp: eqx.nn.MLP) -> eqx.nn.MLP:
    """Zero the final Linear's weight and bias of an MLP."""
    mlp = eqx.tree_at(lambda m: m.layers[-1].weight, mlp, replace_fn=jnp.zeros_like)
    return eqx.tree_at(lambda m: m.layers[-1].bias, mlp, replace_fn=jnp.zeros_like)


class SoftmaxScalingMLP(eqx.Module):
    """Query scaling: q * base_mlp(log n) * (1 + tanh(query_mlp(q))).

    Lets the model sharpen attention (base term, length-aware) with a
    query-dependent modulation that starts at identity.
    """

    base_mlp: eqx.nn.MLP
    query_mlp: eqx.nn.MLP
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self, num_heads: int, head_dim: int, n_hidden: int = 64, init_scale: float = 1.0, *, key: PRNGKeyArray
    ) -> None:
        k1, k2 = jr.split(key)
        # Start as a constant base = init_scale (zero last weight, constant bias) so
        # attention is sharp from the start; the MLP then learns to adapt it.
        base = eqx.nn.MLP(1, num_heads * head_dim, n_hidden, depth=1, activation=jax.nn.gelu, key=k1)
        base = eqx.tree_at(lambda m: m.layers[-1].weight, base, replace_fn=jnp.zeros_like)
        self.base_mlp = eqx.tree_at(
            lambda m: m.layers[-1].bias, base, replace_fn=lambda b: jnp.full_like(b, init_scale)
        )
        self.query_mlp = _zero_last(
            eqx.nn.MLP(head_dim, head_dim, n_hidden, depth=1, activation=jax.nn.gelu, key=k2)
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, q_HSD: Float[Array, "H S Dh"], n: Array) -> Float[Array, "H S Dh"]:
        """Scale queries by a length-aware base and a query-dependent modulation."""
        logn = jnp.log(jnp.maximum(n, 1.0)).reshape(1)
        base = self.base_mlp(logn).reshape(self.num_heads, 1, self.head_dim)
        modulation = 1 + jnp.tanh(jax.vmap(jax.vmap(self.query_mlp))(q_HSD))
        return q_HSD * base * modulation


def _ortho_init(key: PRNGKeyArray, num_classes: int, embed_dim: int) -> Array:
    """Orthogonal-ish class embedding init (QR rows, unit-norm fallback)."""
    k1, k2 = jr.split(key)
    w = jr.normal(k1, (num_classes, embed_dim))
    w = w / jnp.clip(jnp.linalg.norm(w, axis=-1, keepdims=True), min=1e-8)
    k = min(num_classes, embed_dim)
    q, _ = jnp.linalg.qr(jr.normal(k2, (embed_dim, k)))
    return w.at[:k].set(q.T)  # noqa: PD008


class TrainableOrthogonalEmbedding(eqx.Module):
    """Trainable class embeddings with orthogonal initialization."""

    weight: Array

    def __init__(self, num_classes: int, embed_dim: int, *, key: PRNGKeyArray) -> None:
        self.weight = _ortho_init(key, num_classes, embed_dim)

    def __call__(self, idx: Int[Array, " R"]) -> Float[Array, "R embed_dim"]:
        """Look up embeddings for integer class labels."""
        return self.weight[idx]


class MultiheadAttention(eqx.Module):
    """Bias-free multi-head attention with an optional key mask."""

    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    softmax_scaling: SoftmaxScalingMLP | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling: SoftmaxScalingMLP | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3, k4 = jr.split(key, 4)
        inner = num_heads * head_dim
        self.wq = eqx.nn.Linear(dim, inner, use_bias=False, key=k1)
        self.wk = eqx.nn.Linear(dim, inner, use_bias=False, key=k2)
        self.wv = eqx.nn.Linear(dim, inner, use_bias=False, key=k3)
        self.wo = eqx.nn.Linear(inner, dim, use_bias=False, key=k4)
        self.softmax_scaling = softmax_scaling
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _heads(self, proj: eqx.nn.Linear, x: Float[Array, "S dim"]) -> Float[Array, "H S Dh"]:
        out = jax.vmap(proj)(x).reshape(x.shape[0], self.num_heads, self.head_dim)
        return jnp.transpose(out, (1, 0, 2))

    def __call__(
        self,
        x_q: Float[Array, "Sq dim"],
        x_kv: Float[Array, "Sk dim"],
        key_mask: Bool[Array, " Sk"] | None = None,
    ) -> Float[Array, "Sq dim"]:
        """Attend queries to keys/values, masking out non-allowed keys."""
        q = self._heads(self.wq, x_q)
        k = self._heads(self.wk, x_kv)
        v = self._heads(self.wv, x_kv)

        if self.softmax_scaling is not None:
            n = x_kv.shape[0] if key_mask is None else jnp.sum(key_mask)
            q = self.softmax_scaling(q, n)

        scores = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        if key_mask is not None:
            scores = jnp.where(key_mask[None, None, :], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hqk,hkd->hqd", attn, v)  # (H, Sq, Dh)

        out = jnp.transpose(out, (1, 0, 2)).reshape(x_q.shape[0], self.num_heads * self.head_dim)
        return jax.vmap(self.wo)(out)


class CrossAttentionBlock(eqx.Module):
    """Pre-norm cross-attention + MLP (query attends to key/value)."""

    attn: MultiheadAttention
    mlp: eqx.nn.MLP
    norm_q: eqx.nn.RMSNorm
    norm_kv: eqx.nn.RMSNorm
    norm_mlp: eqx.nn.RMSNorm

    def __init__(
        self,
        emsize: int,
        nhead: int,
        dim_ff: int,
        softmax_scaling: SoftmaxScalingMLP | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2 = jr.split(key)
        self.attn = MultiheadAttention(emsize, nhead, emsize // nhead, softmax_scaling, key=k1)
        self.mlp = _make_mlp(emsize, dim_ff, key=k2)
        self.norm_q = eqx.nn.RMSNorm(emsize)
        self.norm_kv = eqx.nn.RMSNorm(emsize)
        self.norm_mlp = eqx.nn.RMSNorm(emsize)

    def __call__(
        self,
        x_q: Float[Array, "Q E"],
        x_kv: Float[Array, "V E"],
        key_mask: Bool[Array, " V"] | None = None,
    ) -> Float[Array, "Q E"]:
        """Cross-attend the query to the key/value sequence, then apply the MLP."""
        x_q = x_q + self.attn(jax.vmap(self.norm_q)(x_q), jax.vmap(self.norm_kv)(x_kv), key_mask)
        return x_q + jax.vmap(self.mlp)(jax.vmap(self.norm_mlp)(x_q))


class TransformerBlock(eqx.Module):
    """Pre-norm self-attention + MLP, with a CLS-readout cross variant."""

    attn: MultiheadAttention
    mlp: eqx.nn.MLP
    norm: eqx.nn.RMSNorm
    norm_mlp: eqx.nn.RMSNorm

    def __init__(self, emsize: int, nhead: int, dim_ff: int, *, key: PRNGKeyArray) -> None:
        k1, k2 = jr.split(key)
        self.attn = MultiheadAttention(emsize, nhead, emsize // nhead, key=k1)
        self.mlp = _make_mlp(emsize, dim_ff, key=k2)
        self.norm = eqx.nn.RMSNorm(emsize)
        self.norm_mlp = eqx.nn.RMSNorm(emsize)

    def __call__(self, x_SE: Float[Array, "S E"]) -> Float[Array, "S E"]:
        """Self-attention over the sequence, then the MLP."""
        h = jax.vmap(self.norm)(x_SE)
        x_SE = x_SE + self.attn(h, h)
        return x_SE + jax.vmap(self.mlp)(jax.vmap(self.norm_mlp)(x_SE))

    def cross(self, q_QE: Float[Array, "Q E"], kv_VE: Float[Array, "V E"]) -> Float[Array, "Q E"]:
        """CLS query attends to the full sequence (column aggregator readout)."""
        q_QE = q_QE + self.attn(jax.vmap(self.norm)(q_QE), jax.vmap(self.norm)(kv_VE))
        return q_QE + jax.vmap(self.mlp)(jax.vmap(self.norm_mlp)(q_QE))


class InducedSelfAttentionBlock(eqx.Module):
    """SetTransformer-style induced attention: inducing points summarize train
    rows, then all rows attend to that summary.
    """

    block1: CrossAttentionBlock
    block2: CrossAttentionBlock
    inducing: Array

    def __init__(self, emsize: int, nhead: int, num_inducing: int, dim_ff: int, *, key: PRNGKeyArray) -> None:
        k1, k2, k3, k4 = jr.split(key, 4)
        scaling = SoftmaxScalingMLP(nhead, emsize // nhead, key=k4)
        self.block1 = CrossAttentionBlock(emsize, nhead, dim_ff, scaling, key=k1)
        self.block2 = CrossAttentionBlock(emsize, nhead, dim_ff, key=k2)
        self.inducing = jr.truncated_normal(k3, -2, 2, (num_inducing, emsize)) * 0.02

    def __call__(
        self, x_RE: Float[Array, "R E"], train_mask: Bool[Array, " R"]
    ) -> Float[Array, "R E"]:
        """Summarize train rows into inducing points, then broadcast to all rows."""
        hidden = self.block1(self.inducing, x_RE, key_mask=train_mask)  # (I, E)
        return self.block2(x_RE, hidden)  # (R, E)


class FeatureDistributionEmbedder(eqx.Module):
    """Stack of induced self-attention blocks applied independently per column."""

    layers: list

    def __init__(
        self, emsize: int, nhead: int, num_inducing: int, dim_ff: int, num_layers: int, *, key: PRNGKeyArray
    ) -> None:
        keys = jr.split(key, num_layers)
        self.layers = [InducedSelfAttentionBlock(emsize, nhead, num_inducing, dim_ff, key=k) for k in keys]

    def __call__(
        self, x_RCE: Float[Array, "R C E"], train_mask: Bool[Array, " R"]
    ) -> Float[Array, "R C E"]:
        """Run each column independently through the induced-attention stack."""
        x_CRE = jnp.transpose(x_RCE, (1, 0, 2))
        for layer in self.layers:
            x_CRE = jax.vmap(layer, in_axes=(0, None))(x_CRE, train_mask)
        return jnp.transpose(x_CRE, (1, 0, 2))


class ColumnAggregator(eqx.Module):
    """Per-row cross-feature interaction: CLS tokens aggregate column info."""

    blocks: list
    cls_tokens: Array
    out_norm: eqx.nn.RMSNorm
    num_cls: int = eqx.field(static=True)

    def __init__(
        self, emsize: int, nhead: int, num_layers: int, dim_ff: int, num_cls: int, *, key: PRNGKeyArray
    ) -> None:
        keys = jr.split(key, num_layers + 1)
        self.blocks = [TransformerBlock(emsize, nhead, dim_ff, key=k) for k in keys[:num_layers]]
        self.cls_tokens = jr.truncated_normal(keys[-1], -2, 2, (num_cls, emsize)) * 0.02
        self.out_norm = eqx.nn.RMSNorm(emsize)
        self.num_cls = num_cls

    def _row(self, x_CE: Float[Array, "C E"]) -> Float[Array, "Cl E"]:
        x = jnp.concatenate([self.cls_tokens, x_CE], axis=0)  # (Cl+C, E)
        for block in self.blocks[:-1]:
            x = block(x)
        cls_out = self.blocks[-1].cross(x[: self.num_cls], x)  # CLS readout
        return jax.vmap(self.out_norm)(cls_out)

    def __call__(self, x_RCE: Float[Array, "R C E"]) -> Float[Array, "R Cl E"]:
        """Aggregate each row's columns into CLS tokens."""
        return jax.vmap(self._row)(x_RCE)


class ICLTransformerBlock(eqx.Module):
    """Pre-norm transformer block where keys/values are restricted to train rows."""

    attn: MultiheadAttention
    mlp: eqx.nn.MLP
    norm: eqx.nn.RMSNorm
    norm_mlp: eqx.nn.RMSNorm

    def __init__(self, emsize: int, nhead: int, dim_ff: int, *, key: PRNGKeyArray) -> None:
        k1, k2, k3 = jr.split(key, 3)
        scaling = SoftmaxScalingMLP(nhead, emsize // nhead, key=k3)
        self.attn = MultiheadAttention(emsize, nhead, emsize // nhead, scaling, key=k1)
        self.mlp = _make_mlp(emsize, dim_ff, key=k2)
        self.norm = eqx.nn.RMSNorm(emsize)
        self.norm_mlp = eqx.nn.RMSNorm(emsize)

    def __call__(
        self, x_RD: Float[Array, "R D"], train_mask: Bool[Array, " R"]
    ) -> Float[Array, "R D"]:
        """Self-attention with keys/values restricted to train rows, then MLP."""
        h = jax.vmap(self.norm)(x_RD)
        x_RD = x_RD + self.attn(h, h, key_mask=train_mask)
        return x_RD + jax.vmap(self.mlp)(jax.vmap(self.norm_mlp)(x_RD))


class ManyClassDecoder(eqx.Module):
    """Attention retrieval decoder: test rows read a weighted average of one-hot
    train targets, then take the log to get logits.
    """

    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    softmax_scaling: SoftmaxScalingMLP
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    max_num_classes: int = eqx.field(static=True)

    def __init__(self, max_num_classes: int, input_size: int, num_heads: int, *, key: PRNGKeyArray) -> None:
        k1, k2, k3 = jr.split(key, 3)
        head_dim = input_size // num_heads
        self.wq = eqx.nn.Linear(input_size, num_heads * head_dim, key=k1)
        self.wk = eqx.nn.Linear(input_size, num_heads * head_dim, key=k2)
        self.softmax_scaling = SoftmaxScalingMLP(num_heads, head_dim, key=k3)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_classes = max_num_classes

    def __call__(
        self,
        train_emb: Float[Array, "R D"],
        test_emb: Float[Array, "R D"],
        targets: Int[Array, " R"],
        train_mask: Bool[Array, " R"],
    ) -> Float[Array, "R T"]:
        """Retrieve a weighted average of one-hot train targets as log-probs."""
        q = jnp.transpose(jax.vmap(self.wq)(test_emb).reshape(-1, self.num_heads, self.head_dim), (1, 0, 2))
        k = jnp.transpose(jax.vmap(self.wk)(train_emb).reshape(-1, self.num_heads, self.head_dim), (1, 0, 2))
        one_hot = jax.nn.one_hot(targets, self.max_num_classes)  # (R, T)

        q = self.softmax_scaling(q, jnp.sum(train_mask))
        scores = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        scores = jnp.where(train_mask[None, None, :], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        probs = jnp.einsum("hqk,kt->hqt", attn, one_hot).mean(0)  # (R, T)
        return jnp.log(jnp.clip(probs, min=1e-5) + 3e-5)


def _normalize(x_RC: Float[Array, "R C"], train_mask: Bool[Array, " R"]) -> Float[Array, "R C"]:
    """Standardize features using train-row statistics, then clip."""
    mask = train_mask[:, None]
    mean = jnp.mean(x_RC, axis=0, keepdims=True, where=mask)
    std = jnp.std(x_RC, axis=0, keepdims=True, where=mask, ddof=1) + 1e-20
    return jnp.clip((x_RC - mean) / std, min=-100, max=100)


def _group_features(x_RC: Float[Array, "R C"], group_size: int) -> Float[Array, "R C G"]:
    """Circular-shift feature groups (matches full_v3._group_features)."""
    shifts = [jnp.roll(x_RC, shift=-(2**i), axis=1) for i in range(group_size)]
    return jnp.stack(shifts, axis=-1)


class NanoTabPFNModel(eqx.Module):
    """TabPFN v3 forward pass: cell embed -> distribution embed -> column
    aggregate -> ICL transformer -> many-class decoder.
    """

    x_embed: eqx.nn.Linear
    col_y_embed: TrainableOrthogonalEmbedding
    icl_y_embed: TrainableOrthogonalEmbedding
    dist_embedder: FeatureDistributionEmbedder
    column_aggregator: ColumnAggregator
    icl_blocks: list
    output_norm: eqx.nn.RMSNorm
    decoder: ManyClassDecoder
    num_outputs: int = eqx.field(static=True)
    feature_group_size: int = eqx.field(static=True)

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
        keys = jr.split(key, 7 + num_layers)
        e = embedding_size
        h = num_attention_heads
        d = e * NUM_CLS_TOKENS  # ICL embedding dim

        self.x_embed = eqx.nn.Linear(FEATURE_GROUP_SIZE, e, key=keys[0])
        self.col_y_embed = TrainableOrthogonalEmbedding(num_outputs, e, key=keys[1])
        self.icl_y_embed = TrainableOrthogonalEmbedding(num_outputs, d, key=keys[2])
        self.dist_embedder = FeatureDistributionEmbedder(
            e, h, NUM_INDUCING, mlp_hidden_size, num_layers, key=keys[3]
        )
        self.column_aggregator = ColumnAggregator(
            e, h, num_layers, mlp_hidden_size, NUM_CLS_TOKENS, key=keys[4]
        )
        self.icl_blocks = [
            ICLTransformerBlock(d, h, mlp_hidden_size, key=k) for k in keys[5 : 5 + num_layers]
        ]
        self.output_norm = eqx.nn.RMSNorm(d)
        self.decoder = ManyClassDecoder(num_outputs, d, h, key=keys[5 + num_layers])
        self.num_outputs = num_outputs
        self.feature_group_size = FEATURE_GROUP_SIZE

    def __call__(
        self,
        x_src: Float[Array, "R C"],
        y_src: Float[Array, " R"],
        train_mask: Bool[Array, " R"],
    ) -> Float[Array, "R T"]:
        """Forward pass for one dataset; returns per-row class logits."""
        idx = jnp.clip(y_src.astype(jnp.int32), min=0, max=self.num_outputs - 1)

        # ---- Cell embedding (normalize -> group -> linear) ----
        x = _normalize(x_src, train_mask)
        x = _group_features(x, self.feature_group_size)  # (R, C, G)
        x_RCE = jax.vmap(jax.vmap(self.x_embed))(x)  # (R, C, E)

        # ---- Target-aware column embedding (train rows only) ----
        y_col = self.col_y_embed(idx) * train_mask[:, None]  # (R, E)
        x_RCE = x_RCE + y_col[:, None, :]

        # ---- Distribution embedder + column aggregation ----
        x_RCE = self.dist_embedder(x_RCE, train_mask)
        cls = self.column_aggregator(x_RCE)  # (R, Cl, E)
        x_RD = cls.reshape(cls.shape[0], -1)  # (R, D)

        # ---- ICL transformer (train rows only attend; y added to train) ----
        x_RD = x_RD + self.icl_y_embed(idx) * train_mask[:, None]
        for block in self.icl_blocks:
            x_RD = block(x_RD, train_mask)
        x_RD = jax.vmap(self.output_norm)(x_RD)

        return self.decoder(x_RD, x_RD, idx, train_mask)


@eqx.filter_jit
def predict(
    model: NanoTabPFNModel,
    x: Float[Array, "R C"],
    y: Float[Array, " R"],
    train_mask: Bool[Array, " R"],
) -> Float[Array, "R T"]:
    """JIT-compiled prediction for a single dataset."""
    return model(x, y, train_mask=train_mask)


class NanoTabPFNClassifier(BaseClassifier):
    """scikit-learn-like interface for the v3 JAX model."""

    def __init__(self, model: NanoTabPFNModel) -> None:
        super().__init__(model, predict)

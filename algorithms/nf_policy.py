import math
from dataclasses import dataclass
from typing import List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg


def _atanh(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jnp.log((1 + x) / (1 - x))


def _normal_log_prob(z: jax.Array) -> jax.Array:
    return -0.5 * (z ** 2 + math.log(2 * math.pi))


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    def setup(self) -> None:
        self.scale = self.param("scale", nn.initializers.ones, (self.dim,))

    def __call__(self, x: jax.Array) -> jax.Array:
        norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.scale


def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[jax.Array, jax.Array]:
    freqs = jnp.arange(0, dim, 2, dtype=jnp.float32)
    freqs = theta ** (-freqs / dim)
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary_pos_emb(
    q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    q1, q2 = jnp.split(q, 2, axis=-1)
    k1, k2 = jnp.split(k, 2, axis=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
    return q_rot, k_rot


class SwiGlu(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0
    out_dim: int = None

    def setup(self) -> None:
        out_dim = self.out_dim if self.out_dim is not None else self.dim
        self.fc1 = nn.Dense(self.hidden_dim, use_bias=False)
        self.fc2 = nn.Dense(self.hidden_dim, use_bias=False)
        self.proj = nn.Dense(out_dim, use_bias=False)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x1 = nn.silu(self.fc1(x))
        x2 = self.fc2(x)
        x = x1 * x2
        x = self.dropout_layer(x, deterministic=not train)
        x = self.proj(x)
        return x


class FlowerAttention(nn.Module):
    dim: int
    n_heads: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    use_rope: bool = False
    max_seq_len: int = 128
    rope_theta: float = 32.0

    def setup(self) -> None:
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Dense(self.dim * 3, use_bias=False)
        self.proj = nn.Dense(self.dim, use_bias=False)
        self.attn_dropout = nn.Dropout(rate=self.attn_pdrop)
        self.resid_dropout = nn.Dropout(rate=self.resid_pdrop)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        if self.use_rope:
            cos, sin = precompute_freqs_1d(self.head_dim, self.max_seq_len, self.rope_theta)
            self.cos = cos
            self.sin = sin

    def __call__(self, x: jax.Array, train: bool, is_causal: bool = False) -> jax.Array:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3, self.head_dim)
        qkv = qkv.transpose(0, 2, 1, 3, 4)  # [B, H, T, 3, D]
        q, k, v = jnp.split(qkv, 3, axis=3)
        q = jnp.squeeze(q, axis=3)
        k = jnp.squeeze(k, axis=3)
        v = jnp.squeeze(v, axis=3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos[:T], self.sin[:T])
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if is_causal:
            mask = jnp.triu(jnp.ones((T, T), dtype=bool), 1)
            attn = jnp.where(mask[None, None, :, :], -jnp.inf, attn)
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not train)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out), deterministic=not train)
        return out


class FlowerCrossAttention(nn.Module):
    dim: int
    n_heads: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1

    def setup(self) -> None:
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Dense(self.dim, use_bias=False)
        self.k_proj = nn.Dense(self.dim, use_bias=False)
        self.v_proj = nn.Dense(self.dim, use_bias=False)
        self.proj = nn.Dense(self.dim, use_bias=False)
        self.attn_dropout = nn.Dropout(rate=self.attn_pdrop)
        self.resid_dropout = nn.Dropout(rate=self.resid_pdrop)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def __call__(self, x: jax.Array, context: jax.Array, train: bool) -> jax.Array:
        B, T, C = x.shape
        _, S, _ = context.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not train)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out), deterministic=not train)
        return out


class FlowBlock(nn.Module):
    dim: int
    heads: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1
    use_cross_attn: bool = False
    use_rope: bool = False
    max_seq_len: int = 128
    rope_theta: float = 32.0

    def setup(self) -> None:
        self.norm1 = RMSNorm(self.dim)
        self.norm2 = RMSNorm(self.dim)
        self.norm3 = RMSNorm(self.dim) if self.use_cross_attn else None
        self.self_attn = FlowerAttention(
            dim=self.dim,
            n_heads=self.heads,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            use_rope=self.use_rope,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
        )
        if self.use_cross_attn:
            self.cross_attn = FlowerCrossAttention(
                dim=self.dim,
                n_heads=self.heads,
                attn_pdrop=self.attn_pdrop,
                resid_pdrop=self.resid_pdrop,
            )
        self.mlp = SwiGlu(self.dim, hidden_dim=self.dim * 4, dropout=self.mlp_pdrop, out_dim=self.dim)

    def __call__(self, x: jax.Array, context: jax.Array, train: bool, is_causal: bool = True) -> jax.Array:
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, train=train, is_causal=is_causal)
        if self.use_cross_attn:
            x_norm = self.norm2(x)
            x = x + self.cross_attn(x_norm, context, train=train)
            x_norm = self.norm3(x)
        else:
            x_norm = self.norm2(x)
        x = x + self.mlp(x_norm, train=train)
        return x


class TransformerBlock(nn.Module):
    in_dim: int
    cond_dim: int
    dim: int = 64
    num_heads: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1
    block_depth: int = 1
    is_causal: bool = True
    use_rope: bool = True
    rope_theta: float = 32.0
    max_seq_len: int = 128

    def setup(self) -> None:
        self.input_proj = nn.Dense(self.dim)
        self.cond_fc1 = nn.Dense(self.dim * 2)
        self.cond_drop = nn.Dropout(rate=self.mlp_pdrop)
        self.cond_fc2 = nn.Dense(self.dim)
        self.cond_proj = nn.Dense(self.dim)
        blocks = []
        for _ in range(self.block_depth):
            blocks.append(
                FlowBlock(
                    dim=self.dim,
                    heads=self.num_heads,
                    attn_pdrop=self.attn_pdrop,
                    resid_pdrop=self.resid_pdrop,
                    mlp_pdrop=self.mlp_pdrop,
                    use_cross_attn=False,
                    use_rope=self.use_rope,
                    max_seq_len=self.max_seq_len,
                    rope_theta=self.rope_theta,
                )
            )
            blocks.append(
                FlowBlock(
                    dim=self.dim,
                    heads=self.num_heads,
                    attn_pdrop=self.attn_pdrop,
                    resid_pdrop=self.resid_pdrop,
                    mlp_pdrop=self.mlp_pdrop,
                    use_cross_attn=True,
                    use_rope=self.use_rope,
                    max_seq_len=self.max_seq_len,
                    rope_theta=self.rope_theta,
                )
            )
        self.blocks = blocks
        self.norm = RMSNorm(self.dim)
        self.ff_mult_dense1 = nn.Dense(self.dim * 2)
        self.ff_mult_drop = nn.Dropout(rate=self.mlp_pdrop)
        self.ff_mult_dense2 = nn.Dense(self.in_dim)
        self.ff_shift_dense1 = nn.Dense(self.dim * 2)
        self.ff_shift_drop = nn.Dropout(rate=self.mlp_pdrop)
        self.ff_shift_dense2 = nn.Dense(self.in_dim)

    def __call__(self, x: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        x = self.input_proj(x)
        cond_enc = self.cond_fc1(cond)
        cond_enc = nn.gelu(cond_enc)
        cond_enc = self.cond_drop(cond_enc, deterministic=not train)
        cond_enc = self.cond_fc2(cond_enc)
        context = self.cond_proj(cond_enc)[:, None, :]
        for block in self.blocks:
            x = block(x, context, train=train, is_causal=self.is_causal)
        x = self.norm(x)
        x_mult = self.ff_mult_dense1(x)
        x_mult = nn.gelu(x_mult)
        x_mult = self.ff_mult_drop(x_mult, deterministic=not train)
        x_mult = self.ff_mult_dense2(x_mult)
        x_shift = self.ff_shift_dense1(x)
        x_shift = nn.gelu(x_shift)
        x_shift = self.ff_shift_drop(x_shift, deterministic=not train)
        x_shift = self.ff_shift_dense2(x_shift)
        return x_mult, x_shift

@dataclass
class FlowConfig:
    num_layers: int = 4
    hidden_dim: int = 256
    n_hiddens: int = 3
    scale_max: float = 1.0


class AffineCoupling(nn.Module):
    dim: int
    hidden_dim: int
    n_hiddens: int
    idx1: jax.Array
    idx2: jax.Array
    scale_max: float = 1.0
    use_layernorm: bool = False
    dropout_rate: float = 0.0
    use_transformer: bool = True
    tf_dim: int = 128
    tf_heads: int = 8
    tf_depth: int = 1
    tf_rope_theta: float = 32.0
    tf_max_seq_len: int = 128

    def setup(self) -> None:
        self.n_out = int(self.idx2.shape[0])
        if self.use_transformer:
            self.tf = TransformerBlock(
                in_dim=1,
                cond_dim=1,
                dim=self.tf_dim,
                num_heads=self.tf_heads,
                attn_pdrop=self.dropout_rate,
                resid_pdrop=self.dropout_rate,
                mlp_pdrop=self.dropout_rate,
                block_depth=self.tf_depth,
                is_causal=True,
                use_rope=True,
                rope_theta=self.tf_rope_theta,
                max_seq_len=self.tf_max_seq_len,
            )

    @nn.compact
    def _st(self, x1: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        if self.use_transformer:
            x_full = jnp.zeros((x1.shape[0], self.dim), dtype=x1.dtype)
            x_full = x_full.at[:, self.idx1].set(x1)
            x_tokens = x_full[..., None]
            x_mult, x_shift = self.tf(x_tokens, cond, train=train)
            s = jnp.squeeze(x_mult, axis=-1)
            t = jnp.squeeze(x_shift, axis=-1)
            s = jnp.tanh(s) * self.scale_max
            s = jnp.take(s, self.idx2, axis=1)
            t = jnp.take(t, self.idx2, axis=1)
            return s, t
        h = jnp.concatenate([x1, cond], axis=-1)
        for _ in range(self.n_hiddens - 1):
            h = nn.Dense(self.hidden_dim)(h)
            if self.use_layernorm:
                h = nn.LayerNorm()(h)
            h = nn.gelu(h)
            if self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
        h = nn.Dense(2 * self.n_out)(h)
        s, t = jnp.split(h, 2, axis=-1)
        s = jnp.tanh(s) * self.scale_max
        return s, t

    def __call__(
        self, x: jax.Array, cond: jax.Array, train: bool, reverse: bool = False
    ) -> Tuple[jax.Array, jax.Array]:
        x1 = jnp.take(x, self.idx1, axis=1)
        x2 = jnp.take(x, self.idx2, axis=1)
        s, t = self._st(x1, cond, train)
        if not reverse:
            y2 = x2 * jnp.exp(s) + t
            y = x
            y = y.at[:, self.idx2].set(y2)
            logdet = jnp.sum(s, axis=-1)
            return y, logdet
        x2_inv = (x2 - t) * jnp.exp(-s)
        x_inv = x
        x_inv = x_inv.at[:, self.idx2].set(x2_inv)
        logdet = -jnp.sum(s, axis=-1)
        return x_inv, logdet


class InvertiblePLU(nn.Module):
    dim: int
    key: jax.Array = jax.random.PRNGKey(0)

    def setup(self) -> None:
        w_shape = (self.dim, self.dim)
        w_init = nn.initializers.orthogonal()(self.key, w_shape)
        P, L, U = jsp_linalg.lu(w_init)
        s = jnp.diag(U)
        U = U - jnp.diag(s)

        self.P = P
        self.P_inv = jsp_linalg.inv(P)
        self.L_init = jnp.tril(L, k=-1)
        self.U_init = jnp.triu(U, k=1)
        self.s_init = s
        self.L = self.param("L", lambda rng, val=self.L_init: val)
        self.U = self.param("U", lambda rng, val=self.U_init: val)
        self.s = self.param("s", lambda rng, val=self.s_init: val)

    def _build(self):
        L = jnp.tril(self.L, k=-1) + jnp.eye(self.dim, dtype=self.L.dtype)
        U = jnp.triu(self.U, k=1)
        s = self.s
        P = self.P
        P_inv = self.P_inv
        return P, P_inv, L, U, s

    def __call__(
        self, x: jax.Array, cond: jax.Array = None, train: bool = False, reverse: bool = False
    ) -> Tuple[jax.Array, jax.Array]:
        P, P_inv, L, U, s = self._build()
        W = P @ L @ (U + jnp.diag(s))
        logdet = jnp.sum(jnp.log(jnp.abs(s)))
        logdet = jnp.full((x.shape[0],), logdet, dtype=x.dtype)
        if not reverse:
            y = x @ W
            return y, logdet
        U2 = U + jnp.diag(s)
        U_inv = jsp_linalg.solve_triangular(U2, jnp.eye(self.dim, dtype=U2.dtype), lower=False)
        L_inv = jsp_linalg.solve_triangular(L, jnp.eye(self.dim, dtype=L.dtype), lower=True, unit_diagonal=True)
        W_inv = U_inv @ L_inv @ P_inv
        z = x @ W_inv
        return z, -logdet


class NFActor(nn.Module):
    action_dim: int
    chunk_len: int
    hidden_dim: int
    n_hiddens: int
    num_layers: int
    scale_max: float = 1.0
    base_dist: str = "normal"
    use_plu: bool = True
    use_layernorm: bool = False
    dropout_rate: float = 0.0
    deterministic_layers: int = 2
    tf_heads: int = 8
    tf_depth: int = 1
    tf_rope_theta: float = 32.0
    tf_max_seq_len: int = 128

    def setup(self) -> None:
        self.dim = self.action_dim * self.chunk_len
        base_mask = (jnp.arange(self.chunk_len) % 2).astype(jnp.float32)
        half = self.chunk_len // 2

        def _init_indices(rng, select_first: bool):
            if self.num_layers == 0:
                return jnp.zeros(
                    (0, half if select_first else self.chunk_len - half), dtype=jnp.int32
                )
            idx_list = []
            key = rng
            for i in range(self.num_layers):
                if i >= self.num_layers - self.deterministic_layers:
                    mask = base_mask if (i % 2 == 0) else 1.0 - base_mask
                    if select_first:
                        idx = jnp.where(mask > 0.5, size=half, fill_value=0)[0]
                    else:
                        idx = jnp.where(mask <= 0.5, size=self.chunk_len - half, fill_value=0)[0]
                else:
                    key, subkey = jax.random.split(key)
                    perm = jax.random.permutation(subkey, self.chunk_len)
                    if select_first:
                        idx = jnp.sort(perm[:half])
                    else:
                        idx = jnp.sort(perm[half:])
                idx_list.append(idx)
            return jnp.stack(idx_list, axis=0)

        mask_key = self.make_rng("mask") if self.has_rng("mask") else jax.random.PRNGKey(0)
        idx1_all = self.variable(
            "constants", "idx1", lambda rng: _init_indices(rng, True), mask_key
        ).value
        idx2_all = self.variable(
            "constants", "idx2", lambda rng: _init_indices(rng, False), mask_key
        ).value
        layers = []
        CouplingTime = nn.vmap(
            target=AffineCoupling,
            in_axes=(1, None, None, None),
            out_axes=1,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.action_dim,
        )
        PLUTime = nn.vmap(
            target=InvertiblePLU,
            in_axes=(1, None, None, None),
            out_axes=1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.action_dim,
        )
        for i in range(self.num_layers):
            layers.append(
                CouplingTime(
                    dim=self.chunk_len,
                    hidden_dim=self.hidden_dim,
                    n_hiddens=self.n_hiddens,
                    idx1=idx1_all[i],
                    idx2=idx2_all[i],
                    scale_max=self.scale_max,
                    use_layernorm=self.use_layernorm,
                    dropout_rate=self.dropout_rate,
                    use_transformer=True,
                    tf_dim=self.hidden_dim,
                    tf_heads=self.tf_heads,
                    tf_depth=self.tf_depth,
                    tf_rope_theta=self.tf_rope_theta,
                    tf_max_seq_len=self.tf_max_seq_len,
                )
            )
            if self.use_plu:
                layers.append(PLUTime(dim=self.chunk_len))
        self.couplings = layers

    def _cond(self, state: jax.Array, prev_actions: jax.Array) -> Tuple[jax.Array, bool]:
        single = state.ndim == 1
        if single:
            state = state[None, :]
            if prev_actions.ndim == 2:
                prev_actions = prev_actions[None, ...]
        prev_flat = prev_actions.reshape(prev_actions.shape[0], -1)
        cond = jnp.hstack([state, prev_flat])
        return cond, single

    def _forward_flow(self, z: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        logdet = jnp.zeros(z.shape[0], dtype=z.dtype)
        x = z
        for layer in self.couplings:
            x, ldj = layer(x, cond, train, False)
            logdet = logdet + jnp.sum(ldj, axis=1)
        return x, logdet

    def _inverse_flow(self, x: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        logdet = jnp.zeros(x.shape[0], dtype=x.dtype)
        z = x
        for layer in reversed(self.couplings):
            z, ldj = layer(z, cond, train, True)
            logdet = logdet + jnp.sum(ldj, axis=1)
        return z, logdet

    def sample(
        self, state: jax.Array, prev_actions: jax.Array, rng: jax.Array, train: bool = False
    ) -> jax.Array:
        cond, single = self._cond(state, prev_actions)
        if self.base_dist == "uniform":
            u = jax.random.uniform(
                rng, shape=(cond.shape[0], self.action_dim, self.chunk_len), minval=-1.0, maxval=1.0
            )
            z = _atanh(u)
        else:
            z = jax.random.normal(rng, shape=(cond.shape[0], self.action_dim, self.chunk_len))
        x, _ = self._forward_flow(z, cond, train)
        actions = jnp.transpose(jnp.tanh(x), (0, 2, 1))
        if single:
            actions = actions[0]
        return actions

    def sample_n(
        self,
        state: jax.Array,
        prev_actions: jax.Array,
        rng: jax.Array,
        num_samples: int,
        z_scale: float = 1.0,
        z_clip: float = 0.0,
        train: bool = False,
    ) -> jax.Array:
        cond, single = self._cond(state, prev_actions)
        if self.base_dist == "uniform":
            u = jax.random.uniform(
                rng,
                shape=(cond.shape[0], num_samples, self.action_dim, self.chunk_len),
                minval=-1.0,
                maxval=1.0,
            )
            if z_scale and z_scale != 1.0:
                u = u * z_scale
            if z_clip and z_clip > 0:
                u = jnp.clip(u, -z_clip, z_clip)
            z = _atanh(u)
        else:
            z = jax.random.normal(
                rng, shape=(cond.shape[0], num_samples, self.action_dim, self.chunk_len)
            )
            if z_clip and z_clip > 0:
                z = jnp.clip(z, -z_clip, z_clip)
            if z_scale and z_scale != 1.0:
                z = z * z_scale

        def _sample_one(z_one):
            x, _ = self._forward_flow(z_one, cond, train)
            return jnp.transpose(jnp.tanh(x), (0, 2, 1))

        actions = jax.vmap(_sample_one, in_axes=1, out_axes=1)(z)
        if single:
            actions = actions[0]
        return actions

    def log_prob(
        self, actions: jax.Array, state: jax.Array, prev_actions: jax.Array, train: bool = False
    ) -> jax.Array:
        cond, single = self._cond(state, prev_actions)
        if actions.ndim == 2:
            actions = actions[None, ...]
        a = jnp.transpose(actions, (0, 2, 1))
        pre_tanh = _atanh(a)
        logdet_tanh = jnp.sum(
            jnp.log(1.0 - jnp.tanh(pre_tanh) ** 2 + 1e-6), axis=(1, 2)
        )
        z, logdet = self._inverse_flow(pre_tanh, cond, train)
        if self.base_dist == "uniform":
            u = jnp.tanh(z)
            logdet_base = jnp.sum(jnp.log(1.0 - u ** 2 + 1e-6), axis=(1, 2))
            base_log_prob = logdet_base + (self.dim * math.log(0.5))
        else:
            base_log_prob = jnp.sum(_normal_log_prob(z), axis=(1, 2))
        log_prob = base_log_prob + logdet - logdet_tanh
        if single:
            log_prob = log_prob[0]
        return log_prob

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

    def setup(self) -> None:
        self.n_out = int(self.idx2.shape[0])

    @nn.compact
    def _st(self, x1: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
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

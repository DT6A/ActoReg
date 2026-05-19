import math
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg


def _atanh(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jnp.log((1 + x) / (1 - x))


def _normal_log_prob(z: jax.Array) -> jax.Array:
    return -0.5 * (z ** 2 + math.log(2 * math.pi))


def GSP(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    # GELU-Sinc-Perturbation (GSP)
    alpha = 0.5
    return jax.nn.gelu(x) * (1.0 + alpha * jax.numpy.sinc(x))


def resolve_activation(name: str):
    name = name.lower()
    if name == "silu":
        return nn.silu
    if name == "gsp":
        return GSP
    raise ValueError(f"Unsupported activation '{name}'. Expected one of: silu, gsp")


class Conditioner(nn.Module):
    out_dim: int
    hidden_dim: int
    n_hiddens: int
    use_layernorm: bool = True
    dropout_rate: float = 0.0
    activation: str = "silu"

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        activation_fn = resolve_activation(self.activation)
        h = x
        for _ in range(max(self.n_hiddens, 1)):
            z = nn.Dense(self.hidden_dim)(h)
            z = activation_fn(z)
            if self.use_layernorm:
                z = nn.LayerNorm()(z)
            z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=not train)
            # Residual MLP block when shapes match.
            if h.shape[-1] == z.shape[-1]:
                h = h + z
            else:
                h = z
        # Reference-style stable start: make coupling initially near-identity.
        out = nn.Dense(
            2 * self.out_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        shift, log_scale = jnp.split(out, 2, axis=-1)
        return shift, log_scale


class CouplingLayer(nn.Module):
    dim: int
    idx1: jax.Array
    idx2: jax.Array
    hidden_dim: int
    n_hiddens: int
    scale_max: float = 1.0
    use_layernorm: bool = True
    dropout_rate: float = 0.0
    activation: str = "silu"

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array, train: bool, reverse: bool) -> Tuple[jax.Array, jax.Array]:
        x1 = jnp.take(x, self.idx1, axis=1)
        x2 = jnp.take(x, self.idx2, axis=1)
        cond_in = jnp.concatenate([cond, x1], axis=-1)
        shift, log_scale = Conditioner(
            out_dim=int(self.idx2.shape[0]),
            hidden_dim=self.hidden_dim,
            n_hiddens=self.n_hiddens,
            use_layernorm=self.use_layernorm,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )(cond_in, train)
        log_scale = jnp.tanh(log_scale) * self.scale_max

        if reverse:
            x2_inv = (x2 - shift) * jnp.exp(-log_scale)
            y = x.at[:, self.idx2].set(x2_inv)
            ldj = -jnp.sum(log_scale, axis=-1)
        else:
            y2 = x2 * jnp.exp(log_scale) + shift
            y = x.at[:, self.idx2].set(y2)
            ldj = jnp.sum(log_scale, axis=-1)
        return y, ldj


class InvertiblePLU(nn.Module):
    dim: int
    key: jax.Array = jax.random.PRNGKey(0)
    min_abs_diag: float = 1e-4

    def setup(self) -> None:
        w_shape = (self.dim, self.dim)
        w_init = nn.initializers.orthogonal()(self.key, w_shape)
        p, l, u = jsp_linalg.lu(w_init)
        s = jnp.diag(u)
        u = u - jnp.diag(s)
        self.p = p
        self.p_inv = jsp_linalg.inv(p)
        self.l_init = jnp.tril(l, k=-1)
        self.u_init = jnp.triu(u, k=1)
        self.s_init = s
        self.l = self.param("L", lambda rng, val=self.l_init: val)
        self.u = self.param("U", lambda rng, val=self.u_init: val)
        self.s = self.param("s", lambda rng, val=self.s_init: val)

    def _build(self):
        l = jnp.tril(self.l, k=-1) + jnp.eye(self.dim, dtype=self.l.dtype)
        u = jnp.triu(self.u, k=1)
        sign = jnp.where(self.s >= 0, 1.0, -1.0)
        s = sign * jnp.maximum(jnp.abs(self.s), self.min_abs_diag)
        return self.p, self.p_inv, l, u, s

    def __call__(
        self,
        x: jax.Array,
        cond: jax.Array = None,
        train: bool = False,
        reverse: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        del cond, train
        p, p_inv, l, u, s = self._build()
        w = p @ l @ (u + jnp.diag(s))
        logdet = jnp.sum(jnp.log(jnp.abs(s)))
        logdet = jnp.full((x.shape[0],), logdet, dtype=x.dtype)
        if not reverse:
            y = x @ w
            return y, logdet
        u2 = u + jnp.diag(s)
        u_inv = jsp_linalg.solve_triangular(u2, jnp.eye(self.dim, dtype=u2.dtype), lower=False)
        l_inv = jsp_linalg.solve_triangular(l, jnp.eye(self.dim, dtype=l.dtype), lower=True, unit_diagonal=True)
        w_inv = u_inv @ l_inv @ p_inv
        z = x @ w_inv
        return z, -logdet


class NFActorFlat(nn.Module):
    action_dim: int
    hidden_dim: int
    n_hiddens: int
    num_layers: int
    scale_max: float = 1.0
    base_dist: str = "normal"
    use_plu: bool = True
    use_layernorm: bool = True
    dropout_rate: float = 0.0
    deterministic_layers: int = 2
    activation: str = "silu"

    def setup(self) -> None:
        half = self.action_dim // 2

        det_layers = max(0, min(self.deterministic_layers, self.num_layers))
        det_start = self.num_layers - det_layers

        def _init_indices(rng, select_first: bool):
            if self.num_layers == 0:
                size = half if select_first else self.action_dim - half
                return jnp.zeros((0, size), dtype=jnp.int32)
            idx_list = []
            key = rng
            for i in range(self.num_layers):
                if i < det_start:
                    key, subkey = jax.random.split(key)
                    perm = jax.random.permutation(subkey, self.action_dim)
                    if select_first:
                        idx = perm[:half]
                    else:
                        idx = perm[half:]
                else:
                    # Deterministic one-sided masks on the tail layers.
                    # Alternate sides so all dimensions are transformed repeatedly.
                    det_i = i - det_start
                    use_right_as_idx1 = (det_i % 2 == 1)
                    if use_right_as_idx1:
                        idx1_det = jnp.arange(self.action_dim - half, self.action_dim, dtype=jnp.int32)
                        idx2_det = jnp.arange(self.action_dim - half, dtype=jnp.int32)
                    else:
                        idx1_det = jnp.arange(half, dtype=jnp.int32)
                        idx2_det = jnp.arange(half, self.action_dim, dtype=jnp.int32)
                    idx = idx1_det if select_first else idx2_det
                idx_list.append(idx)
            return jnp.stack(idx_list, axis=0)

        mask_key = self.make_rng("mask") if self.has_rng("mask") else jax.random.PRNGKey(0)
        idx1_all = self.variable("constants", "idx1", lambda rng: _init_indices(rng, True), mask_key).value
        idx2_all = self.variable("constants", "idx2", lambda rng: _init_indices(rng, False), mask_key).value

        couplings = []
        plus = []
        for i in range(self.num_layers):
            if self.use_plu:
                plus.append(InvertiblePLU(dim=self.action_dim))
            couplings.append(
                CouplingLayer(
                    dim=self.action_dim,
                    idx1=idx1_all[i],
                    idx2=idx2_all[i],
                    hidden_dim=self.hidden_dim,
                    n_hiddens=self.n_hiddens,
                    scale_max=self.scale_max,
                    use_layernorm=self.use_layernorm,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                )
            )
        self.couplings = couplings
        self.plus = plus

    def _cond(self, state: jax.Array) -> Tuple[jax.Array, bool]:
        single = state.ndim == 1
        if single:
            state = state[None, :]
        return state, single

    def _forward_flow(self, z: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        # Decode (reference reverse path): iterate blocks in reverse.
        x = z
        logdet = jnp.zeros((z.shape[0],), dtype=z.dtype)
        for i in range(self.num_layers - 1, -1, -1):
            x, ldj = self.couplings[i](x, cond, train, reverse=False)
            logdet = logdet + ldj
            if self.use_plu:
                x, ldj = self.plus[i](x, cond, train, reverse=True)
            logdet = logdet + ldj
        return x, logdet

    def _inverse_flow(self, x: jax.Array, cond: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        # Encode (reference forward path): PLU -> coupling on each block.
        z = x
        logdet = jnp.zeros((x.shape[0],), dtype=x.dtype)
        for i in range(self.num_layers):
            if self.use_plu:
                z, ldj = self.plus[i](z, cond, train, reverse=False)
                logdet = logdet + ldj
            z, ldj = self.couplings[i](z, cond, train, reverse=True)
            logdet = logdet + ldj
        return z, logdet

    def sample(self, state: jax.Array, rng: jax.Array, train: bool = False) -> jax.Array:
        cond, single = self._cond(state)
        if self.base_dist == "uniform":
            u = jax.random.uniform(rng, shape=(cond.shape[0], self.action_dim), minval=-1.0, maxval=1.0)
            z = _atanh(u)
        else:
            z = jax.random.normal(rng, shape=(cond.shape[0], self.action_dim))
        x, _ = self._forward_flow(z, cond, train)
        actions = jnp.tanh(x)
        if single:
            actions = actions[0]
        return actions

    def sample_n(
        self,
        state: jax.Array,
        rng: jax.Array,
        num_samples: int,
        z_scale: float = 1.0,
        z_clip: float = 0.0,
        train: bool = False,
    ) -> jax.Array:
        cond, single = self._cond(state)
        bsz = cond.shape[0]
        if self.base_dist == "uniform":
            u = jax.random.uniform(rng, shape=(bsz, num_samples, self.action_dim), minval=-1.0, maxval=1.0)
            if z_scale and z_scale != 1.0:
                u = u * z_scale
            if z_clip and z_clip > 0:
                u = jnp.clip(u, -z_clip, z_clip)
            z = _atanh(u)
        else:
            z = jax.random.normal(rng, shape=(bsz, num_samples, self.action_dim))
            if z_clip and z_clip > 0:
                z = jnp.clip(z, -z_clip, z_clip)
            if z_scale and z_scale != 1.0:
                z = z * z_scale

        z_flat = z.reshape(bsz * num_samples, self.action_dim)
        cond_flat = jnp.repeat(cond, repeats=num_samples, axis=0)
        x_flat, _ = self._forward_flow(z_flat, cond_flat, train)
        actions = jnp.tanh(x_flat).reshape(bsz, num_samples, self.action_dim)
        if single:
            actions = actions[0]
        return actions

    def log_prob(self, actions: jax.Array, state: jax.Array, train: bool = False) -> jax.Array:
        cond, single = self._cond(state)
        if actions.ndim == 1:
            actions = actions[None, :]
        pre_tanh = _atanh(actions)
        logdet_tanh = jnp.sum(jnp.log(1.0 - jnp.tanh(pre_tanh) ** 2 + 1e-6), axis=-1)
        z, logdet = self._inverse_flow(pre_tanh, cond, train)
        if self.base_dist == "uniform":
            u = jnp.tanh(z)
            logdet_base = jnp.sum(jnp.log(1.0 - u ** 2 + 1e-6), axis=-1)
            base_log_prob = logdet_base + (self.action_dim * math.log(0.5))
        else:
            base_log_prob = jnp.sum(_normal_log_prob(z), axis=-1)
        log_prob = base_log_prob + logdet - logdet_tanh
        if single:
            log_prob = log_prob[0]
        return log_prob

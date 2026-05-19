import math
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

try:
    from nf_policy_action import GSP, InvertiblePLU, _atanh, _normal_log_prob, resolve_activation
except ImportError:  # pragma: no cover
    from algorithms.nf_policy_action import GSP, InvertiblePLU, _atanh, _normal_log_prob, resolve_activation


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3
DEFAULT_EPS = 1e-6


def _inverse_softplus(x: float) -> float:
    return jnp.log(jnp.expm1(jnp.asarray(x)))


def _sum_except_batch(x: jax.Array) -> jax.Array:
    return jnp.sum(x.reshape((x.shape[0], -1)), axis=-1)


def _select_bins(values: jax.Array, cum_values: jax.Array) -> jax.Array:
    idx = jnp.sum(values[..., None] >= cum_values[..., 1:-1], axis=-1)
    return jnp.clip(idx, 0, cum_values.shape[-1] - 2)


def _gather_bin(values: jax.Array, idx: jax.Array) -> jax.Array:
    return jnp.take_along_axis(values, idx[..., None], axis=-1)[..., 0]


def _safe_div(numerator: jax.Array, denominator: jax.Array, eps: float = DEFAULT_EPS) -> jax.Array:
    safe_denominator = jnp.where(
        jnp.abs(denominator) < eps,
        jnp.where(denominator >= 0, eps, -eps),
        denominator,
    )
    return numerator / safe_denominator


def unconstrained_rational_quadratic_spline(
    inputs: jax.Array,
    unnormalized_widths: jax.Array,
    unnormalized_heights: jax.Array,
    unnormalized_derivatives: jax.Array,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[jax.Array, jax.Array]:
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs, logabsdet = rational_quadratic_spline(
        jnp.clip(inputs, -tail_bound, tail_bound),
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = jnp.where(inside, outputs, inputs)
    logabsdet = jnp.where(inside, logabsdet, jnp.zeros_like(logabsdet))
    return outputs, logabsdet


def rational_quadratic_spline(
    inputs: jax.Array,
    unnormalized_widths: jax.Array,
    unnormalized_heights: jax.Array,
    unnormalized_derivatives: jax.Array,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[jax.Array, jax.Array]:
    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins >= 1.0:
        raise ValueError("min_bin_width * num_bins must be < 1")
    if min_bin_height * num_bins >= 1.0:
        raise ValueError("min_bin_height * num_bins must be < 1")

    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = jnp.cumsum(widths, axis=-1)
    cumwidths = jnp.pad(cumwidths, [(0, 0)] * (cumwidths.ndim - 1) + [(1, 0)])
    cumwidths = left + (right - left) * cumwidths
    cumwidths = cumwidths.at[..., 0].set(left)
    cumwidths = cumwidths.at[..., -1].set(right)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.pad(cumheights, [(0, 0)] * (cumheights.ndim - 1) + [(1, 0)])
    cumheights = bottom + (top - bottom) * cumheights
    cumheights = cumheights.at[..., 0].set(bottom)
    cumheights = cumheights.at[..., -1].set(top)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = min_derivative + jax.nn.softplus(
        unnormalized_derivatives + _inverse_softplus(1.0 - min_derivative)
    )

    bin_idx = _select_bins(inputs, cumheights if inverse else cumwidths)

    input_cumwidths = _gather_bin(cumwidths, bin_idx)
    input_bin_widths = _gather_bin(widths, bin_idx)
    input_cumheights = _gather_bin(cumheights, bin_idx)
    input_bin_heights = _gather_bin(heights, bin_idx)
    delta = input_bin_heights / input_bin_widths
    input_derivatives = _gather_bin(derivatives, bin_idx)
    input_derivatives_plus_one = _gather_bin(derivatives, bin_idx + 1)

    if inverse:
        y_minus_cumheight = inputs - input_cumheights
        dsum = input_derivatives + input_derivatives_plus_one - 2.0 * delta
        a = y_minus_cumheight * dsum + input_bin_heights * (delta - input_derivatives)
        b = input_bin_heights * input_derivatives - y_minus_cumheight * dsum
        c = -delta * y_minus_cumheight
        discriminant = jnp.maximum(b**2 - 4.0 * a * c, 0.0)
        # Add DEFAULT_EPS**2 inside the sqrt to prevent d/dx[sqrt(x)]→∞ at x=0,
        # which produces NaN gradients whenever the quadratic is near-degenerate.
        # The bias is O(1e-12), negligible for any realistic discriminant value.
        quadratic_root = _safe_div(2.0 * c, -b - jnp.sqrt(discriminant + DEFAULT_EPS**2))
        linear_root = _safe_div(-c, b)
        root = jnp.where(jnp.abs(a) < 1e-12, linear_root, quadratic_root)
        root = jnp.clip(root, 0.0, 1.0)
        outputs = root * input_bin_widths + input_cumwidths
        theta = root
    else:
        theta = _safe_div(inputs - input_cumwidths, input_bin_widths)
        theta_one_minus_theta = theta * (1.0 - theta)
        numerator = input_bin_heights * (
            delta * theta**2 + input_derivatives * theta_one_minus_theta
        )
        denominator = delta + (
            input_derivatives + input_derivatives_plus_one - 2.0 * delta
        ) * theta_one_minus_theta
        outputs = input_cumheights + _safe_div(numerator, denominator)

    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = delta + (
        input_derivatives + input_derivatives_plus_one - 2.0 * delta
    ) * theta_one_minus_theta
    derivative_numer = delta**2 * (
        input_derivatives_plus_one * theta**2
        + 2.0 * delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta) ** 2
    )
    derivative_numer = jnp.maximum(derivative_numer, DEFAULT_EPS)
    denominator = jnp.maximum(denominator, DEFAULT_EPS)
    logabsdet = jnp.log(derivative_numer) - 2.0 * jnp.log(denominator)
    if inverse:
        logabsdet = -logabsdet
    return outputs, logabsdet


class SplineConditioner(nn.Module):
    out_dim: int
    hidden_dim: int
    n_hiddens: int
    num_bins: int
    use_layernorm: bool = True
    dropout_rate: float = 0.0
    activation: str = "silu"

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array, jax.Array]:
        activation_fn = resolve_activation(self.activation)
        h = x
        for _ in range(max(self.n_hiddens, 1)):
            z = nn.Dense(self.hidden_dim)(h)
            z = activation_fn(z)
            if self.use_layernorm:
                z = nn.LayerNorm()(z)
            z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=not train)
            h = h + z if h.shape[-1] == z.shape[-1] else z
        params_per_dim = 3 * self.num_bins + 1
        out = nn.Dense(
            self.out_dim * params_per_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        # No hard clip here: softmax on widths/heights is self-normalising for any
        # input magnitude, and softplus on derivatives has a natural lower bound.
        # A hard clip creates zero-gradient dead zones when RL state inputs drive
        # large activations, which is the main source of training stalls.
        out = out.reshape((x.shape[0], self.out_dim, params_per_dim))
        widths = out[..., : self.num_bins]
        heights = out[..., self.num_bins : 2 * self.num_bins]
        derivatives = out[..., 2 * self.num_bins :]
        return widths, heights, derivatives


class SplineCouplingLayer(nn.Module):
    dim: int
    idx1: jax.Array
    idx2: jax.Array
    hidden_dim: int
    n_hiddens: int
    num_bins: int = 8
    tail_bound: float = 5.0
    use_layernorm: bool = True
    dropout_rate: float = 0.0
    activation: str = "silu"

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array, train: bool, reverse: bool) -> Tuple[jax.Array, jax.Array]:
        x1 = jnp.take(x, self.idx1, axis=1)
        x2 = jnp.take(x, self.idx2, axis=1)
        cond_in = jnp.concatenate([cond, x1], axis=-1)
        widths, heights, derivatives = SplineConditioner(
            out_dim=int(self.idx2.shape[0]),
            hidden_dim=self.hidden_dim,
            n_hiddens=self.n_hiddens,
            num_bins=self.num_bins,
            use_layernorm=self.use_layernorm,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )(cond_in, train)
        y2, ldj = unconstrained_rational_quadratic_spline(
            x2,
            widths,
            heights,
            derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
        )
        y = x.at[:, self.idx2].set(y2)
        return y, _sum_except_batch(ldj)


class NSFActorFlat(nn.Module):
    action_dim: int
    hidden_dim: int
    n_hiddens: int
    num_layers: int
    num_bins: int = 8
    tail_bound: float = 5.0
    scale_max: float = 1.0  # unused; kept for API parity with NFActorFlat
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
                    idx = perm[:half] if select_first else perm[half:]
                else:
                    det_i = i - det_start
                    use_right_as_idx1 = det_i % 2 == 1
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
                SplineCouplingLayer(
                    dim=self.action_dim,
                    idx1=idx1_all[i],
                    idx2=idx2_all[i],
                    hidden_dim=self.hidden_dim,
                    n_hiddens=self.n_hiddens,
                    num_bins=self.num_bins,
                    tail_bound=self.tail_bound,
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
        return actions[0] if single else actions

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
        return actions[0] if single else actions

    def log_prob(self, actions: jax.Array, state: jax.Array, train: bool = False) -> jax.Array:
        cond, single = self._cond(state)
        if actions.ndim == 1:
            actions = actions[None, :]
        pre_tanh = _atanh(actions)
        logdet_tanh = jnp.sum(jnp.log(1.0 - jnp.tanh(pre_tanh) ** 2 + 1e-6), axis=-1)
        z, logdet = self._inverse_flow(pre_tanh, cond, train)
        if self.base_dist == "uniform":
            u = jnp.tanh(z)
            logdet_base = jnp.sum(jnp.log(1.0 - u**2 + 1e-6), axis=-1)
            base_log_prob = logdet_base + (self.action_dim * math.log(0.5))
        else:
            base_log_prob = jnp.sum(_normal_log_prob(z), axis=-1)
        log_prob = base_log_prob + logdet - logdet_tanh
        # Clamp to prevent extreme NLL values from dominating the actor loss,
        # which can happen with out-of-distribution states at the RL stage.
        log_prob = jnp.clip(log_prob, -100.0, 100.0)
        return log_prob[0] if single else log_prob

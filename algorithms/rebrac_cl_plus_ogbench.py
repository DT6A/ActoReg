# source: https://github.com/tinkoff-ai/ReBRAC
# https://arxiv.org/abs/2305.09836

import os
import math
import re
import uuid
import random
import time
import sys
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional

import chex
import flax.linen as nn
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax._src import base, combine, transform, wrappers
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

try:
    from nf_policy_action import NFActorFlat
except ImportError:  # pragma: no cover
    from algorithms.nf_policy_action import NFActorFlat


def _import_ogbench():
    try:
        import ogbench  # type: ignore
        return ogbench
    except ImportError:
        ogbench_repo = os.path.expanduser("~/ogbench")
        if os.path.isdir(ogbench_repo) and ogbench_repo not in sys.path:
            sys.path.append(ogbench_repo)
        try:
            import ogbench  # type: ignore
            return ogbench
        except ImportError as exc:
            raise ImportError(
                "OGBench support requires `ogbench` (and its dependencies, including gymnasium). "
                "Install OGBench or make it importable from ~/ogbench."
            ) from exc


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class Config:
    # wandb params
    project: str = "ReBRAC2"
    group: str = "rebrac2"
    name: str = "rebrac-cl-plus"

    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    critic_dropout: float = 0.0
    gamma: float = 0.99
    tau: float = 5e-3

    actor_bc_coef: float = 0.1
    actor_bc_aux_weight: float = 0.0
    actor_bc_aux_loss: str = "mse"  # mse | mae | sum
    critic_bc_coef: float = 0.0

    actor_ln: bool = False
    actor_fn: bool = False
    actor_gn: bool = False
    actor_bn: bool = False
    actor_sn: bool = False
    critic_ln: bool = True

    actor_dropout: float = 0.1
    actor_wd: float = 0.0
    critic_wd: float = 0.0
    value_wd: float = 0.0
    l1_ratio: float = 0.0
    actor_input_noise: float = 0.0
    actor_bc_noise: float = 0.0
    actor_grad_noise: float = 0.01
    critic_objective_noise: float = 0.0
    critic_grad_noise: float = 0.0
    use_prev_action: bool = False
    use_prev_state: bool = False
    use_actor_ema: bool = False
    actor_ema_tau: float = 5e-3

    actor_reset: bool = False
    actor_prereset_mode: bool = True

    use_nf: bool = True
    nf_num_layers: int = 8
    nf_hidden_dim: int = 128
    nf_n_hiddens: int = 2
    nf_scale_max: float = 1.0
    nf_base_dist: str = "normal"
    nf_use_plu: bool = True
    nf_use_layernorm: bool = True
    nf_dropout: float = 0.1
    nf_det_layers: int = 2
    nf_eval_num_samples: Union[int, Sequence[int], str] = 8
    nf_eval_z_scale: float = 1.0
    nf_eval_z_clip: float = 0.0
    use_target_actor: bool = False

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    optimizer_type: str = "adam"  # adam | adan
    decay_schedule: Optional[str] = None
    num_critics: int = 2
    activation: str = "silu"  # silu | gsp

    # training params
    dataset_name: str = "antmaze-large-navigate-singletask-v0"
    ogbench_dataset_dir: str = "~/.ogbench/data"
    ogbench_goal_source: str = "trajectory_final"  # trajectory_final | env_info | oracle_reps | zeros
    ogbench_eval_task_ids: str = "1,2,3,4,5"
    ogbench_append_goal: bool = True
    ogbench_use_masks_for_dones: bool = True
    batch_size: int = 1024
    num_epochs: int = 1000
    num_refinement_epochs: int = 0
    refinement_div: float = 1.0
    il_warmup_epochs: int = 0
    critic_warmup_epochs: int = 0
    critic_next_state_pred_epochs: int = 0
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False

    # evaluation params
    eval_episodes: int = 50
    eval_every: int = 100
    eval_first_action_only: bool = True
    q_infer_step_size: float = 0.0
    q_infer_steps: Union[int, Sequence[int], str] = 0
    use_likelihood_alpha_target: bool = False
    likelihood_alpha_eps: float = 1e-6
    likelihood_stats_batch_size: int = 1024

    # general params
    train_seed: int = 0
    eval_seed: int = 42

    # critic strategy
    use_distributional: bool = True
    n_classes: int = 101
    sigma_frac: float = 0.75
    v_min: float = float("inf")
    v_max: float = float("inf")
    v_expand: float = 0.05
    v_expand_mode: str = "both"

    # IQL params
    use_iql: bool = False
    value_learning_rate: float = 1e-3
    iql_expectile: float = 0.7

    noisy_eval: bool = False
    mlc_job_name: str = None

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


def pytorch_init(fan_in: float) -> Callable:
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


def identity(x: Any) -> Any:
    return x


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


AddDecayedWeightsState = base.EmptyState


def add_elastic_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    l1_ratio: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    def init_fn(params):
        del params
        return AddDecayedWeightsState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * ((1 - l1_ratio) * p + l1_ratio * jnp.sign(p)),
            updates,
            params,
        )
        return updates, state

    if mask is not None:
        return wrappers.masked(base.GradientTransformation(init_fn, update_fn), mask)
    return base.GradientTransformation(init_fn, update_fn)


def adamw_elastic(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    l1_ratio: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
    return combine.chain(
        transform.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        add_elastic_weights(weight_decay, l1_ratio, mask),
        transform.scale_by_learning_rate(learning_rate),
    )


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = False
    groupnorm: bool = False
    featurenorm: bool = False
    batchnorm: bool = False
    spectralnorm: bool = False
    dropout_rate: float = 0.0
    n_hiddens: int = 3
    activation: str = "silu"

    @nn.compact
    def __call__(self, state: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        s_d, h_d = state.shape[-1], self.hidden_dim
        activation_fn = resolve_activation(self.activation)

        def dense_layer(x, fan_in):
            dense = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(fan_in),
                bias_init=nn.initializers.constant(0.1),
            )
            if self.spectralnorm:
                x = nn.SpectralNorm(dense)(x, update_stats=train)
            else:
                x = dense(x)
            return x

        def apply_block(x):
            x = activation_fn(x)
            x = nn.LayerNorm()(x) if self.layernorm else x
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x) if self.featurenorm else x
            x = nn.GroupNorm()(x) if self.groupnorm else x
            x = nn.BatchNorm(use_running_average=not train)(x) if self.batchnorm else x
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            return x

        x = dense_layer(state, s_d)
        x = apply_block(x)
        for _ in range(max(self.n_hiddens - 1, 0)):
            h = dense_layer(x, h_d)
            h = apply_block(h)
            x = x + h

        trunk = x
        last_layer = nn.Sequential(
            [
                activation_fn,
                nn.LayerNorm() if self.layernorm else identity,
                nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
                nn.GroupNorm() if self.groupnorm else identity,
                nn.BatchNorm(use_running_average=not train) if self.batchnorm else identity,
                nn.Dropout(rate=self.dropout_rate, deterministic=not train),
                nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3)),
                nn.tanh,
            ]
        )
        actions = last_layer(trunk)
        return actions, trunk


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3
    n_classes: int = 21
    use_distributional: bool = True
    dropout_rate: float = 0.0
    activation: str = "silu"

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        train: bool = False,
        predict_next_state: bool = False,
    ) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        state_action = jnp.hstack([state, action])
        activation_fn = resolve_activation(self.activation)

        x = nn.Dense(
            self.hidden_dim,
            kernel_init=pytorch_init(s_d + a_d),
            bias_init=nn.initializers.constant(0.1),
        )(state_action)
        x = activation_fn(x)
        x = nn.LayerNorm()(x) if self.layernorm else x
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        for _ in range(self.n_hiddens - 1):
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(h_d),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            h = activation_fn(h)
            h = nn.LayerNorm()(h) if self.layernorm else h
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
            x = x + h

        def head_block(features: jax.Array, out_dim: int, name: str) -> jax.Array:
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(h_d),
                bias_init=nn.initializers.constant(0.1),
                name=f"{name}_dense1",
            )(features)
            h = activation_fn(h)
            h = nn.LayerNorm(name=f"{name}_ln1")(h) if self.layernorm else h
            h = nn.Dropout(rate=self.dropout_rate, name=f"{name}_drop1")(h, deterministic=not train)

            h = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(h_d),
                bias_init=nn.initializers.constant(0.1),
                name=f"{name}_dense2",
            )(h)
            h = activation_fn(h)
            h = nn.LayerNorm(name=f"{name}_ln2")(h) if self.layernorm else h
            h = nn.Dropout(rate=self.dropout_rate, name=f"{name}_drop2")(h, deterministic=not train)

            return nn.Dense(
                out_dim,
                kernel_init=uniform_init(3e-3),
                bias_init=uniform_init(3e-3),
                name=f"{name}_out",
            )(h)

        q_out = head_block(x, self.n_classes if self.use_distributional else 1, "q_head")
        next_state_out = head_block(x, s_d, "next_state_head")
        if predict_next_state:
            return q_out, next_state_out
        return q_out


class Value(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        layers = [
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d), bias_init=nn.initializers.constant(0.1)),
            nn.silu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
                nn.silu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))]
        network = nn.Sequential(layers)
        return network(state).squeeze(-1)


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 2
    layernorm: bool = True
    n_hiddens: int = 3
    n_classes: int = 21
    use_distributional: bool = True
    dropout_rate: float = 0.0
    activation: str = "silu"

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        train: bool = False,
        predict_next_state: bool = False,
    ) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.num_critics,
        )
        return ensemble(
            self.hidden_dim,
            self.layernorm,
            self.n_hiddens,
            self.n_classes,
            self.use_distributional,
            self.dropout_rate,
            self.activation,
        )(state, action, train, predict_next_state)


def calc_return_to_go(is_sparse_reward, rewards, terminals, gamma):
    if len(rewards) == 0:
        return []
    reward_neg = 0
    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        return [float(reward_neg / (1 - gamma))] * len(rewards)

    return_to_go = [0] * len(rewards)
    prev_return = 0
    for i in range(len(rewards)):
        return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
        prev_return = return_to_go[-i - 1]
    return return_to_go


def _parse_task_ids(task_ids: str) -> Tuple[int, ...]:
    if not task_ids:
        return tuple()
    parsed = []
    for token in task_ids.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token))
    return tuple(parsed)


def _is_singletask_ogbench(dataset_name: str) -> bool:
    return "-singletask" in str(dataset_name)


def _concat_obs_goal(obs: np.ndarray, goals: Optional[np.ndarray], append_goal: bool) -> np.ndarray:
    if (not append_goal) or goals is None:
        return obs.astype(np.float32)
    return np.concatenate([obs.astype(np.float32), goals.astype(np.float32)], axis=-1)


def _trajectory_final_goals(next_obs: np.ndarray, terminals: np.ndarray) -> np.ndarray:
    goals = np.zeros_like(next_obs, dtype=np.float32)
    start = 0
    n = len(next_obs)
    for i in range(n):
        is_end = bool(terminals[i]) or (i == n - 1)
        if is_end:
            goal = next_obs[i].astype(np.float32)
            goals[start : i + 1] = goal
            start = i + 1
    return goals


def _dataset_goals(
    env: gym.Env,
    dataset_name: str,
    dataset: Dict[str, np.ndarray],
    goal_source: str,
) -> Optional[np.ndarray]:
    obs = dataset["observations"].astype(np.float32)
    terminals = dataset["terminals"].astype(bool)

    if goal_source == "zeros":
        return np.zeros_like(obs, dtype=np.float32)

    if goal_source == "oracle_reps" and "oracle_reps" in dataset:
        return dataset["oracle_reps"].astype(np.float32)

    if goal_source == "env_info":
        task_ids = _parse_task_ids("1")
        options = dict(task_id=task_ids[0]) if task_ids else {}
        reset_out = env.reset(options=options) if options else env.reset()
        info = reset_out[1] if isinstance(reset_out, tuple) else {}
        goal = info.get("goal", None) if isinstance(info, dict) else None
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float32)
            return np.repeat(goal[None, :], repeats=len(obs), axis=0)

    # Default and fallback.
    return _trajectory_final_goals(dataset["next_observations"].astype(np.float32), terminals)


def qlearning_dataset(
    dataset_name: str,
    dataset_dir: str,
    normalize_reward: bool = False,
    discount: float = 0.99,
    goal_source: str = "trajectory_final",
    append_goal: bool = True,
    use_masks_for_dones: bool = True,
) -> Tuple[Dict[str, np.ndarray], float, float]:
    ogbench = _import_ogbench()
    env, train_dataset, _ = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=os.path.expanduser(dataset_dir),
        compact_dataset=False,
    )

    if "next_observations" not in train_dataset:
        raise ValueError("OGBench dataset must be loaded with compact_dataset=False")

    if "rewards" not in train_dataset:
        raise ValueError(
            "OGBench dataset does not have rewards. "
            "Use a `*-singletask-*` dataset (or relabeled dataset with rewards/masks)."
        )

    obs = train_dataset["observations"].astype(np.float32)
    next_obs = train_dataset["next_observations"].astype(np.float32)
    actions = train_dataset["actions"].astype(np.float32)
    rewards = train_dataset["rewards"].astype(np.float32)
    traj_terminals = train_dataset["terminals"].astype(np.float32)

    if obs.ndim != 2 or next_obs.ndim != 2:
        raise ValueError("This implementation supports only state-vector OGBench observations.")
    if actions.ndim != 2:
        raise ValueError("This implementation supports only continuous action spaces.")

    if normalize_reward:
        rewards = ReplayBuffer.normalize_reward(dataset_name, rewards)

    if use_masks_for_dones and "masks" in train_dataset:
        dones = (1.0 - train_dataset["masks"].astype(np.float32)).astype(np.float32)
    else:
        dones = traj_terminals.copy()

    goals = _dataset_goals(env, dataset_name, train_dataset, goal_source)
    obs_goal = _concat_obs_goal(obs, goals, append_goal)
    next_obs_goal = _concat_obs_goal(next_obs, goals, append_goal)

    n = obs.shape[0]
    prev_obs = np.zeros_like(obs_goal, dtype=np.float32)
    prev_actions = np.zeros_like(actions, dtype=np.float32)
    prev_valid = np.zeros((n,), dtype=np.float32)

    prev_is_terminal = np.concatenate([[True], traj_terminals[:-1] > 0.5], axis=0)
    valid_idx = np.where(~prev_is_terminal)[0]
    prev_obs[valid_idx] = obs_goal[valid_idx - 1]
    prev_actions[valid_idx] = actions[valid_idx - 1]
    prev_valid[valid_idx] = 1.0

    next_actions = np.zeros_like(actions, dtype=np.float32)
    next_actions[:-1] = actions[1:]
    next_actions[dones > 0.5] = 0.0

    mc_returns = np.zeros((n,), dtype=np.float32)
    ret = 0.0
    for i in range(n - 1, -1, -1):
        ret = float(rewards[i]) + discount * ret * (1.0 - float(dones[i]))
        mc_returns[i] = ret
        if traj_terminals[i] > 0.5:
            ret = 0.0

    train_data = {
        "observations": obs_goal,
        "actions": actions,
        "prev_observations": prev_obs,
        "prev_actions": prev_actions,
        "prev_valid": prev_valid,
        "next_observations": next_obs_goal,
        "next_actions": next_actions,
        "rewards": rewards,
        "terminals": dones,
    }
    return train_data, float(np.min(mc_returns)), float(np.max(mc_returns))


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1
    min: float = 0
    max: float = 1

    def create_from_ogbench(
        self,
        dataset_name: str,
        dataset_dir: str = "~/.ogbench/data",
        normalize_reward: bool = False,
        is_normalize: bool = False,
        discount: float = 0.99,
        goal_source: str = "trajectory_final",
        append_goal: bool = True,
        use_masks_for_dones: bool = True,
    ):
        d4rl_data, self.min, self.max = qlearning_dataset(
            dataset_name,
            dataset_dir=dataset_dir,
            discount=discount,
            normalize_reward=normalize_reward,
            goal_source=goal_source,
            append_goal=append_goal,
            use_masks_for_dones=use_masks_for_dones,
        )
        print("Min/Max", self.min, self.max)

        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "prev_states": jnp.asarray(d4rl_data["prev_observations"], dtype=jnp.float32),
            "prev_actions": jnp.asarray(d4rl_data["prev_actions"], dtype=jnp.float32),
            "prev_valid": jnp.asarray(d4rl_data["prev_valid"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
        }

        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(buffer["next_states"], self.mean, self.std)
            prev_states_norm = normalize_states(buffer["prev_states"], self.mean, self.std)
            buffer["prev_states"] = jnp.where(buffer["prev_valid"][:, None] > 0.5, prev_states_norm, 0.0)
        self.data = buffer

    @property
    def size(self) -> int:
        return self.data["states"].shape[0]

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0
        raise NotImplementedError("Reward normalization is implemented only for AntMaze yet!")


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = {}
        for key, (acc, steps) in self.accumulators.items():
            if key in updates:
                value = updates[key]
                new_accumulators[key] = (acc + value, steps + 1)
            else:
                new_accumulators[key] = (acc, steps)
        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        return {
            k: np.array(jnp.where(v[1] > 0, v[0] / v[1], 0.0))
            for k, v in self.accumulators.items()
        }


def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (arr - mean) / (std + eps)


def build_actor_inputs(
    states: jax.Array,
    prev_states: jax.Array,
    prev_actions: jax.Array,
    use_prev_state: bool,
    use_prev_action: bool,
) -> jax.Array:
    parts = [states]
    if use_prev_state:
        parts.append(prev_states)
    if use_prev_action:
        parts.append(prev_actions)
    return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else states


def parse_eval_num_samples(value: Union[int, Sequence[int], str]) -> Sequence[int]:
    if isinstance(value, int):
        parsed = [value]
    elif isinstance(value, str):
        tokens = [tok for tok in re.split(r"[\s,\[\]]+", value.strip()) if tok]
        parsed = [int(tok) for tok in tokens] if tokens else []
    elif isinstance(value, Sequence):
        parsed = [int(v) for v in value]
    else:
        raise ValueError("nf_eval_num_samples must be an int, a sequence of ints, or a comma-separated string")

    if not parsed:
        raise ValueError("nf_eval_num_samples must contain at least one value")
    if any(v < 1 for v in parsed):
        raise ValueError("nf_eval_num_samples values must be >= 1")

    unique = []
    seen = set()
    for v in parsed:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def parse_q_infer_steps(value: Union[int, Sequence[int], str]) -> Sequence[int]:
    if isinstance(value, int):
        parsed = [value]
    elif isinstance(value, str):
        tokens = [tok for tok in re.split(r"[\s,\[\]]+", value.strip()) if tok]
        parsed = [int(tok) for tok in tokens] if tokens else []
    elif isinstance(value, Sequence):
        parsed = [int(v) for v in value]
    else:
        raise ValueError("q_infer_steps must be an int, a sequence of ints, or a comma-separated string")

    if not parsed:
        raise ValueError("q_infer_steps must contain at least one value")
    if any(v < 0 for v in parsed):
        raise ValueError("q_infer_steps values must be >= 0")

    unique = []
    seen = set()
    for v in parsed:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def compute_dataset_action_logprob_stats(
    actor: "ActorTrainState",
    buffer_data: Dict[str, jax.Array],
    use_prev_state: bool,
    use_prev_action: bool,
    batch_size: int,
) -> Tuple[float, float]:
    if batch_size <= 0:
        raise ValueError("likelihood_stats_batch_size must be > 0")

    @jax.jit
    def _log_prob_batch(params, constants, actor_obs, actions):
        return actor.apply_fn(
            {"params": params, "constants": constants},
            actions,
            actor_obs,
            train=False,
            method=NFActorFlat.log_prob,
        )

    n = int(buffer_data["states"].shape[0])
    lp_min = float("inf")
    lp_max = float("-inf")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        states = buffer_data["states"][start:end]
        prev_states = buffer_data["prev_states"][start:end]
        prev_actions = buffer_data["prev_actions"][start:end]
        actions = buffer_data["actions"][start:end]
        actor_obs = build_actor_inputs(states, prev_states, prev_actions, use_prev_state, use_prev_action)
        logp = np.asarray(jax.device_get(_log_prob_batch(actor.params, actor.constants, actor_obs, actions)))
        lp_min = min(lp_min, float(np.min(logp)))
        lp_max = max(lp_max, float(np.max(logp)))

    if not np.isfinite(lp_min) or not np.isfinite(lp_max):
        raise ValueError("Failed to compute finite dataset action likelihood statistics")
    return lp_min, lp_max


def transform_to_probs(target: jax.Array, support: jax.Array, sigma: float) -> jax.Array:
    cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    bin_probs = cdf_evals[1:] - cdf_evals[:-1]
    return bin_probs / (z + 1e-6)


transform_to_probs = jax.vmap(transform_to_probs, in_axes=(0, None, None))


def transform_from_probs(probs: jax.Array, support: jax.Array) -> jax.Array:
    centers = (support[:-1] + support[1:]) / 2
    return jnp.sum(probs * centers)


transform_from_probs = jax.vmap(transform_from_probs, in_axes=(0, None))
transform_from_probs = jax.vmap(transform_from_probs, in_axes=(0, None))


def make_env(env_name: str, seed: int, dataset_dir: str) -> gym.Env:
    ogbench = _import_ogbench()
    env = ogbench.make_env_and_datasets(
        env_name,
        dataset_dir=os.path.expanduser(dataset_dir),
        env_only=True,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (state - state_mean) / state_std

    def scale_reward(reward: float) -> float:
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def evaluate(
    env: gym.Env,
    params: jax.Array,
    batch_stats: jax.Array,
    constants: Any,
    critic: "CriticTrainState",
    action_fn: Callable,
    log_prob_fn: Optional[Callable],
    num_episodes: int,
    seed: int,
    state_mean: jax.Array,
    state_std: jax.Array,
    action_noise: float = 0.0,
    state_noise: float = 0.0,
    q_infer_step_size: float = 0.0,
    q_infer_steps: int = 0,
    eval_task_ids: Tuple[int, ...] = (),
    append_goal: bool = True,
    goal_source: str = "trajectory_final",
    use_prev_state: bool = False,
    use_prev_action: bool = False,
    use_nf: bool = False,
    use_distributional: bool = True,
    nf_eval_num_samples: int = 1,
    nf_eval_z_scale: float = 1.0,
    nf_eval_z_clip: float = 0.0,
    nf_eval_select: str = "q",
) -> Tuple[np.ndarray, Dict]:
    del goal_source
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    key = jax.random.PRNGKey(seed=seed)
    use_refine = q_infer_step_size > 0 and q_infer_steps > 0
    q_infer_steps = max(0, int(q_infer_steps))
    state_mean = np.asarray(state_mean, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)

    def _make_state(obs_raw: np.ndarray, goal_raw: Optional[np.ndarray]) -> np.ndarray:
        obs_raw = np.asarray(obs_raw, dtype=np.float32)
        if not append_goal:
            return obs_raw
        if goal_raw is None:
            goal_raw = np.zeros_like(obs_raw, dtype=np.float32)
        goal_raw = np.asarray(goal_raw, dtype=np.float32)
        return np.concatenate([obs_raw, goal_raw], axis=-1)

    def _normalize_state_vec(state: np.ndarray) -> np.ndarray:
        return (state - state_mean) / (state_std + 1e-8)

    @partial(jax.jit, static_argnums=(5,))
    def policy_action(params_j, batch_stats_j, constants_j, obs_j, rng_j, num_samples_j):
        return action_fn(
            params_j,
            batch_stats_j,
            constants_j,
            obs_j,
            rng_j,
            num_samples_j,
            nf_eval_z_scale,
            nf_eval_z_clip,
        )

    @jax.jit
    def eval_q(obs_j, action_j):
        logits = critic.apply_fn(critic.params, obs_j, action_j)
        if use_distributional:
            probs = nn.softmax(logits, axis=-1)
            q_values = transform_from_probs(probs, critic.support).min(0)
        else:
            q_values = jnp.squeeze(logits, axis=-1)
            if q_values.ndim > 1:
                q_values = q_values.min(0)
        return q_values

    def _refine_action(obs_j, action_j):
        def q_value_sum(a):
            return jnp.sum(eval_q(obs_j, a))

        def body(_, a):
            grad = jax.grad(q_value_sum)(a)
            grad_norm = jnp.linalg.norm(grad, axis=-1, keepdims=True) + 1e-8
            return jnp.clip(a + q_infer_step_size * (grad / grad_norm), -1.0, 1.0)

        return jax.lax.fori_loop(0, q_infer_steps, body, action_j)

    refine_action = jax.jit(_refine_action)

    returns = []
    eval_states = []
    eval_actions = []
    eval_prev_states = []
    eval_prev_actions = []

    for ep_idx in trange(num_episodes, desc="Eval", leave=False):
        reset_kwargs = {}
        if len(eval_task_ids) > 0:
            reset_kwargs["options"] = dict(task_id=eval_task_ids[ep_idx % len(eval_task_ids)])
        if reset_kwargs:
            try:
                reset_out = env.reset(**reset_kwargs)
            except Exception:
                reset_out = env.reset()
        else:
            reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs_raw, info = reset_out
        else:
            obs_raw, info = reset_out, {}
        goal_raw = info.get("goal", None) if isinstance(info, dict) else None

        obs_state = _make_state(obs_raw, goal_raw)
        obs_norm = _normalize_state_vec(obs_state)
        done = False
        total_reward = 0.0
        prev_obs = np.zeros_like(obs_norm, dtype=np.float32)
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

        while not done:
            key, actions_key, states_key = jax.random.split(key, 3)
            obs_for_actor = obs_norm + jax.random.normal(states_key, obs_norm.shape) * state_noise
            actor_obs = build_actor_inputs(
                jnp.asarray(obs_for_actor),
                jnp.asarray(prev_obs),
                jnp.asarray(prev_action),
                use_prev_state,
                use_prev_action,
            )
            eval_states.append(obs_norm)
            eval_prev_states.append(prev_obs)
            eval_prev_actions.append(prev_action)

            action = np.asarray(
                jax.device_get(
                    policy_action(params, batch_stats, constants, actor_obs, actions_key, nf_eval_num_samples)
                )
            )
            if use_nf and nf_eval_num_samples > 1:
                candidates = jnp.asarray(action)
                obs_j = jnp.asarray(obs_for_actor)[None, ...]
                actor_obs_j = jnp.asarray(actor_obs)[None, ...]
                if use_refine:
                    obs_rep = jnp.repeat(obs_j, nf_eval_num_samples, axis=0)
                    candidates = refine_action(obs_rep, candidates)
                if nf_eval_select == "likelihood":
                    if log_prob_fn is None:
                        raise ValueError("nf_eval_select='likelihood' requires log_prob_fn")
                    actor_obs_rep = jnp.repeat(actor_obs_j, nf_eval_num_samples, axis=0)
                    logp = log_prob_fn(params, batch_stats, constants, actor_obs_rep, candidates)
                    best_idx = int(jax.device_get(jnp.argmax(logp)))
                else:
                    obs_rep = jnp.repeat(obs_j, nf_eval_num_samples, axis=0)
                    q_vals = eval_q(obs_rep, candidates)
                    best_idx = int(jax.device_get(jnp.argmax(q_vals)))
                action = np.asarray(jax.device_get(candidates[best_idx]))
            elif use_refine:
                obs_j = jnp.asarray(obs_for_actor)[None, ...]
                action_j = jnp.asarray(action)[None, ...]
                action = np.asarray(jax.device_get(refine_action(obs_j, action_j)[0]))

            eval_actions.append(action)
            executed_action = np.asarray(
                jax.device_get(jnp.clip(action + jax.random.normal(actions_key, action.shape) * action_noise, -1, 1))
            )

            step_result = env.step(executed_action)
            if len(step_result) == 5:
                next_obs_raw, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_obs_raw, reward, done, _ = step_result

            prev_obs = np.asarray(obs_norm, dtype=np.float32)
            prev_action = np.asarray(executed_action, dtype=np.float32)
            obs_state = _make_state(next_obs_raw, goal_raw)
            obs_norm = _normalize_state_vec(obs_state)
            total_reward += reward

        returns.append(total_reward)

    eval_batch = {
        "states": jnp.array(eval_states),
        "actions": jnp.array(eval_actions),
        "prev_states": jnp.array(eval_prev_states),
        "prev_actions": jnp.array(eval_prev_actions),
    }
    return np.array(returns), eval_batch


class CriticTrainState(TrainState):
    target_params: FrozenDict
    support: jax.Array
    sigma: float


class ActorTrainState(TrainState):
    target_params: FrozenDict
    ema_params: FrozenDict
    dropout_key: jax.Array
    batch_stats: Any
    target_batch_stats: Any
    ema_batch_stats: Any
    constants: Any = None
    target_constants: Any = None
    ema_constants: Any = None


class ValueTrainState(TrainState):
    pass


def update_actor(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    beta: float,
    aux_weight: float,
    aux_loss: str,
    tau: float,
    ema_tau: float,
    normalize_q: bool,
    input_noise: float,
    bc_noise: float,
    grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    use_nf: bool,
    use_distributional: bool,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Metrics]:
    key, random_action_key, input_noise_key, bc_noise_key, grad_noise_key = jax.random.split(key, 5)
    dropout_key, new_dropout_key = jax.random.split(actor.dropout_key, 2)
    sample_key, dropout_apply_key, dropout_log_key = jax.random.split(dropout_key, 3)

    in_noise = jax.random.normal(input_noise_key, batch["states"].shape) * input_noise
    b_noise = jax.random.normal(bc_noise_key, batch["actions"].shape) * bc_noise
    actor_inputs = build_actor_inputs(
        batch["states"] + in_noise,
        batch["prev_states"],
        batch["prev_actions"],
        use_prev_state,
        use_prev_action,
    )

    def actor_loss_fn(params: jax.Array):
        if use_nf:
            actions = actor.apply_fn(
                {"params": params, "constants": actor.constants},
                actor_inputs,
                rng=sample_key,
                train=True,
                method=NFActorFlat.sample,
                rngs={"dropout": dropout_apply_key},
            )
            updates = {"batch_stats": actor.batch_stats}
            bc_actions = batch["actions"] + b_noise
            log_probs = actor.apply_fn(
                {"params": params, "constants": actor.constants},
                bc_actions,
                actor_inputs,
                train=True,
                method=NFActorFlat.log_prob,
                rngs={"dropout": dropout_log_key},
            )
            bc_penalty = -log_probs
            nll = jnp.mean(bc_penalty)
        else:
            (actions, _), updates = actor.apply_fn(
                {"params": params, "batch_stats": actor.batch_stats},
                actor_inputs,
                True,
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
            )
            bc_penalty = jnp.sum((actions - batch["actions"] + b_noise) ** 2, axis=-1)
            nll = jnp.mean(bc_penalty)

        diff = actions - batch["actions"]
        mse = jnp.mean(diff ** 2, axis=-1)
        mae = jnp.mean(jnp.abs(diff), axis=-1)
        if aux_loss == "mae":
            aux_bc = mae
        elif aux_loss == "sum":
            aux_bc = mse + mae
        else:
            aux_bc = mse

        logits = critic.apply_fn(critic.params, batch["states"], actions, False)
        if use_distributional:
            probs = nn.softmax(logits, axis=-1)
            q_values = transform_from_probs(probs, critic.support).min(0)
        else:
            q_values = jnp.squeeze(logits, axis=-1)
            if q_values.ndim > 1:
                q_values = q_values.min(0)

        lmbda = 1.0
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1 / (jnp.abs(q_values).mean() + 1e-6))

        bc_term = beta * bc_penalty
        aux_term = aux_weight * aux_bc
        q_term = -lmbda * q_values
        loss = (bc_term + aux_term + q_term).mean()

        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        metrics_payload = {
            "actor_loss": loss,
            "actor_loss_bc_term": bc_term.mean(),
            "actor_loss_aux_term": aux_term.mean(),
            "actor_loss_q_term": q_term.mean(),
            "actor_loss_lmbda": lmbda,
            "bc_mse_policy": jnp.mean((actions - batch["actions"] + b_noise) ** 2),
            "bc_mse_random": jnp.mean((random_actions - batch["actions"]) ** 2),
            "action_mse": jnp.mean((actions - batch["actions"]) ** 2),
            "aux_bc": aux_bc.mean(),
            "nll": nll,
        }
        new_metrics = metrics.update(metrics_payload)
        return loss, (updates, new_metrics)

    grads, (updates, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

    def add_gaussian_noise(gr, noise_std, rng_key):
        def add_noise_to_grad(g, k):
            noise = jax.random.normal(k, g.shape) * noise_std / ((1 + actor.step) ** 0.55)
            return g + noise

        leaves, tree = jax.tree_util.tree_flatten(gr)
        rng_keys = jax.random.split(rng_key, num=len(leaves))
        rng_keys = jax.tree_util.tree_unflatten(tree, rng_keys)
        return jax.tree_util.tree_map(lambda g, k: add_noise_to_grad(g, k), gr, rng_keys)

    grads = add_gaussian_noise(grads, grad_noise, grad_noise_key)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(batch_stats=updates["batch_stats"])
    new_actor = new_actor.replace(
        target_params=optax.incremental_update(new_actor.params, actor.target_params, tau),
        target_batch_stats=optax.incremental_update(new_actor.batch_stats, actor.target_batch_stats, tau),
        ema_params=optax.incremental_update(new_actor.params, actor.ema_params, ema_tau),
        ema_batch_stats=optax.incremental_update(new_actor.batch_stats, actor.ema_batch_stats, ema_tau),
        target_constants=actor.target_constants,
        ema_constants=actor.ema_constants,
        dropout_key=new_dropout_key,
    )
    new_critic = critic.replace(target_params=optax.incremental_update(critic.params, critic.target_params, tau))
    return key, new_actor, new_critic, new_metrics


def update_actor_bc(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    batch: Dict[str, jax.Array],
    beta: float,
    aux_weight: float,
    aux_loss: str,
    tau: float,
    ema_tau: float,
    input_noise: float,
    bc_noise: float,
    grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    use_nf: bool,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, Metrics]:
    key, random_action_key, input_noise_key, bc_noise_key, grad_noise_key = jax.random.split(key, 5)
    dropout_key, new_dropout_key = jax.random.split(actor.dropout_key, 2)
    sample_key, dropout_apply_key, dropout_log_key = jax.random.split(dropout_key, 3)

    in_noise = jax.random.normal(input_noise_key, batch["states"].shape) * input_noise
    b_noise = jax.random.normal(bc_noise_key, batch["actions"].shape) * bc_noise
    actor_inputs = build_actor_inputs(
        batch["states"] + in_noise,
        batch["prev_states"],
        batch["prev_actions"],
        use_prev_state,
        use_prev_action,
    )

    def actor_loss_fn(params: jax.Array):
        if use_nf:
            actions = actor.apply_fn(
                {"params": params, "constants": actor.constants},
                actor_inputs,
                rng=sample_key,
                train=True,
                method=NFActorFlat.sample,
                rngs={"dropout": dropout_apply_key},
            )
            updates = {"batch_stats": actor.batch_stats}
            bc_actions = batch["actions"] + b_noise
            log_probs = actor.apply_fn(
                {"params": params, "constants": actor.constants},
                bc_actions,
                actor_inputs,
                train=True,
                method=NFActorFlat.log_prob,
                rngs={"dropout": dropout_log_key},
            )
            bc_penalty = -log_probs
            nll = jnp.mean(bc_penalty)
        else:
            (actions, _), updates = actor.apply_fn(
                {"params": params, "batch_stats": actor.batch_stats},
                actor_inputs,
                True,
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
            )
            bc_penalty = jnp.sum((actions - batch["actions"] + b_noise) ** 2, axis=-1)
            nll = jnp.mean(bc_penalty)

        diff = actions - batch["actions"]
        mse = jnp.mean(diff ** 2, axis=-1)
        mae = jnp.mean(jnp.abs(diff), axis=-1)
        if aux_loss == "mae":
            aux_bc = mae
        elif aux_loss == "sum":
            aux_bc = mse + mae
        else:
            aux_bc = mse

        bc_term = beta * bc_penalty
        aux_term = aux_weight * aux_bc
        loss = (bc_term + aux_term).mean()

        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        metrics_payload = {
            "actor_loss": loss,
            "actor_loss_bc_term": bc_term.mean(),
            "actor_loss_aux_term": aux_term.mean(),
            "bc_mse_policy": jnp.mean((actions - batch["actions"] + b_noise) ** 2),
            "bc_mse_random": jnp.mean((random_actions - batch["actions"]) ** 2),
            "action_mse": jnp.mean((actions - batch["actions"]) ** 2),
            "aux_bc": aux_bc.mean(),
            "nll": nll,
        }
        new_metrics = metrics.update(metrics_payload)
        return loss, (updates, new_metrics)

    grads, (updates, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

    def add_gaussian_noise(gr, noise_std, rng_key):
        def add_noise_to_grad(g, k):
            noise = jax.random.normal(k, g.shape) * noise_std / ((1 + actor.step) ** 0.55)
            return g + noise

        leaves, tree = jax.tree_util.tree_flatten(gr)
        rng_keys = jax.random.split(rng_key, num=len(leaves))
        rng_keys = jax.tree_util.tree_unflatten(tree, rng_keys)
        return jax.tree_util.tree_map(lambda g, k: add_noise_to_grad(g, k), gr, rng_keys)

    grads = add_gaussian_noise(grads, grad_noise, grad_noise_key)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(
        batch_stats=updates["batch_stats"],
        target_params=optax.incremental_update(new_actor.params, actor.target_params, tau),
        target_batch_stats=optax.incremental_update(new_actor.batch_stats, actor.target_batch_stats, tau),
        ema_params=optax.incremental_update(new_actor.params, actor.ema_params, ema_tau),
        ema_batch_stats=optax.incremental_update(new_actor.batch_stats, actor.ema_batch_stats, ema_tau),
        target_constants=actor.target_constants,
        ema_constants=actor.ema_constants,
        dropout_key=new_dropout_key,
    )
    return key, new_actor, new_metrics


def update_value(
    value: ValueTrainState,
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    expectile: float,
    use_distributional: bool,
) -> Tuple[ValueTrainState, jax.Array]:
    def value_loss_fn(value_params: jax.Array):
        v = value.apply_fn(value_params, batch["states"])
        logits = critic.apply_fn(critic.params, batch["states"], batch["actions"], False)
        if use_distributional:
            probs = nn.softmax(logits, axis=-1)
            q_values = transform_from_probs(probs, critic.support).min(0)
        else:
            q_values = jnp.squeeze(logits, axis=-1)
            if q_values.ndim > 1:
                q_values = q_values.min(0)
        diff = q_values - v
        weight = jnp.where(diff > 0, expectile, 1 - expectile)
        loss = (weight * (diff ** 2)).mean()
        return loss, v

    (loss, _), grads = jax.value_and_grad(value_loss_fn, has_aux=True)(value.params)
    new_value = value.apply_gradients(grads=grads)
    return new_value, loss


def update_critic_iql(
    key: jax.random.PRNGKey,
    critic: CriticTrainState,
    value: ValueTrainState,
    batch: Dict[str, jax.Array],
    gamma: float,
    objective_noise: float,
    grad_noise: float,
    use_distributional: bool,
    epoch: jax.Array,
    next_state_pred_epochs: int,
) -> Tuple[jax.random.PRNGKey, CriticTrainState, jax.Array, jax.Array, jax.Array]:
    key, critic_dropout_key, objective_noise_key, grad_noise_key = jax.random.split(key, 4)
    state_noise_key, action_noise_key = jax.random.split(objective_noise_key)
    state_noise = jax.random.normal(state_noise_key, batch["states"].shape) * objective_noise
    action_noise = jax.random.normal(action_noise_key, batch["actions"].shape) * objective_noise
    v_next = value.apply_fn(value.params, batch["next_states"])
    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * v_next
    next_state_coef = jnp.where((next_state_pred_epochs > 0) & (epoch < next_state_pred_epochs), 1.0, 0.0)

    def critic_loss_fn(critic_params: jax.Array):
        q, next_state_pred = critic.apply_fn(
            critic_params,
            batch["states"] + state_noise,
            jnp.clip(batch["actions"] + action_noise, -1.0, 1.0),
            True,
            True,
            rngs={"dropout": critic_dropout_key},
        )
        if use_distributional:
            q_min = transform_from_probs(nn.softmax(q, axis=-1), critic.support).min(0).mean()
            target_probs = transform_to_probs(target_q, critic.support, critic.sigma)
            q_loss = optax.softmax_cross_entropy(logits=q, labels=target_probs[None, ...]).mean(1).sum(0)
        else:
            q = jnp.squeeze(q, axis=-1)
            q_min = q.min(0).mean() if q.ndim > 1 else q.mean()
            q_loss = jnp.mean((q - target_q) ** 2, axis=1).sum(0) if q.ndim > 1 else jnp.mean((q - target_q) ** 2)

        target_next_state = batch["next_states"]
        if next_state_pred.ndim > 2:
            next_state_loss = jnp.mean((next_state_pred - target_next_state[None, ...]) ** 2, axis=(1, 2)).sum(0)
        else:
            next_state_loss = jnp.mean((next_state_pred - target_next_state) ** 2)
        next_state_loss = next_state_coef * next_state_loss

        loss = q_loss + next_state_loss
        return loss, (q_min, next_state_loss)

    (loss, (q_min, next_state_loss)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)

    def add_gaussian_noise(gr, noise_std, rng_key):
        def add_noise_to_grad(g, k):
            noise = jax.random.normal(k, g.shape) * noise_std / ((1 + critic.step) ** 0.55)
            return g + noise

        leaves, tree = jax.tree_util.tree_flatten(gr)
        rng_keys = jax.random.split(rng_key, num=len(leaves))
        rng_keys = jax.tree_util.tree_unflatten(tree, rng_keys)
        return jax.tree_util.tree_map(lambda g, k: add_noise_to_grad(g, k), gr, rng_keys)

    grads = add_gaussian_noise(grads, grad_noise, grad_noise_key)
    new_critic = critic.apply_gradients(grads=grads)
    return key, new_critic, loss, q_min, next_state_loss


def update_critic(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    gamma: float,
    beta: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    objective_noise: float,
    grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    use_target_actor: bool,
    use_nf: bool,
    use_distributional: bool,
    epoch: jax.Array,
    next_state_pred_epochs: int,
    use_likelihood_alpha_target: bool,
    likelihood_alpha_eps: float,
    likelihood_logprob_min: jax.Array,
    likelihood_logprob_max: jax.Array,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, CriticTrainState, Metrics]:
    key, actions_key, noise_key, critic_dropout_key, objective_noise_key, grad_noise_key = jax.random.split(key, 6)
    state_noise_key, action_noise_key = jax.random.split(objective_noise_key)
    state_noise = jax.random.normal(state_noise_key, batch["states"].shape) * objective_noise
    action_noise = jax.random.normal(action_noise_key, batch["actions"].shape) * objective_noise

    actor_params = actor.target_params if use_target_actor else actor.params
    actor_batch_stats = actor.target_batch_stats if use_target_actor else actor.batch_stats
    actor_constants = actor.target_constants if use_target_actor else actor.constants
    next_prev_states = jnp.where(batch["dones"][:, None] > 0.5, jnp.zeros_like(batch["states"]), batch["states"])
    next_prev_actions = jnp.where(batch["dones"][:, None] > 0.5, jnp.zeros_like(batch["actions"]), batch["actions"])
    next_actor_inputs = build_actor_inputs(
        batch["next_states"],
        next_prev_states,
        next_prev_actions,
        use_prev_state,
        use_prev_action,
    )

    if use_nf:
        next_actions = actor.apply_fn(
            {"params": actor_params, "constants": actor_constants},
            next_actor_inputs,
            rng=actions_key,
            train=False,
            method=NFActorFlat.sample,
        )
    else:
        next_actions, _ = actor.apply_fn(
            {"params": actor_params, "batch_stats": actor_batch_stats},
            next_actor_inputs,
            False,
        )

    noise = jnp.clip(jax.random.normal(noise_key, next_actions.shape) * policy_noise, -noise_clip, noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1, 1)
    bc_penalty = jnp.sum((next_actions - batch["next_actions"]) ** 2, axis=-1)

    logits = critic.apply_fn(critic.target_params, batch["next_states"], next_actions, False)
    if use_distributional:
        probs = nn.softmax(logits, axis=-1)
        next_q = transform_from_probs(probs, critic.support).min(0)
    else:
        next_q = jnp.squeeze(logits, axis=-1)
        if next_q.ndim > 1:
            next_q = next_q.min(0)
    next_q = next_q - beta * bc_penalty

    alpha = jnp.ones_like(next_q)
    if use_likelihood_alpha_target and use_nf:
        next_logp = actor.apply_fn(
            {"params": actor_params, "constants": actor_constants},
            next_actions,
            next_actor_inputs,
            train=False,
            method=NFActorFlat.log_prob,
        )
        denom = jnp.maximum(likelihood_logprob_max - likelihood_logprob_min, likelihood_alpha_eps)
        alpha = (next_logp - likelihood_logprob_min) / denom
    alpha_mean = jnp.mean(alpha)
    alpha_min = jnp.min(alpha)
    alpha_max = jnp.max(alpha)

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q * alpha
    next_state_coef = jnp.where((next_state_pred_epochs > 0) & (epoch < next_state_pred_epochs), 1.0, 0.0)

    def critic_loss_fn(critic_params: jax.Array):
        q, next_state_pred = critic.apply_fn(
            critic_params,
            batch["states"] + state_noise,
            jnp.clip(batch["actions"] + action_noise, -1.0, 1.0),
            True,
            True,
            rngs={"dropout": critic_dropout_key},
        )
        if use_distributional:
            q_min = transform_from_probs(nn.softmax(q, axis=-1), critic.support).min(0).mean()
            target_probs = transform_to_probs(target_q, critic.support, critic.sigma)
            q_loss = optax.softmax_cross_entropy(logits=q, labels=target_probs[None, ...]).mean(1).sum(0)
        else:
            q = jnp.squeeze(q, axis=-1)
            q_min = q.min(0).mean() if q.ndim > 1 else q.mean()
            q_loss = jnp.mean((q - target_q) ** 2, axis=1).sum(0) if q.ndim > 1 else jnp.mean((q - target_q) ** 2)

        target_next_state = batch["next_states"]
        if next_state_pred.ndim > 2:
            next_state_loss = jnp.mean((next_state_pred - target_next_state[None, ...]) ** 2, axis=(1, 2)).sum(0)
        else:
            next_state_loss = jnp.mean((next_state_pred - target_next_state) ** 2)
        next_state_loss = next_state_coef * next_state_loss

        loss = q_loss + next_state_loss
        return loss, (q_min, next_state_loss)

    (loss, (q_min, next_state_loss)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)

    def add_gaussian_noise(gr, noise_std, rng_key):
        def add_noise_to_grad(g, k):
            noise = jax.random.normal(k, g.shape) * noise_std / ((1 + critic.step) ** 0.55)
            return g + noise

        leaves, tree = jax.tree_util.tree_flatten(gr)
        rng_keys = jax.random.split(rng_key, num=len(leaves))
        rng_keys = jax.tree_util.tree_unflatten(tree, rng_keys)
        return jax.tree_util.tree_map(lambda g, k: add_noise_to_grad(g, k), gr, rng_keys)

    grads = add_gaussian_noise(grads, grad_noise, grad_noise_key)
    new_critic = critic.apply_gradients(grads=grads)
    new_metrics = metrics.update(
        {
            "critic_loss": loss,
            "q_min": q_min,
            "critic_next_state_loss": next_state_loss,
            "alpha_mean": alpha_mean,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
        }
    )
    return key, new_critic, new_metrics


def update_td3(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    epoch: jax.Array,
    gamma: float,
    actor_bc_coef: float,
    actor_bc_aux_weight: float,
    actor_bc_aux_loss: str,
    critic_bc_coef: float,
    tau: float,
    actor_ema_tau: float,
    policy_noise: float,
    noise_clip: float,
    critic_objective_noise: float,
    critic_grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    normalize_q: bool,
    actor_input_noise: float,
    actor_bc_noise: float,
    actor_grad_noise: float,
    use_target_actor: bool,
    use_nf: bool,
    use_distributional: bool,
    next_state_pred_epochs: int,
    use_likelihood_alpha_target: bool,
    likelihood_alpha_eps: float,
    likelihood_logprob_min: jax.Array,
    likelihood_logprob_max: jax.Array,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Metrics]:
    key, new_critic, new_metrics = update_critic(
        key,
        actor,
        critic,
        batch,
        gamma,
        critic_bc_coef,
        tau,
        policy_noise,
        noise_clip,
        critic_objective_noise,
        critic_grad_noise,
        use_prev_state,
        use_prev_action,
        use_target_actor,
        use_nf,
        use_distributional,
        epoch,
        next_state_pred_epochs,
        use_likelihood_alpha_target,
        likelihood_alpha_eps,
        likelihood_logprob_min,
        likelihood_logprob_max,
        metrics,
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key,
        actor,
        new_critic,
        batch,
        actor_bc_coef,
        actor_bc_aux_weight,
        actor_bc_aux_loss,
        tau,
        actor_ema_tau,
        normalize_q,
        actor_input_noise,
        actor_bc_noise,
        actor_grad_noise,
        use_prev_state,
        use_prev_action,
        use_nf,
        use_distributional,
        new_metrics,
    )
    return key, new_actor, new_critic, new_metrics


def update_iql(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    value: ValueTrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    epoch: jax.Array,
    gamma: float,
    actor_bc_coef: float,
    actor_bc_aux_weight: float,
    actor_bc_aux_loss: str,
    tau: float,
    actor_ema_tau: float,
    iql_expectile: float,
    critic_objective_noise: float,
    critic_grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    normalize_q: bool,
    actor_input_noise: float,
    actor_bc_noise: float,
    actor_grad_noise: float,
    use_nf: bool,
    use_distributional: bool,
    next_state_pred_epochs: int,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, ValueTrainState, Metrics]:
    new_value, value_loss = update_value(value, critic, batch, iql_expectile, use_distributional)
    key, critic_key = jax.random.split(key)
    key, new_critic, critic_loss, q_min, critic_next_state_loss = update_critic_iql(
        critic_key,
        critic,
        new_value,
        batch,
        gamma,
        critic_objective_noise,
        critic_grad_noise,
        use_distributional,
        epoch,
        next_state_pred_epochs,
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key,
        actor,
        new_critic,
        batch,
        actor_bc_coef,
        actor_bc_aux_weight,
        actor_bc_aux_loss,
        tau,
        actor_ema_tau,
        normalize_q,
        actor_input_noise,
        actor_bc_noise,
        actor_grad_noise,
        use_prev_state,
        use_prev_action,
        use_nf,
        use_distributional,
        metrics,
    )
    new_metrics = new_metrics.update(
        {"critic_loss": critic_loss, "q_min": q_min, "critic_next_state_loss": critic_next_state_loss, "value_loss": value_loss}
    )
    return key, new_actor, new_critic, new_value, new_metrics


def update_iql_no_actor(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    value: ValueTrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    epoch: jax.Array,
    gamma: float,
    iql_expectile: float,
    critic_objective_noise: float,
    critic_grad_noise: float,
    use_distributional: bool,
    next_state_pred_epochs: int,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, ValueTrainState, Metrics]:
    new_value, value_loss = update_value(value, critic, batch, iql_expectile, use_distributional)
    key, critic_key = jax.random.split(key)
    key, new_critic, critic_loss, q_min, critic_next_state_loss = update_critic_iql(
        critic_key,
        critic,
        new_value,
        batch,
        gamma,
        critic_objective_noise,
        critic_grad_noise,
        use_distributional,
        epoch,
        next_state_pred_epochs,
    )
    new_metrics = metrics.update(
        {"critic_loss": critic_loss, "q_min": q_min, "critic_next_state_loss": critic_next_state_loss, "value_loss": value_loss}
    )
    return key, actor, new_critic, new_value, new_metrics


def update_td3_no_targets(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    gamma: float,
    metrics: Metrics,
    epoch: jax.Array,
    actor_bc_coef: float,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    critic_objective_noise: float,
    critic_grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    use_target_actor: bool,
    use_nf: bool,
    use_distributional: bool,
    next_state_pred_epochs: int,
    use_likelihood_alpha_target: bool,
    likelihood_alpha_eps: float,
    likelihood_logprob_min: jax.Array,
    likelihood_logprob_max: jax.Array,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Metrics]:
    key, new_critic, new_metrics = update_critic(
        key,
        actor,
        critic,
        batch,
        gamma,
        critic_bc_coef,
        tau,
        policy_noise,
        noise_clip,
        critic_objective_noise,
        critic_grad_noise,
        use_prev_state,
        use_prev_action,
        use_target_actor,
        use_nf,
        use_distributional,
        epoch,
        next_state_pred_epochs,
        use_likelihood_alpha_target,
        likelihood_alpha_eps,
        likelihood_logprob_min,
        likelihood_logprob_max,
        metrics,
    )
    return key, actor, new_critic, new_metrics


def update_critic_warmup(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    gamma: float,
    metrics: Metrics,
    epoch: jax.Array,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    critic_objective_noise: float,
    critic_grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    use_target_actor: bool,
    use_nf: bool,
    use_distributional: bool,
    next_state_pred_epochs: int,
    use_likelihood_alpha_target: bool,
    likelihood_alpha_eps: float,
    likelihood_logprob_min: jax.Array,
    likelihood_logprob_max: jax.Array,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Metrics]:
    key, new_critic, new_metrics = update_critic(
        key,
        actor,
        critic,
        batch,
        gamma,
        critic_bc_coef,
        tau,
        policy_noise,
        noise_clip,
        critic_objective_noise,
        critic_grad_noise,
        use_prev_state,
        use_prev_action,
        use_target_actor,
        use_nf,
        use_distributional,
        epoch,
        next_state_pred_epochs,
        use_likelihood_alpha_target,
        likelihood_alpha_eps,
        likelihood_logprob_min,
        likelihood_logprob_max,
        metrics,
    )
    new_critic = new_critic.replace(target_params=optax.incremental_update(new_critic.params, critic.target_params, tau))
    return key, actor, new_critic, new_metrics


def update_refinement(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    gamma: float,
    actor_bc_coef: float,
    actor_bc_aux_weight: float,
    actor_bc_aux_loss: str,
    critic_bc_coef: float,
    tau: float,
    actor_ema_tau: float,
    policy_noise: float,
    noise_clip: float,
    normalize_q: bool,
    actor_input_noise: float,
    actor_bc_noise: float,
    actor_grad_noise: float,
    use_prev_state: bool,
    use_prev_action: bool,
    use_nf: bool,
    use_distributional: bool,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Metrics]:
    key, new_actor, new_critic, new_metrics = update_actor(
        key,
        actor,
        critic,
        batch,
        actor_bc_coef,
        actor_bc_aux_weight,
        actor_bc_aux_loss,
        tau,
        actor_ema_tau,
        normalize_q,
        actor_input_noise,
        actor_bc_noise,
        actor_grad_noise,
        use_prev_state,
        use_prev_action,
        use_nf,
        use_distributional,
        metrics,
    )
    return key, new_actor, new_critic, new_metrics


@pyrallis.wrap()
def train(config: Config):
    config.project = "ActoReg"
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    if config.use_iql and config.num_refinement_epochs > 0:
        raise ValueError("IQL mode does not support refinement epochs")
    if config.actor_bc_aux_loss not in {"mse", "mae", "sum"}:
        raise ValueError("actor_bc_aux_loss must be 'mse', 'mae', or 'sum'")
    if config.critic_next_state_pred_epochs < 0:
        raise ValueError("critic_next_state_pred_epochs must be >= 0")
    if config.activation not in {"silu", "gsp"}:
        raise ValueError("activation must be 'silu' or 'gsp'")
    if config.use_likelihood_alpha_target and not config.use_nf:
        raise ValueError("use_likelihood_alpha_target requires use_nf=True")
    if config.likelihood_alpha_eps <= 0:
        raise ValueError("likelihood_alpha_eps must be > 0")
    if config.likelihood_stats_batch_size <= 0:
        raise ValueError("likelihood_stats_batch_size must be > 0")
    eval_num_samples_values = parse_eval_num_samples(config.nf_eval_num_samples)
    eval_q_infer_steps_values = parse_q_infer_steps(config.q_infer_steps)

    wandb.init(config=dict_config, project=config.project, group=config.group, name=config.name, id=str(uuid.uuid4()))
    wandb.mark_preempting()

    use_singletask_rollout = _is_singletask_ogbench(config.dataset_name)
    effective_append_goal = config.ogbench_append_goal and (not use_singletask_rollout)
    eval_task_ids = () if use_singletask_rollout else _parse_task_ids(config.ogbench_eval_task_ids)

    buffer = ReplayBuffer()
    buffer.create_from_ogbench(
        config.dataset_name,
        dataset_dir=config.ogbench_dataset_dir,
        normalize_reward=config.normalize_reward,
        is_normalize=config.normalize_states,
        discount=config.gamma,
        goal_source=config.ogbench_goal_source,
        append_goal=effective_append_goal,
        use_masks_for_dones=config.ogbench_use_masks_for_dones,
    )

    random.seed(config.train_seed)
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key, dropout_key = jax.random.split(key, 4)

    eval_env = make_env(config.dataset_name, seed=config.eval_seed, dataset_dir=config.ogbench_dataset_dir)

    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    init_prev_state = buffer.data["prev_states"][0][None, ...]
    init_prev_action = buffer.data["prev_actions"][0][None, ...]
    init_actor_state = build_actor_inputs(
        init_state,
        init_prev_state,
        init_prev_action,
        config.use_prev_state,
        config.use_prev_action,
    )

    if config.use_nf:
        actor_module = NFActorFlat(
            action_dim=init_action.shape[-1],
            hidden_dim=config.nf_hidden_dim,
            n_hiddens=config.nf_n_hiddens,
            num_layers=config.nf_num_layers,
            scale_max=config.nf_scale_max,
            base_dist=config.nf_base_dist,
            use_plu=config.nf_use_plu,
            use_layernorm=config.nf_use_layernorm,
            dropout_rate=config.nf_dropout,
            deterministic_layers=config.nf_det_layers,
            activation=config.activation,
        )
        reset_module = NFActorFlat(
            action_dim=init_action.shape[-1],
            hidden_dim=config.nf_hidden_dim,
            n_hiddens=config.nf_n_hiddens,
            num_layers=config.nf_num_layers,
            scale_max=config.nf_scale_max,
            base_dist=config.nf_base_dist,
            use_plu=config.nf_use_plu,
            use_layernorm=config.nf_use_layernorm,
            dropout_rate=config.nf_dropout,
            deterministic_layers=config.nf_det_layers,
            activation=config.activation,
        )
    else:
        if config.actor_prereset_mode:
            actor_module = DetActor(
                action_dim=init_action.shape[-1],
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                featurenorm=config.actor_fn,
                groupnorm=config.actor_gn,
                batchnorm=config.actor_bn,
                spectralnorm=config.actor_sn,
                dropout_rate=config.actor_dropout,
                n_hiddens=config.actor_n_hiddens,
                activation=config.activation,
            )
        else:
            actor_module = DetActor(
                action_dim=init_action.shape[-1],
                hidden_dim=config.hidden_dim,
                layernorm=False,
                featurenorm=False,
                groupnorm=False,
                spectralnorm=False,
                dropout_rate=0.0,
                n_hiddens=config.actor_n_hiddens,
                activation=config.activation,
            )
        reset_module = DetActor(
            action_dim=init_action.shape[-1],
            hidden_dim=config.hidden_dim,
            layernorm=config.actor_ln,
            featurenorm=config.actor_fn,
            groupnorm=config.actor_gn,
            dropout_rate=config.actor_dropout,
            n_hiddens=config.actor_n_hiddens,
            activation=config.activation,
        )

    if config.decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(config.actor_learning_rate, config.num_epochs * config.num_updates_on_epoch)
        actor_lr = schedule_fn
    elif config.decay_schedule == "linear":
        schedule_fn = optax.linear_schedule(
            config.actor_learning_rate,
            config.actor_learning_rate / 10,
            config.num_epochs * config.num_updates_on_epoch,
        )
        actor_lr = schedule_fn
    elif config.decay_schedule == "exp":
        schedule_fn = optax.exponential_decay(
            config.actor_learning_rate,
            config.num_epochs * config.num_updates_on_epoch,
            0.99,
        )
        actor_lr = schedule_fn
    else:
        actor_lr = config.actor_learning_rate

    if config.optimizer_type == "adan":
        optimizer = optax.adan(learning_rate=actor_lr, weight_decay=config.actor_wd)
    elif config.optimizer_type == "adam":
        optimizer = adamw_elastic(learning_rate=actor_lr, weight_decay=config.actor_wd, l1_ratio=config.l1_ratio)
    else:
        raise ValueError("optimizer_type must be 'adam' or 'adan'")

    if config.use_nf:
        init_vars = actor_module.init(
            {"params": actor_key, "mask": actor_key},
            init_actor_state,
            rng=actor_key,
            train=False,
            method=NFActorFlat.sample,
        )
    else:
        init_vars = actor_module.init(actor_key, init_actor_state, False)

    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=init_vars["params"],
        batch_stats=init_vars["batch_stats"] if "batch_stats" in init_vars else {},
        target_params=init_vars["params"],
        ema_params=init_vars["params"],
        target_batch_stats=init_vars["batch_stats"] if "batch_stats" in init_vars else {},
        ema_batch_stats=init_vars["batch_stats"] if "batch_stats" in init_vars else {},
        constants=init_vars.get("constants", {}),
        target_constants=init_vars.get("constants", {}),
        ema_constants=init_vars.get("constants", {}),
        dropout_key=dropout_key,
        tx=optimizer,
    )

    n_classes_eff = config.n_classes if config.use_distributional else 1
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=config.num_critics,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
        n_classes=n_classes_eff,
        use_distributional=config.use_distributional,
        dropout_rate=config.critic_dropout,
        activation=config.activation,
    )

    v_min, v_max = config.v_min, config.v_max
    if v_min == float("inf"):
        v_min = buffer.min
    if v_max == float("inf"):
        v_max = buffer.max

    expand = (v_max - v_min) * config.v_expand
    if config.v_expand_mode == "both":
        v_min -= expand / 2
        v_max += expand / 2
    elif config.v_expand_mode == "min":
        v_min -= expand
    elif config.v_expand_mode == "max":
        v_max += expand
    else:
        raise ValueError("Invalid expansion")

    if config.optimizer_type == "adan":
        critic_tx = optax.adan(learning_rate=config.critic_learning_rate, weight_decay=config.critic_wd)
    else:
        critic_tx = optax.adam(learning_rate=config.critic_learning_rate)

    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        support=jnp.linspace(v_min, v_max, n_classes_eff + 1, dtype=jnp.float32),
        sigma=config.sigma_frac * (v_max - v_min) / n_classes_eff,
        tx=critic_tx,
    )

    value = None
    if config.use_iql:
        if config.optimizer_type == "adan":
            value_tx = optax.adan(learning_rate=config.value_learning_rate, weight_decay=config.value_wd)
        else:
            value_tx = optax.adam(learning_rate=config.value_learning_rate)
        value_module = Value(hidden_dim=config.hidden_dim, layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens)
        value = ValueTrainState.create(
            apply_fn=value_module.apply,
            params=value_module.init(critic_key, init_state),
            tx=value_tx,
        )

    reset_mods = 1 if config.actor_prereset_mode else 0

    update_td3_partial = partial(
        update_td3,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        actor_bc_aux_weight=config.actor_bc_aux_weight,
        actor_bc_aux_loss=config.actor_bc_aux_loss,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        actor_ema_tau=config.actor_ema_tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        critic_objective_noise=config.critic_objective_noise,
        critic_grad_noise=config.critic_grad_noise,
        use_prev_state=config.use_prev_state,
        use_prev_action=config.use_prev_action,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise * reset_mods,
        actor_bc_noise=config.actor_bc_noise * reset_mods,
        actor_grad_noise=config.actor_grad_noise * reset_mods,
        use_target_actor=config.use_target_actor,
        use_nf=config.use_nf,
        use_distributional=config.use_distributional,
        next_state_pred_epochs=config.critic_next_state_pred_epochs,
        use_likelihood_alpha_target=config.use_likelihood_alpha_target,
        likelihood_alpha_eps=config.likelihood_alpha_eps,
    )

    update_td3_no_targets_partial = partial(
        update_td3_no_targets,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        critic_objective_noise=config.critic_objective_noise,
        critic_grad_noise=config.critic_grad_noise,
        use_prev_state=config.use_prev_state,
        use_prev_action=config.use_prev_action,
        use_target_actor=config.use_target_actor,
        use_nf=config.use_nf,
        use_distributional=config.use_distributional,
        next_state_pred_epochs=config.critic_next_state_pred_epochs,
        use_likelihood_alpha_target=config.use_likelihood_alpha_target,
        likelihood_alpha_eps=config.likelihood_alpha_eps,
    )

    update_iql_partial = partial(
        update_iql,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        actor_bc_aux_weight=config.actor_bc_aux_weight,
        actor_bc_aux_loss=config.actor_bc_aux_loss,
        tau=config.tau,
        actor_ema_tau=config.actor_ema_tau,
        iql_expectile=config.iql_expectile,
        critic_objective_noise=config.critic_objective_noise,
        critic_grad_noise=config.critic_grad_noise,
        use_prev_state=config.use_prev_state,
        use_prev_action=config.use_prev_action,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise * reset_mods,
        actor_bc_noise=config.actor_bc_noise * reset_mods,
        actor_grad_noise=config.actor_grad_noise * reset_mods,
        use_nf=config.use_nf,
        use_distributional=config.use_distributional,
        next_state_pred_epochs=config.critic_next_state_pred_epochs,
    )

    update_iql_no_actor_partial = partial(
        update_iql_no_actor,
        gamma=config.gamma,
        iql_expectile=config.iql_expectile,
        critic_objective_noise=config.critic_objective_noise,
        critic_grad_noise=config.critic_grad_noise,
        use_distributional=config.use_distributional,
        next_state_pred_epochs=config.critic_next_state_pred_epochs,
    )

    update_refinement_partial = partial(
        update_refinement,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef / config.refinement_div,
        actor_bc_aux_weight=config.actor_bc_aux_weight,
        actor_bc_aux_loss=config.actor_bc_aux_loss,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        actor_ema_tau=config.actor_ema_tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise,
        actor_bc_noise=config.actor_bc_noise,
        actor_grad_noise=config.actor_grad_noise,
        use_prev_state=config.use_prev_state,
        use_prev_action=config.use_prev_action,
        use_nf=config.use_nf,
        use_distributional=config.use_distributional,
    )

    update_actor_bc_partial = partial(
        update_actor_bc,
        beta=config.actor_bc_coef,
        aux_weight=config.actor_bc_aux_weight,
        aux_loss=config.actor_bc_aux_loss,
        tau=config.tau,
        ema_tau=config.actor_ema_tau,
        input_noise=config.actor_input_noise * reset_mods,
        bc_noise=config.actor_bc_noise * reset_mods,
        grad_noise=config.actor_grad_noise * reset_mods,
        use_prev_state=config.use_prev_state,
        use_prev_action=config.use_prev_action,
        use_nf=config.use_nf,
    )

    update_critic_warmup_partial = partial(
        update_critic_warmup,
        gamma=config.gamma,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        critic_objective_noise=config.critic_objective_noise,
        critic_grad_noise=config.critic_grad_noise,
        use_prev_state=config.use_prev_state,
        use_prev_action=config.use_prev_action,
        use_target_actor=config.use_target_actor,
        use_nf=config.use_nf,
        use_distributional=config.use_distributional,
        next_state_pred_epochs=config.critic_next_state_pred_epochs,
        use_likelihood_alpha_target=config.use_likelihood_alpha_target,
        likelihood_alpha_eps=config.likelihood_alpha_eps,
    )

    full_metrics_to_log = [
        "critic_loss",
        "critic_next_state_loss",
        "q_min",
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "actor_loss",
        "actor_loss_bc_term",
        "actor_loss_aux_term",
        "actor_loss_q_term",
        "actor_loss_lmbda",
        "aux_bc",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
    ]
    if config.use_iql:
        full_metrics_to_log.append("value_loss")

    actor_metrics_to_log = [
        "actor_loss",
        "actor_loss_bc_term",
        "actor_loss_aux_term",
        "aux_bc",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
        "nll",
    ]
    full_metrics_to_log.append("nll")

    critic_metrics_to_log = ["critic_loss", "critic_next_state_loss", "q_min", "alpha_mean", "alpha_min", "alpha_max"]
    if config.use_iql:
        critic_metrics_to_log.append("value_loss")

    delayed_updates = jnp.equal(jnp.arange(config.num_updates_on_epoch) % config.policy_freq, 0)

    def run_td3_updates(carry, buffer_data):
        buffer_size = buffer_data["states"].shape[0]
        key, indices_key = jax.random.split(carry["key"])
        batch_indices = jax.random.randint(
            indices_key,
            shape=(config.num_updates_on_epoch, config.batch_size),
            minval=0,
            maxval=buffer_size,
        )

        def body(carry, inputs):
            indices, do_update = inputs
            batch = jax.tree_util.tree_map(lambda arr: arr[indices], buffer_data)

            if config.use_iql:
                full_update = partial(
                    update_iql_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    value=carry["value"],
                    batch=batch,
                    metrics=carry["metrics"],
                    epoch=carry["epoch"],
                )
                update = partial(
                    update_iql_no_actor_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    value=carry["value"],
                    batch=batch,
                    metrics=carry["metrics"],
                    epoch=carry["epoch"],
                )
                key, new_actor, new_critic, new_value, new_metrics = jax.lax.cond(do_update, full_update, update)
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "value": new_value,
                    "metrics": new_metrics,
                    "epoch": carry["epoch"],
                    "likelihood_logprob_min": carry["likelihood_logprob_min"],
                    "likelihood_logprob_max": carry["likelihood_logprob_max"],
                }
            else:
                full_update = partial(
                    update_td3_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    batch=batch,
                    metrics=carry["metrics"],
                    epoch=carry["epoch"],
                    likelihood_logprob_min=carry["likelihood_logprob_min"],
                    likelihood_logprob_max=carry["likelihood_logprob_max"],
                )
                update = partial(
                    update_td3_no_targets_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    batch=batch,
                    metrics=carry["metrics"],
                    epoch=carry["epoch"],
                    likelihood_logprob_min=carry["likelihood_logprob_min"],
                    likelihood_logprob_max=carry["likelihood_logprob_max"],
                )
                key, new_actor, new_critic, new_metrics = jax.lax.cond(do_update, full_update, update)
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "metrics": new_metrics,
                    "epoch": carry["epoch"],
                    "likelihood_logprob_min": carry["likelihood_logprob_min"],
                    "likelihood_logprob_max": carry["likelihood_logprob_max"],
                }
            return new_carry, None

        inner_carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
            "epoch": carry["epoch"],
            "likelihood_logprob_min": carry["likelihood_logprob_min"],
            "likelihood_logprob_max": carry["likelihood_logprob_max"],
        }
        if config.use_iql:
            inner_carry["value"] = carry["value"]
        inner_carry, _ = jax.lax.scan(body, inner_carry, (batch_indices, delayed_updates))
        return inner_carry

    def run_actor_bc_updates(carry, buffer_data):
        buffer_size = buffer_data["states"].shape[0]
        key, indices_key = jax.random.split(carry["key"])
        value_state = carry.get("value")
        batch_indices = jax.random.randint(
            indices_key,
            shape=(config.num_updates_on_epoch, config.batch_size),
            minval=0,
            maxval=buffer_size,
        )

        def body(carry, indices):
            batch = jax.tree_util.tree_map(lambda arr: arr[indices], buffer_data)
            key, new_actor, new_metrics = update_actor_bc_partial(
                key=carry["key"],
                actor=carry["actor"],
                batch=batch,
                metrics=carry["metrics"],
            )
            new_carry = {
                "key": key,
                "actor": new_actor,
                "critic": carry["critic"],
                "metrics": new_metrics,
                "epoch": carry["epoch"],
                "likelihood_logprob_min": carry["likelihood_logprob_min"],
                "likelihood_logprob_max": carry["likelihood_logprob_max"],
            }
            if config.use_iql:
                new_carry["value"] = carry["value"]
            return new_carry, None

        carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
            "epoch": carry["epoch"],
            "likelihood_logprob_min": carry["likelihood_logprob_min"],
            "likelihood_logprob_max": carry["likelihood_logprob_max"],
        }
        if config.use_iql:
            carry["value"] = value_state
        carry, _ = jax.lax.scan(body, carry, batch_indices)
        return carry

    def run_critic_updates(carry, buffer_data):
        buffer_size = buffer_data["states"].shape[0]
        key, indices_key = jax.random.split(carry["key"])
        value_state = carry.get("value")
        batch_indices = jax.random.randint(
            indices_key,
            shape=(config.num_updates_on_epoch, config.batch_size),
            minval=0,
            maxval=buffer_size,
        )

        def body(carry, indices):
            batch = jax.tree_util.tree_map(lambda arr: arr[indices], buffer_data)
            if config.use_iql:
                key, new_actor, new_critic, new_value, new_metrics = update_iql_no_actor_partial(
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    value=carry["value"],
                    batch=batch,
                    metrics=carry["metrics"],
                    epoch=carry["epoch"],
                )
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "value": new_value,
                    "metrics": new_metrics,
                    "epoch": carry["epoch"],
                    "likelihood_logprob_min": carry["likelihood_logprob_min"],
                    "likelihood_logprob_max": carry["likelihood_logprob_max"],
                }
            else:
                key, new_actor, new_critic, new_metrics = update_critic_warmup_partial(
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    batch=batch,
                    metrics=carry["metrics"],
                    epoch=carry["epoch"],
                    likelihood_logprob_min=carry["likelihood_logprob_min"],
                    likelihood_logprob_max=carry["likelihood_logprob_max"],
                )
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "metrics": new_metrics,
                    "epoch": carry["epoch"],
                    "likelihood_logprob_min": carry["likelihood_logprob_min"],
                    "likelihood_logprob_max": carry["likelihood_logprob_max"],
                }
            return new_carry, None

        carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
            "epoch": carry["epoch"],
            "likelihood_logprob_min": carry["likelihood_logprob_min"],
            "likelihood_logprob_max": carry["likelihood_logprob_max"],
        }
        if config.use_iql:
            carry["value"] = value_state
        carry, _ = jax.lax.scan(body, carry, batch_indices)
        return carry

    def run_refinement_updates(carry, buffer_data):
        buffer_size = buffer_data["states"].shape[0]
        key, indices_key = jax.random.split(carry["key"])
        batch_indices = jax.random.randint(
            indices_key,
            shape=(config.num_updates_on_epoch, config.batch_size),
            minval=0,
            maxval=buffer_size,
        )

        def body(carry, indices):
            batch = jax.tree_util.tree_map(lambda arr: arr[indices], buffer_data)
            full_update = partial(
                update_refinement_partial,
                key=carry["key"],
                actor=carry["actor"],
                critic=carry["critic"],
                batch=batch,
                metrics=carry["metrics"],
            )
            key, new_actor, new_critic, new_metrics = full_update()
            new_carry = {
                "key": key,
                "actor": new_actor,
                "critic": new_critic,
                "metrics": new_metrics,
                "epoch": carry["epoch"],
                "likelihood_logprob_min": carry["likelihood_logprob_min"],
                "likelihood_logprob_max": carry["likelihood_logprob_max"],
            }
            return new_carry, None

        carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
            "epoch": carry["epoch"],
            "likelihood_logprob_min": carry["likelihood_logprob_min"],
            "likelihood_logprob_max": carry["likelihood_logprob_max"],
        }
        carry, _ = jax.lax.scan(body, carry, batch_indices)
        return carry

    run_td3_updates = jax.jit(run_td3_updates)
    run_actor_bc_updates = jax.jit(run_actor_bc_updates)
    run_critic_updates = jax.jit(run_critic_updates)
    run_refinement_updates = jax.jit(run_refinement_updates)

    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "epoch": jnp.array(0, dtype=jnp.int32),
        "likelihood_logprob_min": jnp.array(0.0, dtype=jnp.float32),
        "likelihood_logprob_max": jnp.array(1.0, dtype=jnp.float32),
    }
    if config.use_iql:
        update_carry["value"] = value

    @partial(jax.jit, static_argnums=(5, 6, 7))
    def actor_action_fn(
        params: jax.Array,
        batch_stats: jax.Array,
        constants: Any,
        obs: jax.Array,
        rng: jax.Array,
        num_samples: int,
        z_scale: float,
        z_clip: float,
    ):
        if config.use_nf:
            if num_samples > 1:
                return actor.apply_fn(
                    {"params": params, "constants": constants},
                    obs,
                    rng=rng,
                    num_samples=num_samples,
                    z_scale=z_scale,
                    z_clip=z_clip,
                    train=False,
                    method=NFActorFlat.sample_n,
                )
            return actor.apply_fn(
                {"params": params, "constants": constants},
                obs,
                rng=rng,
                train=False,
                method=NFActorFlat.sample,
            )
        return actor.apply_fn({"params": params, "batch_stats": batch_stats}, obs, False)[0]

    @jax.jit
    def actor_logprob_fn(
        params: jax.Array,
        batch_stats: jax.Array,
        constants: Any,
        obs: jax.Array,
        actions: jax.Array,
    ):
        del batch_stats
        return actor.apply_fn(
            {"params": params, "constants": constants},
            actions,
            obs,
            train=False,
            method=NFActorFlat.log_prob,
        )

    il_end = config.il_warmup_epochs
    critic_end = il_end + config.critic_warmup_epochs

    for epoch in trange(config.num_epochs, desc="ReBRAC+ Epochs"):
        stage = "rl"
        if epoch < il_end:
            stage = "il"
        elif epoch < critic_end:
            stage = "critic"

        if config.use_likelihood_alpha_target and config.use_nf:
            needs_stats = (
                (il_end > 0 and epoch == il_end)
                or (il_end == 0 and epoch == 0)
            )
            if needs_stats:
                lp_min, lp_max = compute_dataset_action_logprob_stats(
                    update_carry["actor"],
                    buffer.data,
                    config.use_prev_state,
                    config.use_prev_action,
                    config.likelihood_stats_batch_size,
                )
                update_carry["likelihood_logprob_min"] = jnp.array(lp_min, dtype=jnp.float32)
                update_carry["likelihood_logprob_max"] = jnp.array(lp_max, dtype=jnp.float32)
                wandb.log(
                    {
                        "epoch": epoch,
                        "ReBRACPlus/likelihood_logprob_min": lp_min,
                        "ReBRACPlus/likelihood_logprob_max": lp_max,
                    }
                )

        update_fn = run_td3_updates
        metrics_list = full_metrics_to_log
        if stage == "il":
            update_fn = run_actor_bc_updates
            metrics_list = actor_metrics_to_log
        elif stage == "critic":
            update_fn = run_critic_updates
            metrics_list = critic_metrics_to_log

        if epoch == config.num_epochs - config.num_refinement_epochs:
            print("Refinement stage")
            update_fn = run_refinement_updates
            metrics_list = full_metrics_to_log
            if config.actor_reset:
                if config.use_nf:
                    reset_vars = reset_module.init(
                        {"params": actor_key, "mask": actor_key},
                        init_actor_state,
                        rng=actor_key,
                        train=False,
                        method=NFActorFlat.sample,
                    )
                else:
                    reset_vars = reset_module.init(actor_key, init_actor_state, False)
                actor = ActorTrainState.create(
                    apply_fn=reset_module.apply,
                    params=reset_vars["params"],
                    batch_stats=reset_vars["batch_stats"] if "batch_stats" in reset_vars else {},
                    target_params=reset_vars["params"],
                    ema_params=reset_vars["params"],
                    target_batch_stats=reset_vars["batch_stats"] if "batch_stats" in reset_vars else {},
                    ema_batch_stats=reset_vars["batch_stats"] if "batch_stats" in reset_vars else {},
                    constants=reset_vars.get("constants", {}),
                    target_constants=reset_vars.get("constants", {}),
                    ema_constants=reset_vars.get("constants", {}),
                    dropout_key=dropout_key,
                    tx=optimizer,
                )
                update_carry.update(actor=actor)

        update_carry["epoch"] = jnp.array(epoch, dtype=jnp.int32)
        update_carry["metrics"] = Metrics.create(metrics_list)
        update_carry = update_fn(update_carry, buffer.data)
        mean_metrics = update_carry["metrics"].compute()

        wandb.log({"epoch": epoch, **{f"ReBRACPlus/{k}": v for k, v in mean_metrics.items()}})

        force_eval = epoch == il_end - 1 or epoch == critic_end - 1
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1 or force_eval:
            eval_actor_params = update_carry["actor"].ema_params if config.use_actor_ema else update_carry["actor"].params
            eval_actor_batch_stats = (
                update_carry["actor"].ema_batch_stats if config.use_actor_ema else update_carry["actor"].batch_stats
            )
            eval_actor_constants = (
                update_carry["actor"].ema_constants if config.use_actor_ema else update_carry["actor"].constants
            )
            eval_select = "likelihood" if stage == "il" else "q"
            eval_q_step_size = 0.0 if stage == "il" else config.q_infer_step_size
            eval_q_steps_values = eval_q_infer_steps_values
            eval_metrics = {"epoch": epoch}
            multi_eval = len(eval_num_samples_values) > 1
            multi_qs = len(eval_q_steps_values) > 1
            for ns in eval_num_samples_values:
                ns_suffix = f"_ns_{ns}" if multi_eval else ""
                for qs in eval_q_steps_values:
                    qs_suffix = f"_qs_{qs}" if multi_qs else ""
                    suffix = f"{ns_suffix}{qs_suffix}"
                    eval_returns, _ = evaluate(
                        eval_env,
                        eval_actor_params,
                        eval_actor_batch_stats,
                        eval_actor_constants,
                        update_carry["critic"],
                        actor_action_fn,
                        actor_logprob_fn if config.use_nf else None,
                        config.eval_episodes,
                        seed=config.eval_seed,
                        state_mean=buffer.mean,
                        state_std=buffer.std,
                        q_infer_step_size=eval_q_step_size,
                        q_infer_steps=qs,
                        eval_task_ids=eval_task_ids,
                        append_goal=effective_append_goal,
                        goal_source=config.ogbench_goal_source,
                        use_prev_state=config.use_prev_state,
                        use_prev_action=config.use_prev_action,
                        use_nf=config.use_nf,
                        use_distributional=config.use_distributional,
                        nf_eval_num_samples=ns,
                        nf_eval_z_scale=config.nf_eval_z_scale,
                        nf_eval_z_clip=config.nf_eval_z_clip,
                        nf_eval_select=eval_select,
                    )
                    normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                    eval_metrics[f"eval/return_mean{suffix}"] = np.mean(eval_returns)
                    eval_metrics[f"eval/return_std{suffix}"] = np.std(eval_returns)
                    eval_metrics[f"eval/normalized_score_mean{suffix}"] = np.mean(normalized_score)
                    eval_metrics[f"eval/normalized_score_std{suffix}"] = np.std(normalized_score)

                    if config.noisy_eval:
                        for sn, an in [(0.0, 0.2), (0.05, 0.0)]:
                            returns, _ = evaluate(
                                eval_env,
                                eval_actor_params,
                                eval_actor_batch_stats,
                                eval_actor_constants,
                                update_carry["critic"],
                                actor_action_fn,
                                actor_logprob_fn if config.use_nf else None,
                                config.eval_episodes,
                                seed=config.eval_seed,
                                state_mean=buffer.mean,
                                state_std=buffer.std,
                                action_noise=an,
                                state_noise=sn,
                                q_infer_step_size=eval_q_step_size,
                                q_infer_steps=qs,
                                eval_task_ids=eval_task_ids,
                                append_goal=effective_append_goal,
                                goal_source=config.ogbench_goal_source,
                                use_prev_state=config.use_prev_state,
                                use_prev_action=config.use_prev_action,
                                use_nf=config.use_nf,
                                use_distributional=config.use_distributional,
                                nf_eval_num_samples=ns,
                                nf_eval_z_scale=config.nf_eval_z_scale,
                                nf_eval_z_clip=config.nf_eval_z_clip,
                                nf_eval_select=eval_select,
                            )
                            normalized_returns = eval_env.get_normalized_score(returns) * 100.0
                            eval_metrics[f"eval/normalized_score_mean{suffix}_sn_{sn}_an_{an}"] = np.mean(
                                normalized_returns
                            )

            wandb.log(eval_metrics)

    wandb.finish()
    raise SystemExit(0)


if __name__ == "__main__":
    train()

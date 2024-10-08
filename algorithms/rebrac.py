# source: https://github.com/tinkoff-ai/ReBRAC
# https://arxiv.org/abs/2305.09836

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from flax.training import checkpoints
from tqdm.auto import trange

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class Config:
    # wandb params
    project: str = "ActReg"
    group: str = "rebrac"
    name: str = "rebrac"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    actor_fn: bool = False
    actor_gn: bool = False
    critic_ln: bool = True
    actor_dropout: float = 0.0
    actor_wd: float = 0.0
    actor_input_noise: float = 0.0
    actor_bc_noise: float = 0.0
    actor_grad_noise: float = 0.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    decay_schedule: str = None
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10
    # general params
    train_seed: int = 0
    eval_seed: int = 42

    mlc_job_name: str = None
    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


def pytorch_init(fan_in: float) -> Callable:
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def identity(x: Any) -> Any:
    return x


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = False
    groupnorm: bool = False
    featurenorm: bool = False
    dropout_rate: float = 0.0
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
            nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
            nn.GroupNorm() if self.groupnorm else identity,
            nn.Dropout(rate=self.dropout_rate, deterministic=not train),
        ]
        for _ in range(self.n_hiddens - 2):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
                nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
                nn.GroupNorm() if self.groupnorm else identity,
                nn.Dropout(rate=self.dropout_rate, deterministic=not train),
            ]
        layers += [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(h_d),
                bias_init=nn.initializers.constant(0.1),
            ),
        ]

        net = nn.Sequential(layers)
        trunk = net(state)

        last_layer = nn.Sequential(
            [
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
                nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
                nn.GroupNorm() if self.groupnorm else identity,
                nn.Dropout(rate=self.dropout_rate, deterministic=not train),
                nn.Dense(
                    self.action_dim,
                    kernel_init=uniform_init(1e-3),
                    bias_init=uniform_init(1e-3),
                ),
                nn.tanh
            ]
        )
        actions = last_layer(trunk)

        return actions, trunk


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(
            state, action
        )
        return q_values


def qlearning_dataset(
    env: gym.Env,
    dataset: Dict = None,
    terminate_on_end: bool = False,
    validation_frac: float = 0.05,
    **kwargs,
) -> Tuple[Dict, Dict]:
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    episode_ends = []

    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        new_action = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition
            episode_step = 0
            episode_ends.append(i)
            continue
        if done_bool or final_timestep:
            episode_step = 0
            episode_ends.append(i)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    cnt_episodes = len(episode_ends)
    cnt_train_episodes = int(cnt_episodes * (1 - validation_frac))

    print("TOTAL NUMBER OF EPISODES:", cnt_episodes)
    print("TRAIN EPISODES:", cnt_train_episodes)
    print("VAL EPISODES:", cnt_episodes - cnt_train_episodes)

    last_train_ep_idx = episode_ends[cnt_train_episodes]
    print("Last train idx:", last_train_ep_idx, "out of", len(obs_))
    train_data = {
        "observations": np.array(obs_)[:last_train_ep_idx],
        "actions": np.array(action_)[:last_train_ep_idx],
        "next_observations": np.array(next_obs_)[:last_train_ep_idx],
        "next_actions": np.array(next_action_)[:last_train_ep_idx],
        "rewards": np.array(reward_)[:last_train_ep_idx],
        "terminals": np.array(done_)[:last_train_ep_idx],
    }

    val_data = {
        "observations": np.array(obs_)[last_train_ep_idx:],
        "actions": np.array(action_)[last_train_ep_idx:],
        "next_observations": np.array(next_obs_)[last_train_ep_idx:],
        "next_actions": np.array(next_action_)[last_train_ep_idx:],
        "rewards": np.array(reward_)[last_train_ep_idx:],
        "terminals": np.array(done_)[last_train_ep_idx:],
    }
    return train_data, val_data


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    val_data: Dict[str, jax.Array] = None
    random_data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def create_from_d4rl(
        self,
        dataset_name: str,
        normalize_reward: bool = False,
        is_normalize: bool = False,
    ):
        d4rl_data, val_data = qlearning_dataset(gym.make(dataset_name))

        state_min = np.min(d4rl_data["observations"], axis=0)
        state_max = np.max(d4rl_data["observations"], axis=0)

        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
        }
        val_buffer = {
            "states": jnp.asarray(val_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(val_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(val_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                val_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(val_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(val_data["terminals"], dtype=jnp.float32),
        }

        random_buffer = {
            "states": jnp.asarray(np.random.uniform(0, 1, (val_data["observations"].shape[0], state_max.shape[0])) * (state_max - state_min) + state_min, dtype=jnp.float32),
            "actions": jnp.asarray(np.random.uniform(-1, 1, (val_data["observations"].shape[0], val_data["actions"].shape[1])), dtype=jnp.float32),
        }

        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
            val_buffer["states"] = normalize_states(val_buffer["states"], self.mean, self.std)
            val_buffer["next_states"] = normalize_states(
                val_buffer["next_states"], self.mean, self.std
            )
            random_buffer["states"] = normalize_states(random_buffer["states"], self.mean, self.std)
            random_buffer["next_states"] = normalize_states(
                random_buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, buffer["rewards"]
            )
            val_buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, val_buffer["rewards"]
            )
        self.data = buffer
        self.val_data = val_buffer
        self.random_data = random_buffer

    @property
    def size(self) -> int:
        # WARN: It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(
        self, key: jax.random.PRNGKey, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size
        )
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def sample_n_first(
        self, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jnp.arange(0, batch_size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def sample_random(
            self, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jnp.arange(0, batch_size)
        batch = jax.tree_map(lambda arr: arr[indices], self.random_data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError(
                "Reward normalization is implemented only for AntMaze yet!"
            )


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(
    arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward: float) -> float:
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def evaluate(
    env: gym.Env,
    params: jax.Array,
    expert_params: jax.Array,
    action_fn: Callable,
    expert_action_fn: Callable,
    num_episodes: int,
    seed: int,
) -> Tuple[np.ndarray, float, Dict]:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    expert_mses = []
    eval_states = []
    eval_actions = []

    for _ in trange(num_episodes, desc="Eval", leave=False):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            eval_states.append(obs)
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            eval_actions.append(action)
            expert_action = np.asarray(jax.device_get(expert_action_fn(expert_params, obs)))
            expert_mses.append(((action - expert_action) ** 2))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    eval_batch = {
        "states": jnp.array(eval_states),
        "actions": jnp.array(eval_actions),
    }
    return np.array(returns), np.mean(expert_mses), eval_batch


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict
    dropout_key: jax.Array


def compute_dead_neurons_statistic(logits: jax.Array):
    positive_count = jnp.sum(logits > 0, axis=0)
    dead_count = jnp.sum((positive_count == 0))
    return dead_count / logits.shape[1]


def compute_feature_norm(logits: jax.Array):
    norms = jnp.linalg.norm(logits, axis=1)
    means = jnp.mean(logits, axis=1)
    stds = jnp.std(logits, axis=1)
    return jnp.mean(norms), jnp.mean(means), jnp.mean(stds)


def compute_feature_rank_pca(logits: jax.Array):
    cutoff = 0.01
    threshold = 1 - cutoff

    svals = jnp.linalg.svd(logits, full_matrices=False, compute_uv=False)
    sval_squares = svals ** 2
    sval_squares_sum = jnp.sum(sval_squares)
    cumsum_squares = jnp.cumsum(sval_squares)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum)
    approximate_ranks = (~threshold_crossed).sum() + 1

    return approximate_ranks


def compute_feature_statistics(logits: jax.Array):
    dead_neurons_frac = compute_dead_neurons_statistic(logits)
    feature_norms, feature_means, feature_stds = compute_feature_norm(logits)
    pca_rank = compute_feature_rank_pca(logits)

    return {
        "dead_neurons_frac": dead_neurons_frac,
        "feature_norms": feature_norms,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "pca_rank": pca_rank,
    }


def update_actor(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: Dict[str, jax.Array],
    beta: float,
    tau: float,
    normalize_q: bool,
    input_noise: float,
    bc_noise: float,
    grad_noise: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, random_action_key, input_noise_key, bc_noise_key, grad_noise_key = jax.random.split(key, 5)
    dropout_key, new_dropout_key = jax.random.split(actor.dropout_key, 2)

    in_noise = jax.random.normal(input_noise_key, batch["states"].shape) * input_noise
    b_noise = jax.random.normal(bc_noise_key, batch["actions"].shape) * bc_noise

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        actions, preact = actor.apply_fn(params, batch["states"] + in_noise, True, rngs={'dropout': dropout_key})
        bc_penalty = ((actions - batch["actions"] + b_noise) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()

        # logging stuff
        random_actions = jax.random.uniform(
            random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
        )
        new_metrics = metrics.update(
            {
                "actor_loss": loss,
                "bc_mse_policy": bc_penalty.mean(),
                "bc_mse_random": ((random_actions - batch["actions"]) ** 2)
                .sum(-1)
                .mean(),
                "action_mse": ((actions - batch["actions"]) ** 2).mean(),
            }
        )
        return loss, new_metrics

    grads, new_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

    def add_gaussian_noise(gr, noise_std, rng_key):
        def add_noise_to_grad(g, rng_key):
            noise = jax.random.normal(rng_key, g.shape) * noise_std / ((1 + actor.step) ** 0.55)
            return g + noise

        leaves, tree = jax.tree_util.tree_flatten(gr)
        rng_keys = jax.random.split(rng_key, num=len(leaves))
        rng_keys = jax.tree_util.tree_unflatten(tree, rng_keys)

        noisy_grads = jax.tree_util.tree_map(lambda g, k: add_noise_to_grad(g, k), gr, rng_keys)
        return noisy_grads

    grads = add_gaussian_noise(grads, grad_noise, grad_noise_key)
    new_actor = actor.apply_gradients(grads=grads)

    new_actor = new_actor.replace(
        target_params=optax.incremental_update(actor.params, actor.target_params, tau),
        dropout_key=new_dropout_key,
    )
    new_critic = critic.replace(
        target_params=optax.incremental_update(critic.params, critic.target_params, tau)
    )

    actor_params = new_actor.params
    actor_params = jax.tree_util.tree_map(lambda x: x.reshape(-1), actor_params)
    flat_vals, _ = jax.tree.flatten(actor_params)
    flat_mean = jnp.mean(jnp.concatenate(flat_vals))

    new_metrics = new_metrics.update(
        {"weights/actor_weights_mean": flat_mean}
    )
    return key, new_actor, new_critic, new_metrics


def update_critic(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    batch: Dict[str, jax.Array],
    gamma: float,
    beta: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions, preact = actor.apply_fn(actor.target_params, batch["next_states"], False)
    noise = jax.numpy.clip(
        (jax.random.normal(actions_key, next_actions.shape) * policy_noise),
        -noise_clip,
        noise_clip,
    )
    next_actions = jax.numpy.clip(next_actions + noise, -1, 1)
    bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
    next_q = critic.apply_fn(
        critic.target_params, batch["next_states"], next_actions
    ).min(0)
    next_q = next_q - beta * bc_penalty

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        critic.params
    )
    new_critic = critic.apply_gradients(grads=grads)
    new_metrics = metrics.update(
        {
            "critic_loss": loss,
            "q_min": q_min,
        }
    )
    return key, new_critic, new_metrics


def update_td3(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    gamma: float,
    actor_bc_coef: float,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    normalize_q: bool,
    actor_input_noise: float,
    actor_bc_noise: float,
    actor_grad_noise: float,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
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
        metrics,
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key, actor, new_critic, batch, actor_bc_coef, tau, normalize_q, actor_input_noise, actor_bc_noise, actor_grad_noise, new_metrics
    )
    return key, new_actor, new_critic, new_metrics


def update_td3_no_targets(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    batch: Dict[str, Any],
    gamma: float,
    metrics: Metrics,
    actor_bc_coef: float,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
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
        metrics,
    )
    return key, actor, new_critic, new_metrics


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        action = actor.apply_fn(actor.params, obs, False)[0]
        return action

    return _action_fn


def eval_actor(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: Dict[str, jax.Array],
    beta: float,
    normalize_q: bool,
) -> Tuple[jax.random.PRNGKey, Dict]:
    key, random_action_key = jax.random.split(key, 2)
    metrics = {}

    def actor_loss_fn(params: jax.Array) -> jax.Array:
        actions, preact = actor.apply_fn(params, batch["states"], False)

        bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()

        metrics.update(
            {
                "actor_loss": loss,
                "action_mse": ((actions - batch["actions"]) ** 2).mean(),
            }
        )
        metrics.update(compute_feature_statistics(preact))

        return loss

    actor_loss_fn(actor.params)

    return key, metrics


@pyrallis.wrap()
def train(config: Config):
    config.project = "ActoReg"
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key, dropout_key = jax.random.split(key, 4)

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

    actor_module = DetActor(
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
        layernorm=config.actor_ln,
        featurenorm=config.actor_fn,
        groupnorm=config.actor_gn,
        dropout_rate=config.actor_dropout,
        n_hiddens=config.actor_n_hiddens,
    )
    expert_module = DetActor(
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
        layernorm=False,
        featurenorm=False,
        groupnorm=False,
        dropout_rate=0.0,
        n_hiddens=config.actor_n_hiddens,
    )

    if config.decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(config.actor_learning_rate, config.num_epochs * config.num_updates_on_epoch)
        optimizer = optax.adamw(learning_rate=schedule_fn, weight_decay=config.actor_wd)
    elif config.decay_schedule == "linear":
        schedule_fn = optax.linear_schedule(config.actor_learning_rate, config.actor_learning_rate / 10, config.num_epochs * config.num_updates_on_epoch)
        optimizer = optax.adamw(learning_rate=schedule_fn, weight_decay=config.actor_wd)
    elif config.decay_schedule == "exp":
        schedule_fn = optax.exponential_decay(config.actor_learning_rate, config.num_epochs * config.num_updates_on_epoch, 0.99)
        optimizer = optax.adamw(learning_rate=schedule_fn, weight_decay=config.actor_wd)
    else:
        optimizer = optax.adamw(learning_rate=config.actor_learning_rate, weight_decay=config.actor_wd)

    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state, False),
        target_params=actor_module.init(actor_key, init_state, False),
        dropout_key=dropout_key,
        tx=optimizer,
    )

    expert_actor = ActorTrainState.create(
        apply_fn=expert_module.apply,
        params=expert_module.init(actor_key, init_state, False),
        target_params=expert_module.init(actor_key, init_state, False),
        dropout_key=dropout_key,
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    save_name = config.dataset_name.split("-")[0]
    if "antmaze" in config.dataset_name:
        save_name += "-" + config.dataset_name.split("-")[1]
    print("Loading expert from:", f'expert_checkpoints/{save_name}')
    expert_actor = checkpoints.restore_checkpoint(ckpt_dir=f'expert_checkpoints/{save_name}', target=expert_actor)

    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=2,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    update_td3_partial = partial(
        update_td3,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise,
        actor_bc_noise=config.actor_bc_noise,
        actor_grad_noise=config.actor_grad_noise,
    )

    update_td3_no_targets_partial = partial(
        update_td3_no_targets,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
    )

    def td3_loop_update_step(i: int, carry: TrainState):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        full_update = partial(
            update_td3_partial,
            key=key,
            actor=carry["actor"],
            critic=carry["critic"],
            batch=batch,
            metrics=carry["metrics"],
        )

        update = partial(
            update_td3_no_targets_partial,
            key=key,
            actor=carry["actor"],
            critic=carry["critic"],
            batch=batch,
            metrics=carry["metrics"],
        )

        key, new_actor, new_critic, new_metrics = jax.lax.cond(
            update_carry["delayed_updates"][i], full_update, update
        )

        carry.update(key=key, actor=new_actor, critic=new_critic, metrics=new_metrics)
        return carry

    # metrics
    bc_metrics_to_log = [
        "critic_loss",
        "q_min",
        "actor_loss",
        "batch_entropy",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
        "weights/actor_weights_mean",
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "buffer": buffer,
        "delayed_updates": jax.numpy.equal(
            jax.numpy.arange(config.num_updates_on_epoch) % config.policy_freq, 0
        ).astype(int),
    }

    @jax.jit
    def actor_action_fn(params: jax.Array, obs: jax.Array):
        return actor.apply_fn(params, obs, False)[0]

    @jax.jit
    def expert_action_fn(params: jax.Array, obs: jax.Array):
        return expert_actor.apply_fn(params, obs, False)[0]

    for epoch in trange(config.num_epochs, desc="ReBRAC Epochs"):
        # metrics for accumulation during epoch and logging to wandb
        # we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)

        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=td3_loop_update_step,
            init_val=update_carry,
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log(
            {"epoch": epoch, **{f"ReBRAC/{k}": v for k, v in mean_metrics.items()}}
        )

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns, expert_mse, eval_batch = evaluate(
                eval_env,
                update_carry["actor"].params,
                expert_actor.params,
                actor_action_fn,
                expert_action_fn,
                config.eval_episodes,
                seed=config.eval_seed,
            )
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            eval_metrics = {
                    "epoch": epoch,
                    "eval/return_mean": np.mean(eval_returns),
                    "eval/return_std": np.std(eval_returns),
                    "eval/normalized_score_mean": np.mean(normalized_score),
                    "eval/normalized_score_std": np.std(normalized_score),
                    "eval_metrics/expert_mse": expert_mse,
                }

            new_key, val_metrics = eval_actor(update_carry["key"], update_carry["actor"], update_carry["critic"],
                                              buffer.val_data, config.actor_bc_coef, config.normalize_q)
            new_key, train_metrics = eval_actor(new_key, update_carry["actor"], update_carry["critic"],
                                                buffer.sample_n_first(buffer.val_data["states"].shape[0]),
                                                config.actor_bc_coef, config.normalize_q)
            new_key, eval_metrics_1 = eval_actor(new_key, update_carry["actor"], update_carry["critic"],
                                                eval_batch,
                                                config.actor_bc_coef, config.normalize_q)
            new_key, rand_metrics = eval_actor(new_key, update_carry["actor"], update_carry["critic"],
                                                buffer.sample_random(buffer.val_data["states"].shape[0]),
                                                config.actor_bc_coef, config.normalize_q)
            buffer.sample_random(buffer.val_data["states"].shape[0])
            del eval_metrics_1["action_mse"]

            for k in train_metrics:
                eval_metrics[f"train_metrics/{k}"] = train_metrics[k]
            for k in val_metrics:
                eval_metrics[f"validation_metrics/{k}"] = val_metrics[k]
            for k in eval_metrics_1:
                eval_metrics[f"eval_metrics/{k}"] = eval_metrics_1[k]
            for k in rand_metrics:
                eval_metrics[f"rand_metrics/{k}"] = rand_metrics[k]

            update_carry["key"] = new_key

            wandb.log(
                eval_metrics
            )

if __name__ == "__main__":
    train()

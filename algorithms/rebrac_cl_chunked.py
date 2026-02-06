# source: https://github.com/tinkoff-ai/ReBRAC
# https://arxiv.org/abs/2305.09836

import os

#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
import uuid
import random
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import optax
from optax._src import base, combine, transform
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from flax.training import checkpoints
from tqdm.auto import trange

from nf_policy import NFActor
default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class Config:
    # wandb params
    project: str = "ReBRAC2"
    group: str = "rebrac2"
    name: str = "rebrac-2"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 0.1
    critic_bc_coef: float = 0.0
    actor_ln: bool = False
    actor_fn: bool = False
    actor_gn: bool = False
    actor_bn: bool = False
    actor_sn: bool = False
    critic_ln: bool = True
    actor_dropout: float = 0.1
    actor_wd: float = 0.0
    l1_ratio: float = 0.0
    actor_input_noise: float = 0.0
    actor_bc_noise: float = 0.0
    actor_grad_noise: float = 0.01
    actor_reset: bool = False
    actor_prereset_mode: bool = True
    use_nf: bool = True
    nf_num_layers: int = 8
    nf_hidden_dim: int = 64
    nf_n_hiddens: int = 2
    nf_use_transformer: bool = False
    nf_scale_max: float = 1.0
    nf_base_dist: str = "normal"
    nf_use_plu: bool = False
    nf_use_layernorm: bool = True
    nf_dropout: float = 0.1
    nf_det_layers: int = 2
    nf_eval_num_samples: int = 8
    nf_eval_z_scale: float = 1.0
    nf_eval_z_clip: float = 0.0
    policy_noise: float = 0.0
    noise_clip: float = 0.0
    policy_freq: int = 2
    normalize_q: bool = True
    decay_schedule: str = None
    num_critics: int = 2
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 1000
    num_refinement_epochs: int = 0
    refinement_div: float = 1
    il_warmup_epochs: int = 0
    critic_warmup_epochs: int = 0
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    action_chunk_len: int = 6
    action_chunk_stride: int = 1
    rtc_prefix_len: Optional[int] = None
    q_infer_step_size: float = 0.1
    q_infer_steps: int = 1
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 100
    eval_num_envs: int = 1
    # general params
    train_seed: int = 0
    eval_seed: int = 42
    # classification

    n_classes: int = 101
    sigma_frac: float = 0.75
    v_min: float = float('inf')
    v_max: float = float('inf')
    v_expand: float = 0.05
    v_expand_mode: str = "both"
    # IQL params
    use_iql: bool = False
    value_learning_rate: float = 1e-3
    iql_expectile: float = 0.5

    noisy_eval: bool = False
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


AddDecayedWeightsState = base.EmptyState


def add_elastic_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    l1_ratio: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
  def init_fn(params):
    del params
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda g, p: g + weight_decay * ((1 - l1_ratio) * p + l1_ratio * jnp.sign(p)), updates, params)
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask)
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
    chunk_len: int = 1
    rtc_prefix_len: int = 0
    hidden_dim: int = 256
    layernorm: bool = False
    groupnorm: bool = False
    featurenorm: bool = False
    batchnorm: bool = False
    spectralnorm: bool = False
    dropout_rate: float = 0.0
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, prev_actions: jax.Array, train: bool) -> Tuple[jax.Array, jax.Array]:
        single = state.ndim == 1
        if single:
            state = state[None, :]
            if prev_actions.ndim == 2:
                prev_actions = prev_actions[None, ...]
        s_d, h_d = state.shape[-1], self.hidden_dim
        prev_flat = prev_actions.reshape(prev_actions.shape[0], -1)
        state = jnp.hstack([state, prev_flat])
        s_d = state.shape[-1]
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
            nn.BatchNorm(use_running_average=not train) if self.batchnorm else identity,
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
                nn.BatchNorm(use_running_average=not train) if self.batchnorm else identity,
                nn.Dropout(rate=self.dropout_rate, deterministic=not train),
            ]


        net = nn.Sequential(layers)

        trunk = nn.Dense(
            self.hidden_dim,
            kernel_init=pytorch_init(h_d),
            bias_init=nn.initializers.constant(0.1),
        )(net(state)) if not self.spectralnorm else nn.SpectralNorm(nn.Dense(
            self.hidden_dim,
            kernel_init=pytorch_init(h_d),
            bias_init=nn.initializers.constant(0.1),
        ))(net(state), update_stats=train)

        # trunk = net(state)

        last_layer = nn.Sequential(
            [
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
                nn.LayerNorm(use_bias=False, use_scale=False) if self.featurenorm else identity,
                nn.GroupNorm() if self.groupnorm else identity,
                nn.BatchNorm(use_running_average=not train) if self.batchnorm else identity,
                nn.Dropout(rate=self.dropout_rate, deterministic=not train),
                nn.Dense(
                    self.action_dim * self.chunk_len,
                    kernel_init=uniform_init(1e-3),
                    bias_init=uniform_init(1e-3),
                ),
                nn.tanh
            ]
        )
        actions = last_layer(trunk)
        actions = actions.reshape(actions.shape[0], self.chunk_len, self.action_dim)
        if single:
            actions = actions[0]
            trunk = trunk[0]

        return actions, trunk


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3
    n_classes: int = 21

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        if action.ndim > 2:
            action = action.reshape(action.shape[0], -1)
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
            # nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
            nn.Dense(self.n_classes, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action)  # .squeeze(-1)
        return out


class Value(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
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
        layers += [nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))]
        network = nn.Sequential(layers)
        out = network(state).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3
    n_classes: int = 21

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
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens, self.n_classes)(
            state, action
        )
        return q_values


def calc_return_to_go(is_sparse_reward, rewards, terminals, gamma):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return []
    reward_neg = 0
    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return return_to_go


def qlearning_dataset(
        env: gym.Env,
        dataset_name: Dict = None,
        normalize_reward=False,
        dataset=None,
        terminate_on_end: bool = False,
        discount=0.99,
        chunk_len: int = 5,
        chunk_stride: int = 1,
        rtc_prefix_len: Optional[int] = None,
        **kwargs,
) -> Tuple[Dict, float, float]:
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    if normalize_reward:
        dataset['rewards'] = ReplayBuffer.normalize_reward(dataset_name, dataset['rewards'])

    if chunk_len < 1:
        raise ValueError("chunk_len must be >= 1")
    if chunk_stride < 1:
        raise ValueError("chunk_stride must be >= 1")
    if rtc_prefix_len is None:
        rtc_prefix_len = chunk_len // 2
    if rtc_prefix_len < 0:
        raise ValueError("rtc_prefix_len must be >= 0")
    if rtc_prefix_len > chunk_len:
        raise ValueError("rtc_prefix_len must be <= chunk_len")

    N = dataset["rewards"].shape[0]
    is_sparse = "antmaze" in dataset_name

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    mc_returns_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    episode_ends = []
    last_transition_idx = -1
    episode_rewards = []
    episode_terminals = []

    for i in range(N - 1):
        if episode_step == 0:
            episode_rewards = []
            episode_terminals = []

        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition
            mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
            episode_step = 0
            if last_transition_idx >= 0 and (not episode_ends or episode_ends[-1] != last_transition_idx):
                episode_ends.append(last_transition_idx)
            continue
        if done_bool or final_timestep:
            episode_step = 0

        episode_rewards.append(reward)
        episode_terminals.append(done_bool)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        last_transition_idx = len(obs_) - 1
        if done_bool or final_timestep:
            if not episode_ends or episode_ends[-1] != last_transition_idx:
                episode_ends.append(last_transition_idx)
        episode_step += 1

    if episode_step != 0:
        mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)

    print("SHAPE", np.array(mc_returns_).shape, np.array(reward_).shape, np.array(done_).shape)
    assert np.array(mc_returns_).shape == np.array(reward_).shape

    cls_rewards = np.array(mc_returns_)

    episode_ends = [-1] + episode_ends
    intervals = [(episode_ends[i] + 1, episode_ends[i + 1] + 1) for i in range(len(episode_ends) - 1)]

    c_obs = []
    c_action = []
    c_next_obs = []
    c_next_action = []
    c_reward = []
    c_done = []
    c_prev_action = []
    c_valid = []

    discount_powers = discount ** np.arange(chunk_len)
    action_dim = action_[0].shape[-1] if action_ else 0
    zero_next_action = np.zeros((chunk_len, action_dim), dtype=np.float32)

    for ep_start, ep_end in intervals:
        ep_len = ep_end - ep_start
        if ep_len <= 0:
            continue
        last_start = ep_end - chunk_len
        if last_start < ep_start:
            starts = [ep_start]
        else:
            starts = range(ep_start, last_start + 1, chunk_stride)
        for t in starts:
            slice_end = min(t + chunk_len, ep_end)
            length = slice_end - t
            action_chunk = np.zeros((chunk_len, action_dim), dtype=np.float32)
            reward_chunk = np.zeros((chunk_len,), dtype=np.float32)
            done_chunk = np.zeros((chunk_len,), dtype=np.float32)
            if length > 0:
                action_chunk[:length] = np.asarray(action_[t:slice_end])
                reward_chunk[:length] = np.asarray(reward_[t:slice_end])
                done_chunk[:length] = np.asarray(done_[t:slice_end]).astype(np.float32)

            no_next = length < chunk_len
            done_flag = bool(np.any(done_chunk)) or no_next
            next_action_chunk = np.zeros((chunk_len, action_dim), dtype=np.float32)
            if not done_flag:
                next_start = t + chunk_len
                next_action_end = min(next_start + chunk_len, ep_end)
                next_len = max(0, next_action_end - next_start)
                if next_len < chunk_len:
                    done_flag = True
                if next_len > 0:
                    next_action_chunk[:next_len] = np.asarray(
                        action_[next_start:next_start + next_len]
                    )

            cum_done = np.cumsum(done_chunk)
            done_mask = (cum_done - done_chunk) > 0
            reward_mask = 1.0 - done_mask.astype(np.float32)
            discounted_reward = np.sum(discount_powers * reward_chunk * reward_mask)
            valid = np.zeros((chunk_len,), dtype=np.float32)
            if length > 0:
                valid[:length] = 1.0 - done_mask[:length].astype(np.float32)

            c_obs.append(obs_[t])
            c_action.append(action_chunk)
            c_next_obs.append(next_obs_[min(t + chunk_len - 1, ep_end - 1)])
            c_next_action.append(next_action_chunk)
            c_reward.append(discounted_reward)
            c_done.append(done_flag)
            c_valid.append(valid)
            if rtc_prefix_len > 0:
                prev = np.zeros((rtc_prefix_len, action_dim), dtype=np.float32)
                available = t - ep_start
                take = min(rtc_prefix_len, available)
                if take > 0:
                    prev[-take:] = np.asarray(action_[t - take:t])
                c_prev_action.append(prev)
            else:
                c_prev_action.append(np.zeros((0, action_dim), dtype=np.float32))

    train_data = {
        "observations": np.asarray(c_obs),
        "actions": np.asarray(c_action),
        "prev_actions": np.asarray(c_prev_action),
        "next_observations": np.asarray(c_next_obs),
        "next_actions": np.asarray(c_next_action),
        "rewards": np.asarray(c_reward),
        "terminals": np.asarray(c_done),
        "valid": np.asarray(c_valid),
    }
    print("Trains obs size:", len(train_data['observations']))

    return train_data, jnp.min(cls_rewards), jnp.max(cls_rewards)


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

    def create_from_d4rl(
            self,
            dataset_name: str,
            normalize_reward: bool = False,
            is_normalize: bool = False,
            discount: float = 0.99,
            chunk_len: int = 5,
            chunk_stride: int = 1,
            rtc_prefix_len: Optional[int] = None,
    ):
        d4rl_data, self.min, self.max = qlearning_dataset(
            gym.make(dataset_name),
            dataset_name,
            discount=discount,
            chunk_len=chunk_len,
            chunk_stride=chunk_stride,
            rtc_prefix_len=rtc_prefix_len,
            normalize_reward=normalize_reward,
        )
        print("Min/Max", self.min, self.max)

        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "prev_actions": jnp.asarray(d4rl_data["prev_actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
            "valid": jnp.asarray(d4rl_data["valid"], dtype=jnp.float32),
        }

        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        self.data = buffer

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
        batch = jax.tree_util.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def sample_n_first(
            self, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jnp.arange(0, batch_size)
        batch = jax.tree_util.tree_map(lambda arr: arr[indices], self.data)
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
        new_accumulators = {}
        for key, (acc, steps) in self.accumulators.items():
            if key in updates:
                value = updates[key]
                new_accumulators[key] = (acc + value, steps + 1)
            else:
                new_accumulators[key] = (acc, steps)
        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(
        arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


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
        batch_stats: jax.Array,
        critic: "CriticTrainState",
        action_fn: Callable,
        log_prob_fn: Optional[Callable],
        num_episodes: int,
        seed: int,
        action_noise: float = 0,
        state_noise: float = 0,
        rtc_prefix_len: Optional[int] = None,
        q_infer_step_size: float = 0.0,
        q_infer_steps: int = 0,
        use_nf: bool = False,
        nf_eval_num_samples: int = 1,
        nf_eval_z_scale: float = 1.0,
        nf_eval_z_clip: float = 0.0,
        nf_eval_select: str = "q",
) -> Tuple[np.ndarray, Dict]:
    if rtc_prefix_len is None:
        rtc_prefix_len = 0

    key = jax.random.PRNGKey(seed=seed)
    use_refine = q_infer_step_size > 0 and q_infer_steps > 0
    q_infer_steps = max(0, int(q_infer_steps))

    @partial(jax.jit, static_argnums=(5,))
    def policy_action(params_j, batch_stats_j, obs_j, prev_actions_j, rng_j, num_samples_j):
        return action_fn(
            {"params": params_j, "batch_stats": batch_stats_j},
            obs_j,
            prev_actions_j,
            rng_j,
            num_samples_j,
            nf_eval_z_scale,
            nf_eval_z_clip,
        )

    @jax.jit
    def eval_q(obs_j, action_j):
        logits = critic.apply_fn(critic.params, obs_j, action_j)
        probs = nn.softmax(logits, axis=-1)
        q_values = transform_from_probs(probs, critic.support).min(0)
        return q_values

    def _refine_action_chunk(obs_j, action_j):
        def q_value(a):
            return eval_q(obs_j, a)[0]

        def body(_, a):
            grad = jax.grad(q_value)(a)
            grad_norm = jnp.linalg.norm(grad) + 1e-8
            return jnp.clip(a + q_infer_step_size * (grad / grad_norm), -1.0, 1.0)

        return jax.lax.fori_loop(0, q_infer_steps, body, action_j)

    refine_action_chunk = jax.jit(_refine_action_chunk)

    eval_states = []
    eval_actions = []
    returns = []

    if hasattr(env, "num_envs"):
        num_envs = int(env.num_envs)
        action_dim = env.single_action_space.shape[0]
        obs = env.reset()
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        action_buffers = [[] for _ in range(num_envs)]
        prev_prefix = np.zeros((num_envs, rtc_prefix_len, action_dim), dtype=np.float32)
        done = np.zeros(num_envs, dtype=bool)
        base_eps = num_episodes // num_envs
        remainder = num_episodes % num_envs
        per_env_target = np.array(
            [base_eps + (1 if i < remainder else 0) for i in range(num_envs)],
            dtype=np.int32,
        )
        per_env_done = np.zeros(num_envs, dtype=np.int32)

        while len(returns) < num_episodes:
            actions = []
            for i in range(num_envs):
                if done[i]:
                    actions.append(np.zeros(action_dim, dtype=np.float32))
                    continue
                key, actions_key, states_key = jax.random.split(key, 3)
                obs_i = obs[i] + jax.random.normal(states_key, obs[i].shape) * state_noise
                eval_states.append(obs_i)

                if not action_buffers[i]:
                    action_chunk = np.asarray(jax.device_get(
                        policy_action(
                            params, batch_stats, obs_i, prev_prefix[i], actions_key, nf_eval_num_samples
                        )
                    ))
                    if use_nf and nf_eval_num_samples > 1:
                        obs_j = jnp.asarray(obs_i)[None, ...]
                        cand_actions = jnp.asarray(action_chunk)
                        if cand_actions.ndim == 3:
                            cand_actions = cand_actions[None, ...]
                        cand_actions = cand_actions[0]
                        if nf_eval_select == "likelihood":
                            obs_rep = jnp.repeat(obs_j, nf_eval_num_samples, axis=0)
                            prev_rep = jnp.repeat(
                                jnp.asarray(prev_prefix[i])[None, ...], nf_eval_num_samples, axis=0
                            )
                            logp = log_prob_fn(
                                params, batch_stats, obs_rep, prev_rep, cand_actions
                            )
                            best_idx = int(jax.device_get(jnp.argmax(logp)))
                        else:
                            obs_rep = jnp.repeat(obs_j, nf_eval_num_samples, axis=0)
                            q_vals = eval_q(obs_rep, cand_actions)
                            best_idx = int(jax.device_get(jnp.argmax(q_vals)))
                        action_chunk = np.asarray(jax.device_get(cand_actions[best_idx]))
                    action_buffers[i] = list(action_chunk) if action_chunk.ndim > 1 else [action_chunk]
                    if rtc_prefix_len > 0:
                        prev_prefix[i] = action_chunk[-rtc_prefix_len:].reshape(rtc_prefix_len, action_dim)

                    if use_refine:
                        obs_j = jnp.asarray(obs_i)[None, ...]
                        action_j = jnp.asarray(action_chunk)
                        if action_j.ndim == 2:
                            action_j = action_j[None, ...]
                        action_j = refine_action_chunk(obs_j, action_j)
                        action_chunk = np.asarray(action_j[0])
                        action_buffers[i] = list(action_chunk)
                        if rtc_prefix_len > 0:
                            prev_prefix[i] = action_chunk[-rtc_prefix_len:]

                action = action_buffers[i].pop(0)
                eval_actions.append(action)
                action = jnp.clip(action + jax.random.normal(actions_key, action.shape) * action_noise, -1, 1)
                actions.append(action)

            step_result = env.step(np.stack(actions, axis=0))
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                step_done = np.logical_or(terminated, truncated)
            else:
                obs, reward, step_done, _ = step_result

            episode_returns += reward
            done = step_done

            if np.any(done):
                for i in np.where(done)[0]:
                    if per_env_done[i] < per_env_target[i]:
                        returns.append(float(episode_returns[i]))
                        per_env_done[i] += 1
                    if len(returns) >= num_episodes:
                        break
                    if per_env_done[i] < per_env_target[i]:
                        reset_out = env.envs[i].reset()
                        obs[i] = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                        episode_returns[i] = 0.0
                        action_buffers[i] = []
                        prev_prefix[i] = 0.0
                    else:
                        done[i] = True
    else:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        action_dim = env.action_space.shape[0]

        for _ in trange(num_episodes, desc="Eval", leave=False):
            obs, done = env.reset(), False
            total_reward = 0.0
            action_buffer = []
            prev_prefix = np.zeros((rtc_prefix_len, action_dim), dtype=np.float32)
            while not done:
                key, actions_key, states_key = jax.random.split(key, 3)
                obs = obs + jax.random.normal(states_key, obs.shape) * state_noise

                eval_states.append(obs)
                if not action_buffer:
                    action_chunk = np.asarray(jax.device_get(
                        policy_action(
                            params, batch_stats, obs, prev_prefix, actions_key, nf_eval_num_samples
                        )
                    ))
                    if use_nf and nf_eval_num_samples > 1:
                        obs_j = jnp.asarray(obs)[None, ...]
                        cand_actions = jnp.asarray(action_chunk)
                        if cand_actions.ndim == 3:
                            cand_actions = cand_actions[None, ...]
                        cand_actions = cand_actions[0]
                        if nf_eval_select == "likelihood":
                            obs_rep = jnp.repeat(obs_j, nf_eval_num_samples, axis=0)
                            prev_rep = jnp.repeat(
                                jnp.asarray(prev_prefix)[None, ...], nf_eval_num_samples, axis=0
                            )
                            logp = log_prob_fn(
                                params, batch_stats, obs_rep, prev_rep, cand_actions
                            )
                            best_idx = int(jax.device_get(jnp.argmax(logp)))
                        else:
                            obs_rep = jnp.repeat(obs_j, nf_eval_num_samples, axis=0)
                            q_vals = eval_q(obs_rep, cand_actions)
                            best_idx = int(jax.device_get(jnp.argmax(q_vals)))
                        action_chunk = np.asarray(jax.device_get(cand_actions[best_idx]))
                    if action_chunk.ndim == 1:
                        action_buffer = [action_chunk]
                    else:
                        action_buffer = list(action_chunk)
                    if rtc_prefix_len > 0:
                        if action_chunk.ndim == 1:
                            prev_prefix = action_chunk[-rtc_prefix_len:].reshape(rtc_prefix_len, action_dim)
                        else:
                            prev_prefix = action_chunk[-rtc_prefix_len:]
                    if use_refine:
                        obs_j = jnp.asarray(obs)[None, ...]
                        action_j = jnp.asarray(action_chunk)
                        if action_j.ndim == 2:
                            action_j = action_j[None, ...]
                        action_j = refine_action_chunk(obs_j, action_j)
                        action_chunk = np.asarray(action_j[0])
                        action_buffer = list(action_chunk)
                        if rtc_prefix_len > 0:
                            prev_prefix = action_chunk[-rtc_prefix_len:]
                action = action_buffer.pop(0)
                eval_actions.append(action)
                action = jnp.clip(action + jax.random.normal(actions_key, action.shape) * action_noise, -1, 1)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            returns.append(total_reward)

    eval_batch = {
        "states": jnp.array(eval_states),
        "actions": jnp.array(eval_actions),
    }
    return np.array(returns), eval_batch


class CriticTrainState(TrainState):
    target_params: FrozenDict
    support: jax.Array
    sigma: float


class ActorTrainState(TrainState):
    target_params: FrozenDict
    dropout_key: jax.Array
    batch_stats: Any
    target_batch_stats: Any
    constants: Any = None
    target_constants: Any = None


class ValueTrainState(TrainState):
    pass


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
        use_nf: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, random_action_key, input_noise_key, bc_noise_key, grad_noise_key = jax.random.split(key, 5)
    dropout_key, new_dropout_key = jax.random.split(actor.dropout_key, 2)
    sample_key, dropout_apply_key, dropout_log_key = jax.random.split(dropout_key, 3)

    in_noise = jax.random.normal(input_noise_key, batch["states"].shape) * input_noise
    b_noise = jax.random.normal(bc_noise_key, batch["actions"].shape) * bc_noise

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        if use_nf:
            actions = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats, 'constants': actor.constants},
                batch["states"] + in_noise,
                batch["prev_actions"],
                rng=sample_key,
                train=True,
                method=NFActor.sample,
                rngs={'dropout': dropout_apply_key},
            )
            updates = {'batch_stats': actor.batch_stats}
            bc_actions = batch["actions"] + b_noise
            log_probs = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats, 'constants': actor.constants},
                bc_actions,
                batch["states"] + in_noise,
                batch["prev_actions"],
                train=True,
                method=NFActor.log_prob,
                rngs={'dropout': dropout_log_key},
            )
            bc_penalty = -log_probs
            valid = batch.get("valid", None)
            if valid is not None:
                valid_ratio = jnp.clip(jnp.mean(valid, axis=1), 0.0, 1.0)
                bc_penalty = bc_penalty * valid_ratio
            nll = jnp.mean(bc_penalty)
        else:
            (actions, preact), updates = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats},
                batch["states"] + in_noise,
                batch["prev_actions"],
                True, rngs={'dropout': dropout_key},
                mutable=['batch_stats'],
            )
            valid = batch.get("valid", None)
            if valid is None:
                bc_penalty = jnp.sum((actions - batch["actions"] + b_noise) ** 2, axis=(-1, -2))
            else:
                bc_penalty = jnp.sum(
                    (actions - batch["actions"] + b_noise) ** 2 * valid[..., None],
                    axis=(-1, -2),
                )
            nll = None

        logits = critic.apply_fn(critic.params, batch["states"], actions)
        probs = nn.softmax(logits, axis=-1)
        q_values = transform_from_probs(probs, critic.support).min(0)

        lmbda = 1
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1 / jnp.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()

        # logging stuff
        random_actions = jax.random.uniform(
            random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
        )
        if valid is None:
            bc_mse_policy = jnp.sum((actions - batch["actions"] + b_noise) ** 2, axis=(-1, -2)).mean()
            action_mse = ((actions - batch["actions"]) ** 2).mean()
            bc_mse_random = jnp.sum((random_actions - batch["actions"]) ** 2, axis=(-1, -2)).mean()
        else:
            bc_mse_policy = jnp.sum(
                (actions - batch["actions"] + b_noise) ** 2 * valid[..., None], axis=(-1, -2)
            ).mean()
            action_mse = jnp.sum(
                (actions - batch["actions"]) ** 2 * valid[..., None], axis=(-1, -2)
            ).mean()
            bc_mse_random = jnp.sum(
                (random_actions - batch["actions"]) ** 2 * valid[..., None], axis=(-1, -2)
            ).mean()
        metrics_payload = {
            "actor_loss": loss,
            "bc_mse_policy": bc_mse_policy,
            "bc_mse_random": bc_mse_random,
            "action_mse": action_mse,
        }
        if nll is not None:
            metrics_payload["nll"] = nll
        new_metrics = metrics.update(metrics_payload)
        return loss, (updates, new_metrics)

    grads, (updates, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

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
    # print(grads, flush=True)
    new_actor = actor.apply_gradients(grads=grads)

    new_actor = new_actor.replace(
        batch_stats=updates['batch_stats'],
    )
    new_actor = new_actor.replace(
        target_params=optax.incremental_update(actor.params, actor.target_params, tau),
        target_batch_stats=optax.incremental_update(actor.batch_stats, actor.target_batch_stats, tau),
        dropout_key=new_dropout_key,
    )
    new_critic = critic.replace(
        target_params=optax.incremental_update(critic.params, critic.target_params, tau)
    )

    actor_params = new_actor.params
    actor_params = jax.tree_util.tree_map(lambda x: x.reshape(-1), actor_params)
    flat_vals, _ = jax.tree_util.tree_flatten(actor_params)
    flat_mean = jnp.mean(jnp.concatenate(flat_vals))

    new_metrics = new_metrics.update(
        {"weights/actor_weights_mean": flat_mean}
    )
    return key, new_actor, new_critic, new_metrics


def update_actor_bc(
        key: jax.random.PRNGKey,
        actor: TrainState,
        batch: Dict[str, jax.Array],
        beta: float,
        tau: float,
        input_noise: float,
        bc_noise: float,
        grad_noise: float,
        use_nf: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, random_action_key, input_noise_key, bc_noise_key, grad_noise_key = jax.random.split(key, 5)
    dropout_key, new_dropout_key = jax.random.split(actor.dropout_key, 2)
    sample_key, dropout_apply_key, dropout_log_key = jax.random.split(dropout_key, 3)

    in_noise = jax.random.normal(input_noise_key, batch["states"].shape) * input_noise
    b_noise = jax.random.normal(bc_noise_key, batch["actions"].shape) * bc_noise

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        if use_nf:
            actions = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats, 'constants': actor.constants},
                batch["states"] + in_noise,
                batch["prev_actions"],
                rng=sample_key,
                train=True,
                method=NFActor.sample,
                rngs={'dropout': dropout_apply_key},
            )
            updates = {'batch_stats': actor.batch_stats}
            bc_actions = batch["actions"] + b_noise
            log_probs = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats, 'constants': actor.constants},
                bc_actions,
                batch["states"] + in_noise,
                batch["prev_actions"],
                train=True,
                method=NFActor.log_prob,
                rngs={'dropout': dropout_log_key},
            )
            bc_penalty = -log_probs
            valid = batch.get("valid", None)
            if valid is not None:
                valid_ratio = jnp.clip(jnp.mean(valid, axis=1), 0.0, 1.0)
                bc_penalty = bc_penalty * valid_ratio
            nll = jnp.mean(bc_penalty)
        else:
            (actions, _), updates = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats},
                batch["states"] + in_noise,
                batch["prev_actions"],
                True, rngs={'dropout': dropout_key},
                mutable=['batch_stats'],
            )
            valid = batch.get("valid", None)
            if valid is None:
                bc_penalty = jnp.sum((actions - batch["actions"] + b_noise) ** 2, axis=(-1, -2))
            else:
                bc_penalty = jnp.sum(
                    (actions - batch["actions"] + b_noise) ** 2 * valid[..., None],
                    axis=(-1, -2),
                )
            nll = None

        loss = (beta * bc_penalty).mean()

        random_actions = jax.random.uniform(
            random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
        )
        if valid is None:
            bc_mse_policy = jnp.sum((actions - batch["actions"] + b_noise) ** 2, axis=(-1, -2)).mean()
            action_mse = ((actions - batch["actions"]) ** 2).mean()
            bc_mse_random = jnp.sum((random_actions - batch["actions"]) ** 2, axis=(-1, -2)).mean()
        else:
            bc_mse_policy = jnp.sum(
                (actions - batch["actions"] + b_noise) ** 2 * valid[..., None], axis=(-1, -2)
            ).mean()
            action_mse = jnp.sum(
                (actions - batch["actions"]) ** 2 * valid[..., None], axis=(-1, -2)
            ).mean()
            bc_mse_random = jnp.sum(
                (random_actions - batch["actions"]) ** 2 * valid[..., None], axis=(-1, -2)
            ).mean()
        metrics_payload = {
            "actor_loss": loss,
            "bc_mse_policy": bc_mse_policy,
            "bc_mse_random": bc_mse_random,
            "action_mse": action_mse,
        }
        if nll is not None:
            metrics_payload["nll"] = nll
        new_metrics = metrics.update(metrics_payload)
        return loss, (updates, new_metrics)

    grads, (updates, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

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
        batch_stats=updates['batch_stats'],
        target_params=optax.incremental_update(actor.params, actor.target_params, tau),
        target_batch_stats=optax.incremental_update(actor.batch_stats, actor.target_batch_stats, tau),
        dropout_key=new_dropout_key,
    )

    actor_params = new_actor.params
    actor_params = jax.tree_util.tree_map(lambda x: x.reshape(-1), actor_params)
    flat_vals, _ = jax.tree_util.tree_flatten(actor_params)
    flat_mean = jnp.mean(jnp.concatenate(flat_vals))

    new_metrics = new_metrics.update(
        {"weights/actor_weights_mean": flat_mean}
    )
    return key, new_actor, new_metrics


def update_value(
        value: ValueTrainState,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        expectile: float,
) -> Tuple[ValueTrainState, jax.Array]:
    def value_loss_fn(value_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        v = value.apply_fn(value_params, batch["states"])
        logits = critic.apply_fn(critic.params, batch["states"], batch["actions"])
        probs = nn.softmax(logits, axis=-1)
        q_values = transform_from_probs(probs, critic.support).min(0)
        diff = q_values - v
        weight = jnp.where(diff > 0, expectile, 1 - expectile)
        loss = (weight * (diff ** 2)).mean()
        return loss, v

    (loss, _), grads = jax.value_and_grad(value_loss_fn, has_aux=True)(value.params)
    new_value = value.apply_gradients(grads=grads)
    return new_value, loss


def update_critic_iql(
        critic: CriticTrainState,
        value: ValueTrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        chunk_len: int,
) -> Tuple[CriticTrainState, jax.Array, jax.Array]:
    v_next = value.apply_fn(value.params, batch["next_states"])
    target_q = batch["rewards"] + (1 - batch["dones"]) * (gamma ** chunk_len) * v_next

    def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = transform_from_probs(nn.softmax(q, axis=-1), critic.support).min(0).mean()
        target_probs = transform_to_probs(target_q, critic.support, critic.sigma)
        loss = optax.softmax_cross_entropy(logits=q, labels=target_probs[None, ...]).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    return new_critic, loss, q_min

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
        chunk_len: int,
        rtc_prefix_len: int,
        use_nf: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key, noise_key = jax.random.split(key, 3)

    if rtc_prefix_len > 0:
        next_prev_actions = batch["actions"][:, -rtc_prefix_len:, :]
    else:
        action_dim = batch["actions"].shape[-1]
        next_prev_actions = jnp.zeros((batch["actions"].shape[0], 0, action_dim), dtype=batch["actions"].dtype)

    if use_nf:
        next_actions = actor.apply_fn(
            {
                'params': actor.target_params,
                'batch_stats': actor.target_batch_stats,
                'constants': actor.target_constants,
            },
            batch["next_states"],
            next_prev_actions,
            rng=actions_key,
            train=False,
            method=NFActor.sample,
        )
    else:
        next_actions, preact = actor.apply_fn(
            {
                'params': actor.target_params,
                'batch_stats': actor.target_batch_stats,
            },
            batch["next_states"],
            next_prev_actions,
            False
        )
    noise = jnp.clip(
        (jax.random.normal(noise_key, next_actions.shape) * policy_noise),
        -noise_clip,
        noise_clip,
    )
    next_actions = jnp.clip(next_actions + noise, -1, 1)
    bc_penalty = jnp.sum((next_actions - batch["next_actions"]) ** 2, axis=(-1, -2))
    logits = critic.apply_fn(critic.target_params, batch["next_states"], next_actions)
    probs = nn.softmax(logits, axis=-1)
    next_q = transform_from_probs(probs, critic.support).min(0)
    next_q = next_q - beta * bc_penalty

    target_q = batch["rewards"] + (1 - batch["dones"]) * (gamma ** chunk_len) * next_q

    def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = transform_from_probs(nn.softmax(q, axis=-1), critic.support).min(0).mean()
        target_probs = transform_to_probs(target_q, critic.support, critic.sigma)

        loss = optax.softmax_cross_entropy(logits=q, labels=target_probs[None, ...]).mean(1).sum(0)
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
        chunk_len: int,
        rtc_prefix_len: int,
        use_nf: bool,
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
        chunk_len,
        rtc_prefix_len,
        use_nf,
        metrics,
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key, actor, new_critic, batch, actor_bc_coef, tau, normalize_q, actor_input_noise, actor_bc_noise,
        actor_grad_noise, use_nf, new_metrics
    )
    return key, new_actor, new_critic, new_metrics


def update_iql(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        value: ValueTrainState,
        batch: Dict[str, Any],
        metrics: Metrics,
        gamma: float,
        actor_bc_coef: float,
        tau: float,
        chunk_len: int,
        iql_expectile: float,
        normalize_q: bool,
        actor_input_noise: float,
        actor_bc_noise: float,
        actor_grad_noise: float,
        use_nf: bool,
) -> Tuple[jax.random.PRNGKey, TrainState, CriticTrainState, ValueTrainState, Metrics]:
    new_value, value_loss = update_value(value, critic, batch, iql_expectile)
    new_critic, critic_loss, q_min = update_critic_iql(
        critic, new_value, batch, gamma, chunk_len
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key, actor, new_critic, batch, actor_bc_coef, tau, normalize_q, actor_input_noise, actor_bc_noise,
        actor_grad_noise, use_nf, metrics
    )
    new_metrics = new_metrics.update(
        {
            "critic_loss": critic_loss,
            "q_min": q_min,
            "value_loss": value_loss,
        }
    )
    return key, new_actor, new_critic, new_value, new_metrics


def update_iql_no_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        value: ValueTrainState,
        batch: Dict[str, Any],
        metrics: Metrics,
        gamma: float,
        chunk_len: int,
        iql_expectile: float,
) -> Tuple[jax.random.PRNGKey, TrainState, CriticTrainState, ValueTrainState, Metrics]:
    new_value, value_loss = update_value(value, critic, batch, iql_expectile)
    new_critic, critic_loss, q_min = update_critic_iql(
        critic, new_value, batch, gamma, chunk_len
    )
    new_metrics = metrics.update(
        {
            "critic_loss": critic_loss,
            "q_min": q_min,
            "value_loss": value_loss,
        }
    )
    return key, actor, new_critic, new_value, new_metrics


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
        chunk_len: int,
        rtc_prefix_len: int,
        use_nf: bool,
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
        chunk_len,
        rtc_prefix_len,
        use_nf,
        metrics,
    )
    return key, actor, new_critic, new_metrics


def update_critic_warmup(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, Any],
        gamma: float,
        metrics: Metrics,
        critic_bc_coef: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        chunk_len: int,
        rtc_prefix_len: int,
        use_nf: bool,
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
        chunk_len,
        rtc_prefix_len,
        use_nf,
        metrics,
    )
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(new_critic.params, critic.target_params, tau)
    )
    return key, actor, new_critic, new_metrics


def update_refinement(
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
        use_nf: bool,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, new_actor, new_critic, new_metrics = update_actor(
        key, actor, critic, batch, actor_bc_coef, tau, normalize_q, actor_input_noise, actor_bc_noise,
        actor_grad_noise, use_nf, metrics
    )
    return key, new_actor, new_critic, new_metrics


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array, prev_actions: jax.Array) -> jax.Array:
        action = actor.apply_fn(
            {
                "params": actor.params,
                "batch_stats": actor.batch_stats,
            },
            obs, prev_actions, False
        )[0]
        return action

    return _action_fn


@pyrallis.wrap()
def train(config: Config):
    config.project = "ActoReg"
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    if config.rtc_prefix_len is None:
        config.rtc_prefix_len = config.action_chunk_len // 2
    if config.rtc_prefix_len > config.action_chunk_len:
        raise ValueError("rtc_prefix_len must be <= action_chunk_len")
    if config.use_iql and config.num_refinement_epochs > 0:
        raise ValueError("IQL mode does not support refinement epochs")

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
        config.dataset_name, config.normalize_reward, config.normalize_states, discount=config.gamma,
        chunk_len=config.action_chunk_len,
        chunk_stride=config.action_chunk_stride,
        rtc_prefix_len=config.rtc_prefix_len,
    )
    random.seed(config.train_seed)
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key, dropout_key = jax.random.split(key, 4)

    if config.eval_num_envs > 1:
        def make_eval_env(i):
            def _fn():
                env = make_env(config.dataset_name, seed=config.eval_seed + i)
                return wrap_env(env, buffer.mean, buffer.std)
            return _fn
        eval_env = gym.vector.SyncVectorEnv(
            [make_eval_env(i) for i in range(config.eval_num_envs)]
        )
    else:
        eval_env = make_env(config.dataset_name, seed=config.eval_seed)
        eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    init_prev_actions = buffer.data["prev_actions"][0][None, ...]

    if config.use_nf:
        actor_module = NFActor(
            action_dim=init_action.shape[-1],
            chunk_len=config.action_chunk_len,
            hidden_dim=config.nf_hidden_dim,
            n_hiddens=config.nf_n_hiddens,
            num_layers=config.nf_num_layers,
            use_transformer=config.nf_use_transformer,
            scale_max=config.nf_scale_max,
            base_dist=config.nf_base_dist,
            use_plu=config.nf_use_plu,
            use_layernorm=config.nf_use_layernorm,
            dropout_rate=config.nf_dropout,
            deterministic_layers=config.nf_det_layers,
        )
        reset_module = NFActor(
            action_dim=init_action.shape[-1],
            chunk_len=config.action_chunk_len,
            hidden_dim=config.nf_hidden_dim,
            n_hiddens=config.nf_n_hiddens,
            num_layers=config.nf_num_layers,
            use_transformer=config.nf_use_transformer,
            scale_max=config.nf_scale_max,
            base_dist=config.nf_base_dist,
            use_plu=config.nf_use_plu,
            use_layernorm=config.nf_use_layernorm,
            dropout_rate=config.nf_dropout,
            deterministic_layers=config.nf_det_layers,
        )
    else:
        if config.actor_prereset_mode:
            actor_module = DetActor(
                action_dim=init_action.shape[-1],
                chunk_len=config.action_chunk_len,
                rtc_prefix_len=config.rtc_prefix_len,
                hidden_dim=config.hidden_dim,
                layernorm=config.actor_ln,
                featurenorm=config.actor_fn,
                groupnorm=config.actor_gn,
                batchnorm=config.actor_bn,
                spectralnorm=config.actor_sn,
                dropout_rate=config.actor_dropout,
                n_hiddens=config.actor_n_hiddens,
            )
        else:
            actor_module = DetActor(
                action_dim=init_action.shape[-1],
                chunk_len=config.action_chunk_len,
                rtc_prefix_len=config.rtc_prefix_len,
                hidden_dim=config.hidden_dim,
                layernorm=False,
                featurenorm=False,
                groupnorm=False,
                spectralnorm=False,
                dropout_rate=0.0,
                n_hiddens=config.actor_n_hiddens,
            )
        reset_module = DetActor(
            action_dim=init_action.shape[-1],
            chunk_len=config.action_chunk_len,
            rtc_prefix_len=config.rtc_prefix_len,
            hidden_dim=config.hidden_dim,
            layernorm=config.actor_ln,
            featurenorm=config.actor_fn,
            groupnorm=config.actor_gn,
            dropout_rate=config.actor_dropout,
            n_hiddens=config.actor_n_hiddens,
        )

    if config.decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(config.actor_learning_rate,
                                                  config.num_epochs * config.num_updates_on_epoch)
        optimizer = adamw_elastic(learning_rate=schedule_fn, weight_decay=config.actor_wd, l1_ratio=config.l1_ratio)
    elif config.decay_schedule == "linear":
        schedule_fn = optax.linear_schedule(config.actor_learning_rate, config.actor_learning_rate / 10,
                                            config.num_epochs * config.num_updates_on_epoch)
        optimizer = adamw_elastic(learning_rate=schedule_fn, weight_decay=config.actor_wd, l1_ratio=config.l1_ratio)
    elif config.decay_schedule == "exp":
        schedule_fn = optax.exponential_decay(config.actor_learning_rate,
                                              config.num_epochs * config.num_updates_on_epoch, 0.99)
        optimizer = adamw_elastic(learning_rate=schedule_fn, weight_decay=config.actor_wd, l1_ratio=config.l1_ratio)
    else:
        optimizer = adamw_elastic(learning_rate=config.actor_learning_rate, weight_decay=config.actor_wd, l1_ratio=config.l1_ratio)

    if config.use_nf:
        init_vars = actor_module.init(
            {"params": actor_key, "mask": actor_key},
            init_state,
            init_prev_actions,
            rng=actor_key,
            method=NFActor.sample,
        )
    else:
        init_vars = actor_module.init(actor_key, init_state, init_prev_actions, False)
    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=init_vars['params'],
        batch_stats=init_vars['batch_stats'] if 'batch_stats' in init_vars else {},
        target_params=init_vars['params'],
        target_batch_stats=init_vars['batch_stats'] if 'batch_stats' in init_vars else {},
        constants=init_vars.get("constants", {}),
        target_constants=init_vars.get("constants", {}),
        dropout_key=dropout_key,
        tx=optimizer,
    )

    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=config.num_critics,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
        n_classes=config.n_classes,
    )

    v_min, v_max = config.v_min, config.v_max
    if v_min == float('inf'):
        v_min = buffer.min
    if v_max == float('inf'):
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

    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        support=jnp.linspace(v_min, v_max, config.n_classes + 1, dtype=jnp.float32),
        sigma=config.sigma_frac * (v_max - v_min) / config.n_classes,
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )
    value = None
    if config.use_iql:
        value_module = Value(
            hidden_dim=config.hidden_dim,
            layernorm=config.critic_ln,
            n_hiddens=config.critic_n_hiddens,
        )
        value = ValueTrainState.create(
            apply_fn=value_module.apply,
            params=value_module.init(critic_key, init_state),
            tx=optax.adam(learning_rate=config.value_learning_rate),
        )

    reset_mods = 1 if config.actor_prereset_mode else 0

    update_td3_partial = partial(
        update_td3,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise * reset_mods,
        actor_bc_noise=config.actor_bc_noise * reset_mods,
        actor_grad_noise=config.actor_grad_noise * reset_mods,
        chunk_len=config.action_chunk_len,
        rtc_prefix_len=config.rtc_prefix_len,
        use_nf=config.use_nf,
    )

    update_td3_no_targets_partial = partial(
        update_td3_no_targets,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        chunk_len=config.action_chunk_len,
        rtc_prefix_len=config.rtc_prefix_len,
        use_nf=config.use_nf,
    )
    update_iql_partial = partial(
        update_iql,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef,
        tau=config.tau,
        chunk_len=config.action_chunk_len,
        iql_expectile=config.iql_expectile,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise * reset_mods,
        actor_bc_noise=config.actor_bc_noise * reset_mods,
        actor_grad_noise=config.actor_grad_noise * reset_mods,
        use_nf=config.use_nf,
    )
    update_iql_no_actor_partial = partial(
        update_iql_no_actor,
        gamma=config.gamma,
        chunk_len=config.action_chunk_len,
        iql_expectile=config.iql_expectile,
    )

    update_refinement_partial = partial(
        update_refinement,
        gamma=config.gamma,
        actor_bc_coef=config.actor_bc_coef / config.refinement_div,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise,
        actor_bc_noise=config.actor_bc_noise,
        actor_grad_noise=config.actor_grad_noise,
        use_nf=config.use_nf,
    )
    update_actor_bc_partial = partial(
        update_actor_bc,
        beta=config.actor_bc_coef,
        tau=config.tau,
        input_noise=config.actor_input_noise * reset_mods,
        bc_noise=config.actor_bc_noise * reset_mods,
        grad_noise=config.actor_grad_noise * reset_mods,
        use_nf=config.use_nf,
    )
    update_critic_warmup_partial = partial(
        update_critic_warmup,
        gamma=config.gamma,
        critic_bc_coef=config.critic_bc_coef,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        chunk_len=config.action_chunk_len,
        rtc_prefix_len=config.rtc_prefix_len,
        use_nf=config.use_nf,
    )

    # metrics
    full_metrics_to_log = [
        "critic_loss",
        "q_min",
        "actor_loss",
        "batch_entropy",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
        "weights/actor_weights_mean",
    ]
    if config.use_iql:
        full_metrics_to_log.append("value_loss")
    actor_metrics_to_log = [
        "actor_loss",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
        "weights/actor_weights_mean",
    ]
    if config.use_nf:
        full_metrics_to_log.append("nll")
        actor_metrics_to_log.append("nll")
    critic_metrics_to_log = [
        "critic_loss",
        "q_min",
    ]
    if config.use_iql:
        critic_metrics_to_log.append("value_loss")
    delayed_updates = jnp.equal(
        jnp.arange(config.num_updates_on_epoch) % config.policy_freq, 0
    )

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
                )
                update = partial(
                    update_iql_no_actor_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    value=carry["value"],
                    batch=batch,
                    metrics=carry["metrics"],
                )
                key, new_actor, new_critic, new_value, new_metrics = jax.lax.cond(
                    do_update, full_update, update
                )
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "value": new_value,
                    "metrics": new_metrics,
                }
            else:
                full_update = partial(
                    update_td3_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    batch=batch,
                    metrics=carry["metrics"],
                )

                update = partial(
                    update_td3_no_targets_partial,
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    batch=batch,
                    metrics=carry["metrics"],
                )

                key, new_actor, new_critic, new_metrics = jax.lax.cond(
                    do_update, full_update, update
                )

                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "metrics": new_metrics,
                }
            return new_carry, None

        inner_carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
        }
        if config.use_iql:
            inner_carry["value"] = carry["value"]
        inner_carry, _ = jax.lax.scan(
            body, inner_carry, (batch_indices, delayed_updates)
        )
        return inner_carry

    def run_actor_bc_updates(carry, buffer_data):
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
            }
            return new_carry, None

        carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
        }
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
                )
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "value": new_value,
                    "metrics": new_metrics,
                }
            else:
                key, new_actor, new_critic, new_metrics = update_critic_warmup_partial(
                    key=carry["key"],
                    actor=carry["actor"],
                    critic=carry["critic"],
                    batch=batch,
                    metrics=carry["metrics"],
                )
                new_carry = {
                    "key": key,
                    "actor": new_actor,
                    "critic": new_critic,
                    "metrics": new_metrics,
                }
            return new_carry, None

        carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
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
            }
            return new_carry, None

        carry = {
            "key": key,
            "actor": carry["actor"],
            "critic": carry["critic"],
            "metrics": carry["metrics"],
        }
        carry, _ = jax.lax.scan(
            body, carry, batch_indices
        )
        return carry

    run_td3_updates = jax.jit(run_td3_updates)
    run_actor_bc_updates = jax.jit(run_actor_bc_updates)
    run_critic_updates = jax.jit(run_critic_updates)
    run_refinement_updates = jax.jit(run_refinement_updates)

    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
    }
    if config.use_iql:
        update_carry["value"] = value

    @partial(jax.jit, static_argnums=(4, 5, 6))
    def actor_action_fn(
        params: jax.Array,
        obs: jax.Array,
        prev_actions: jax.Array,
        rng: jax.Array,
        num_samples: int,
        z_scale: float,
        z_clip: float,
    ):
        if config.use_nf:
            if num_samples > 1:
                return actor.apply_fn(
                    {**params, "constants": actor.constants},
                    obs,
                    prev_actions,
                    rng=rng,
                    num_samples=num_samples,
                    z_scale=z_scale,
                    z_clip=z_clip,
                    train=False,
                    method=NFActor.sample_n,
                )
            return actor.apply_fn(
                {**params, "constants": actor.constants},
                obs,
                prev_actions,
                rng=rng,
                train=False,
                method=NFActor.sample,
            )
        return actor.apply_fn(params, obs, prev_actions, False)[0]

    @jax.jit
    def actor_logprob_fn(
        params: jax.Array,
        batch_stats: jax.Array,
        obs: jax.Array,
        prev_actions: jax.Array,
        actions: jax.Array,
    ):
        return actor.apply_fn(
            {
                "params": params,
                "batch_stats": batch_stats,
                "constants": actor.constants,
            },
            actions,
            obs,
            prev_actions,
            train=False,
            method=NFActor.log_prob,
        )

    il_end = config.il_warmup_epochs
    critic_end = il_end + config.critic_warmup_epochs

    for epoch in trange(config.num_epochs, desc="ReBRAC Epochs"):
        stage = "rl"
        if epoch < il_end:
            stage = "il"
        elif epoch < critic_end:
            stage = "critic"

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
                    reset_params = reset_module.init(
                        {"params": actor_key, "mask": actor_key},
                        init_state,
                        init_prev_actions,
                        rng=actor_key,
                        method=NFActor.sample,
                    )
                else:
                    reset_params = reset_module.init(actor_key, init_state, init_prev_actions, False)
                actor = ActorTrainState.create(
                    apply_fn=reset_module.apply,
                    params=reset_params,
                    batch_stats=reset_params['batch_stats'] if 'batch_stats' in reset_params else {},
                    target_params=reset_params,
                    target_batch_stats=reset_params['batch_stats'] if 'batch_stats' in reset_params else {},
                    constants=reset_params.get("constants", {}),
                    target_constants=reset_params.get("constants", {}),
                    dropout_key=dropout_key,
                    tx=optimizer,
                )
                update_carry.update(actor=actor)

        # metrics for accumulation during epoch and logging to wandb
        # we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(metrics_list)

        update_carry = update_fn(update_carry, buffer.data)
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log(
            {"epoch": epoch, **{f"ReBRAC/{k}": v for k, v in mean_metrics.items()}}
        )

        force_eval = epoch == il_end - 1 or epoch == critic_end - 1
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1 or force_eval:
            eval_select = "likelihood" if stage == "il" else "q"
            eval_q_step_size = 0.0 if stage == "il" else config.q_infer_step_size
            eval_q_steps = 0 if stage == "il" else config.q_infer_steps
            eval_returns, eval_batch = evaluate(
                eval_env,
                update_carry["actor"].params,
                update_carry["actor"].batch_stats,
                update_carry["critic"],
                actor_action_fn,
                actor_logprob_fn if config.use_nf else None,
                config.eval_episodes,
                seed=config.eval_seed,
                rtc_prefix_len=config.rtc_prefix_len,
                q_infer_step_size=eval_q_step_size,
                q_infer_steps=eval_q_steps,
                use_nf=config.use_nf,
                nf_eval_num_samples=config.nf_eval_num_samples,
                nf_eval_z_scale=config.nf_eval_z_scale,
                nf_eval_z_clip=config.nf_eval_z_clip,
                nf_eval_select=eval_select,
            )

            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            else:
                normalized_score = eval_env.envs[0].get_normalized_score(eval_returns) * 100.0

            eval_metrics = {
                "epoch": epoch,
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score),
            }
            if config.noisy_eval:
                for (sn, an) in [
                    (0.0, 0.2), (0.05, 0.0)
                ]:
                    returns, _ = evaluate(
                        eval_env,
                        update_carry["actor"].params,
                        update_carry["actor"].batch_stats,
                        update_carry["critic"],
                        actor_action_fn,
                        actor_logprob_fn if config.use_nf else None,
                        config.eval_episodes,
                        seed=config.eval_seed,
                        action_noise=an,
                        state_noise=sn,
                        rtc_prefix_len=config.rtc_prefix_len,
                        q_infer_step_size=eval_q_step_size,
                        q_infer_steps=eval_q_steps,
                        use_nf=config.use_nf,
                        nf_eval_num_samples=config.nf_eval_num_samples,
                        nf_eval_z_scale=config.nf_eval_z_scale,
                        nf_eval_z_clip=config.nf_eval_z_clip,
                        nf_eval_select=eval_select,
                    )
                    if hasattr(eval_env, "get_normalized_score"):
                        normalized_returns = eval_env.get_normalized_score(returns) * 100.0
                    else:
                        normalized_returns = eval_env.envs[0].get_normalized_score(returns) * 100.0
                    eval_metrics[f"eval/normalized_score_mean_sn_{sn}_an_{an}"] = np.mean(normalized_returns)
            wandb.log(
                eval_metrics
            )


if __name__ == "__main__":
    train()

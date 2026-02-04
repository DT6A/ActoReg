# Normalizing Flow imitation learning with chunked actions.

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import uuid
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import chex
import d4rl  # noqa: F401
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

from algorithms.nf_policy import NFActor
from algorithms.rebrac_cl_chunked import ReplayBuffer, make_env, wrap_env


@dataclass
class Config:
    # wandb params
    project: str = "ActoReg"
    group: str = "nf-il"
    name: str = "nf-il"
    # flow params
    hidden_dim: int = 64
    n_hiddens: int = 2
    num_layers: int = 8
    scale_max: float = 1.0
    base_dist: str = "normal"
    use_transformer: bool = True
    use_plu: bool = False
    use_layernorm: bool = False
    dropout_rate: float = 0.1
    deterministic_layers: int = 2
    # training params
    learning_rate: float = 3e-4
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 200
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    action_chunk_len: int = 6
    action_chunk_stride: int = 1
    rtc_prefix_len: Optional[int] = None
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10
    eval_num_envs: int = 1
    # general params
    train_seed: int = 0
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


@chex.dataclass
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Tuple[str, ...]) -> "Metrics":
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
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


class NFTrainState(TrainState):
    constants: FrozenDict
    dropout_key: jax.Array


def evaluate_nf(
    env: gym.Env,
    actor: NFTrainState,
    num_episodes: int,
    seed: int,
    rtc_prefix_len: Optional[int] = None,
    action_noise: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, jax.Array]]:
    if rtc_prefix_len is None:
        rtc_prefix_len = 0

    key = jax.random.PRNGKey(seed=seed)
    sample_chunks = jax.jit(
        lambda params, constants, obs_batch, prefix_batch, keys: jax.vmap(
            lambda o, p, k: actor.apply_fn(
                {"params": params, "constants": constants},
                o,
                p,
                k,
                train=False,
                method=NFActor.sample,
            )
        )(obs_batch, prefix_batch, keys)
    )
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

        while len(returns) < num_episodes:
            empty_mask = np.array([not buf for buf in action_buffers], dtype=bool)
            need_sample = np.logical_and(~done, empty_mask)
            if np.any(need_sample):
                key, sample_key = jax.random.split(key)
                keys = jax.random.split(sample_key, num_envs)
                obs_batch = jnp.asarray(obs)
                prefix_batch = jnp.asarray(prev_prefix)
                action_chunks = np.asarray(
                    sample_chunks(actor.params, actor.constants, obs_batch, prefix_batch, keys)
                )
                for i in np.where(need_sample)[0]:
                    action_chunk = action_chunks[i]
                    action_buffers[i] = (
                        list(action_chunk) if action_chunk.ndim > 1 else [action_chunk]
                    )
                    if rtc_prefix_len > 0:
                        prev_prefix[i] = action_chunk[-rtc_prefix_len:].reshape(
                            rtc_prefix_len, action_dim
                        )

            actions = []
            for i in range(num_envs):
                if done[i]:
                    actions.append(np.zeros(action_dim, dtype=np.float32))
                    continue
                key, action_key, noise_key = jax.random.split(key, 3)
                obs_i = obs[i]
                eval_states.append(obs_i)

                action = action_buffers[i].pop(0)
                eval_actions.append(action)
                if action_noise > 0:
                    noise = jax.random.normal(noise_key, action.shape) * action_noise
                    action = jnp.clip(action + noise, -1, 1)
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
                    returns.append(float(episode_returns[i]))
                    if len(returns) >= num_episodes:
                        break
                    reset_out = env.envs[i].reset()
                    obs[i] = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    episode_returns[i] = 0.0
                    action_buffers[i] = []
                    prev_prefix[i] = 0.0
                done = np.zeros(num_envs, dtype=bool)
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
                key, action_key, noise_key = jax.random.split(key, 3)
                eval_states.append(obs)
                if not action_buffer:
                    action_chunk = np.asarray(
                        actor.apply_fn(
                            {"params": actor.params, "constants": actor.constants},
                            obs,
                            prev_prefix,
                            action_key,
                            train=False,
                            method=NFActor.sample,
                        )
                    )
                    if action_chunk.ndim == 1:
                        action_buffer = [action_chunk]
                    else:
                        action_buffer = list(action_chunk)
                    if rtc_prefix_len > 0:
                        if action_chunk.ndim == 1:
                            prev_prefix = action_chunk[-rtc_prefix_len:].reshape(
                                rtc_prefix_len, action_dim
                            )
                        else:
                            prev_prefix = action_chunk[-rtc_prefix_len:]

                action = action_buffer.pop(0)
                eval_actions.append(action)
                if action_noise > 0:
                    noise = jax.random.normal(noise_key, action.shape) * action_noise
                    action = jnp.clip(action + noise, -1, 1)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            returns.append(total_reward)

    eval_batch = {
        "states": jnp.array(eval_states),
        "actions": jnp.array(eval_actions),
    }
    return np.array(returns), eval_batch


def update_actor(
    actor: NFTrainState,
    batch: Dict[str, jax.Array],
    metrics: Metrics,
) -> Tuple[NFTrainState, Metrics]:
    dropout_key, new_dropout_key = jax.random.split(actor.dropout_key, 2)

    def loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        logp = actor.apply_fn(
            {"params": params, "constants": actor.constants},
            batch["actions"],
            batch["states"],
            batch["prev_actions"],
            train=True,
            method=NFActor.log_prob,
            rngs={"dropout": dropout_key},
        )
        loss = -jnp.mean(logp)
        new_metrics = metrics.update(
            {
                "nll": loss,
                "logp_mean": jnp.mean(logp),
            }
        )
        return loss, new_metrics

    grads, new_metrics = jax.grad(loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(dropout_key=new_dropout_key)
    return new_actor, new_metrics


@pyrallis.wrap()
def train(config: Config):
    config.project = "ActoReg"
    dict_config = asdict(config)

    if config.rtc_prefix_len is None:
        config.rtc_prefix_len = config.action_chunk_len // 2
    if config.rtc_prefix_len > config.action_chunk_len:
        raise ValueError("rtc_prefix_len must be <= action_chunk_len")

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
        config.dataset_name,
        config.normalize_reward,
        config.normalize_states,
        chunk_len=config.action_chunk_len,
        chunk_stride=config.action_chunk_stride,
        rtc_prefix_len=config.rtc_prefix_len,
    )

    random.seed(config.train_seed)
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, dropout_key = jax.random.split(key, 3)

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

    actor_module = NFActor(
        action_dim=init_action.shape[-1],
        chunk_len=config.action_chunk_len,
        hidden_dim=config.hidden_dim,
        n_hiddens=config.n_hiddens,
        num_layers=config.num_layers,
        use_transformer=config.use_transformer,
        scale_max=config.scale_max,
        base_dist=config.base_dist,
        use_plu=config.use_plu,
        use_layernorm=config.use_layernorm,
        dropout_rate=config.dropout_rate,
        deterministic_layers=config.deterministic_layers,
    )

    init_vars = actor_module.init(
        {"params": actor_key, "dropout": actor_key, "mask": actor_key},
        init_state,
        init_prev_actions,
        actor_key,
        train=True,
        method=NFActor.sample,
    )

    actor = NFTrainState.create(
        apply_fn=actor_module.apply,
        params=init_vars["params"],
        constants=init_vars.get("constants", {}),
        dropout_key=dropout_key,
        tx=optax.adam(learning_rate=config.learning_rate),
    )

    update_actor_partial = jax.jit(update_actor)

    def run_updates(carry, buffer_data):
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
            actor, metrics = update_actor_partial(carry["actor"], batch, carry["metrics"])
            new_carry = {
                "key": carry["key"],
                "actor": actor,
                "metrics": metrics,
            }
            return new_carry, None

        carry, _ = jax.lax.scan(body, carry, batch_indices)
        carry["key"] = key
        return carry

    run_updates = jax.jit(run_updates)

    update_carry = {
        "key": key,
        "actor": actor,
    }

    metrics_to_log = ("nll", "logp_mean")

    for epoch in trange(config.num_epochs, desc="NF-IL Epochs"):
        update_carry["metrics"] = Metrics.create(metrics_to_log)
        update_carry = run_updates(update_carry, buffer.data)

        mean_metrics = update_carry["metrics"].compute()
        wandb.log(
            {"epoch": epoch, **{f"NF-IL/{k}": v for k, v in mean_metrics.items()}}
        )

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns, _ = evaluate_nf(
                eval_env,
                update_carry["actor"],
                config.eval_episodes,
                seed=config.eval_seed,
                rtc_prefix_len=config.rtc_prefix_len,
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
            wandb.log(eval_metrics)


if __name__ == "__main__":
    train()

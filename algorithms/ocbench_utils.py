"""State-only OCBench data and rollout adapters for ReBRAC-v2."""

from pathlib import Path

import numpy as np


def import_ocbench():
    try:
        import ocbench
    except ImportError as error:
        raise ImportError('Install OCBench with: pip install "ocbench[mjwarp]==1.0.1"') from error
    return ocbench


def environment_spec(name, backend="native"):
    ocbench = import_ocbench()
    if name.startswith("visual-"):
        raise ValueError("Only state-based OCBench environments are supported.")
    if backend not in {"native", "cpu", "mjwarp"}:
        raise ValueError("ocbench_eval_backend must be native, cpu, or mjwarp")
    if backend == "cpu" and "-cpu-" not in name:
        family, suffix = name.split("-", 1)
        name = f"{family}-cpu-{suffix}"
    elif backend == "mjwarp":
        name = name.replace("-cpu-", "-", 1)
    family, selected_backend, kwargs, horizon = ocbench.parse_env_spec(name)
    if kwargs.get("ob_type", "states") != "states":
        raise ValueError("Only state observations are supported.")
    return name, family, selected_backend, horizon


def training_paths(config):
    if config.ocbench_dataset_path:
        path = Path(config.ocbench_dataset_path).expanduser()
        if path.is_dir():
            paths = sorted(str(item) for item in path.glob("*.npz") if not item.name.endswith("-val.npz"))
        else:
            paths = [str(path)] if path.is_file() else []
        if config.ocbench_num_shards is not None:
            paths = paths[:config.ocbench_num_shards]
        if not paths:
            raise FileNotFoundError(f"No training .npz files found at {path}")
        return paths
    paths, _ = import_ocbench().download_datasets(
        config.dataset_name,
        dataset_root=config.ocbench_dataset_dir,
        num_shards=config.ocbench_num_shards,
    )
    return paths


class ActionTransform:
    """Map bounded policy actions to raw OCBench actions and back."""

    def __init__(self, center, scale):
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, actions, mode="bounds", percentile=1.0, clip=5.0):
        if mode == "bounds":
            return cls(np.zeros(actions.shape[-1]), np.ones(actions.shape[-1]))
        if mode != "percentile":
            raise ValueError("ocbench_action_normalization must be bounds or percentile")
        if not 0 <= percentile < 50 or clip <= 0:
            raise ValueError("Require 0 <= action percentile < 50 and action clip > 0")
        low = np.empty(actions.shape[-1], dtype=np.float32)
        high = np.empty_like(low)
        for dimension in range(actions.shape[-1]):
            low[dimension], high[dimension] = np.percentile(
                actions[:, dimension], [percentile, 100 - percentile]
            )
        center = (low + high) / 2
        scale = (high - low) / 2 * clip
        center[-1], scale[-1] = 0, 1
        return cls(center, scale)

    def normalize(self, actions):
        values = np.divide(
            np.asarray(actions, dtype=np.float32) - self.center,
            self.scale,
            out=np.zeros_like(actions, dtype=np.float32),
            where=self.scale > 0,
        )
        return np.clip(values, -1, 1)

    def unnormalize(self, actions):
        return np.clip(np.clip(actions, -1, 1) * self.scale + self.center, -1, 1)


class ReplayBuffer:
    """Compact host replay; construct transitions only for sampled minibatches."""

    def __init__(self, dataset, config):
        required = {"observations", "actions", "rewards", "masks", "terminals", "observation_interval"}
        if required - dataset.keys():
            raise ValueError(f"Missing OCBench dataset fields: {sorted(required - dataset.keys())}")
        if int(dataset["observation_interval"]) != 1:
            raise ValueError("State-only training requires observation_interval=1.")
        self.observations = np.asarray(dataset["observations"], dtype=np.float32)
        self.actions = np.asarray(dataset["actions"], dtype=np.float32)
        self.rewards = np.asarray(dataset["rewards"], dtype=np.float32)
        self.masks = np.asarray(dataset["masks"], dtype=np.float32)
        terminals = np.asarray(dataset["terminals"])
        if self.observations.ndim != 2 or self.actions.ndim != 2 or self.actions.shape[1] != 7:
            raise ValueError("Expected vector observations and seven-dimensional OCBench actions.")
        size = len(self.actions)
        if size == 0 or any(values.shape != (size,) for values in (self.rewards, self.masks, terminals)):
            raise ValueError("Transition arrays must be nonempty with matching lengths.")
        for values in (self.observations, self.actions, self.rewards, self.masks, terminals):
            for start in range(0, len(values), 100_000):
                if not np.isfinite(values[start:start + 100_000]).all():
                    raise ValueError("Dataset contains nonfinite values.")
        if not np.isin(self.masks, [0, 1]).all() or not np.isin(terminals, [0, 1]).all():
            raise ValueError("Masks and terminals must be binary.")
        if np.any((self.masks == 0) & (terminals == 0)):
            raise ValueError("Environment termination must also end the dataset episode.")
        self.ends = np.flatnonzero(terminals) + 1
        self.starts = np.concatenate([[0], self.ends[:-1]])
        if not len(self.ends) or self.ends[-1] != size or len(self.observations) != size + len(self.ends):
            raise ValueError("Expected T+1 observations and T transitions per complete episode.")
        self.action_transform = ActionTransform.fit(
            self.actions, config.ocbench_action_normalization,
            config.ocbench_action_percentile, config.ocbench_action_clip,
        )
        keep = np.ones(len(self.ends), dtype=bool)
        if config.ocbench_success_only:
            if config.dataset_name.startswith("bowling-"):
                raise ValueError("Bowling uses fractional rewards; set ocbench_success_only=false.")
            keep = np.maximum.reduceat(self.rewards, self.starts) >= 1
        self.episodes = np.flatnonzero(keep)
        if not len(self.episodes):
            raise ValueError("No episodes remain after success filtering.")
        self.lengths = self.ends[self.episodes] - self.starts[self.episodes]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.size = int(self.cumulative_lengths[-1])
        self.config = config
        self.rng = np.random.default_rng(config.train_seed)
        self.mean, self.std = np.float32(0), np.float32(1)
        if config.normalize_states:
            total = np.zeros(self.observations.shape[-1], dtype=np.float64)
            squares = np.zeros_like(total)
            for episode in self.episodes:
                start = self.starts[episode] + episode
                end = self.ends[episode] + episode
                for offset in range(start, end, 100_000):
                    values = self.observations[offset:min(offset + 100_000, end)].astype(np.float64)
                    total += values.sum(0)
                    squares += np.square(values).sum(0)
            self.mean = (total / self.size).astype(np.float32)
            self.std = (np.sqrt(np.maximum(squares / self.size - self.mean.astype(np.float64) ** 2, 0)) + 1e-3).astype(np.float32)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Replay indices out of bounds")
        selected = np.searchsorted(self.cumulative_lengths, indices, side="right")
        episode = self.episodes[selected]
        previous_end = np.where(selected > 0, self.cumulative_lengths[np.maximum(selected - 1, 0)], 0)
        transitions = self.starts[episode] + indices - previous_end
        observations = transitions + episode
        previous_valid = transitions > self.starts[episode]
        horizon = np.minimum(self.config.n_step, self.ends[episode] - transitions)
        last = transitions + horizon - 1
        last_observations = last + episode

        def states(positions):
            return (self.observations[positions] - self.mean) / self.std

        def actions(positions):
            return self.action_transform.normalize(self.actions[positions])

        def next_actions(positions):
            valid = positions + 1 < self.ends[episode]
            return np.where(valid[..., None], actions(np.minimum(positions + 1, len(self.actions) - 1)), 0)

        accumulated = np.zeros(indices.shape, dtype=np.float32)
        for offset in range(self.config.n_step):
            positions = np.minimum(transitions + offset, last)
            accumulated += (offset < horizon) * self.config.gamma ** offset * self.rewards[positions]
        return {
            "states": states(observations),
            "actions": actions(transitions),
            "prev_states": np.where(previous_valid[..., None], states(np.maximum(observations - 1, 0)), 0),
            "prev_actions": np.where(previous_valid[..., None], actions(np.maximum(transitions - 1, 0)), 0),
            "prev_valid": previous_valid.astype(np.float32),
            "next_states": states(observations + 1),
            "next_actions": next_actions(transitions),
            "rewards": self.rewards[transitions],
            "dones": 1 - self.masks[transitions],
            "n_step_states": states(last_observations + 1),
            "n_step_actions": next_actions(last),
            "n_step_prev_states": states(last_observations) * self.masks[last][..., None],
            "n_step_prev_actions": actions(last) * self.masks[last][..., None],
            "n_step_rewards": accumulated,
            "n_step_dones": 1 - self.masks[last],
            "n_step_horizon": horizon.astype(np.float32),
        }

    def sample(self, updates, batch_size):
        return self.batch(self.rng.integers(self.size, size=(updates, batch_size)))


def load_replay(config):
    paths = training_paths(config)
    print(f"Loading {len(paths)} training shards into host RAM; validation data is not used.", flush=True)
    dataset = import_ocbench().load_dataset(paths)
    replay = ReplayBuffer(dataset, config)
    print(f"Replay: {replay.size:,} transitions / {len(replay.episodes):,} episodes after filtering", flush=True)
    return replay, paths


def evaluate_policy(env, policy, transform, config, mean=0.0, std=1.0):
    """Evaluate on fixed seeds, excluding steps after each world's first done."""
    _, family, backend, horizon = environment_spec(config.dataset_name, config.ocbench_eval_backend)
    rng = np.random.default_rng(config.eval_seed)
    seeds = rng.integers(0, 2 ** 32, size=config.eval_episodes, dtype=np.uint32)
    returns, successes, lengths = [], [], []
    width = min(config.ocbench_eval_batch_size, config.eval_episodes) if backend == "mjwarp" else 1
    for start in range(0, config.eval_episodes, width):
        count = min(width, config.eval_episodes - start)
        reset_seeds = seeds[start:start + count]
        if backend == "mjwarp":
            reset_seeds = np.pad(reset_seeds, (0, width - count), mode="edge")
            observations, _ = env.reset(seeds=reset_seeds)
        else:
            observations, _ = env.reset(seed=int(reset_seeds[0]))
            observations = observations[None]
        observations = (np.asarray(observations, dtype=np.float32) - mean) / std
        previous_states = np.zeros_like(observations)
        previous_actions = np.zeros((width, 7), dtype=np.float32)
        done = np.arange(width) >= count
        episode_returns = np.zeros(width, dtype=np.float64)
        episode_successes = np.zeros(width, dtype=np.float32)
        episode_lengths = np.zeros(width, dtype=np.int32)
        for step in range(horizon):
            active = ~done
            normalized_actions = np.asarray(policy(observations, previous_states, previous_actions), dtype=np.float32)
            if normalized_actions.shape != (width, 7) or not np.isfinite(normalized_actions).all():
                raise ValueError("Policy must return finite actions with shape (batch, 7).")
            raw_actions = transform.unnormalize(normalized_actions)
            raw_actions[done] = 0
            next_observations, rewards, terminated, truncated, info = env.step(
                raw_actions if backend == "mjwarp" else raw_actions[0]
            )
            if backend == "cpu":
                next_observations = next_observations[None]
            rewards = np.asarray(rewards).reshape(width)
            episode_returns[active] += rewards[active]
            episode_lengths[active] += 1
            success = np.asarray(info.get("success", np.zeros(width))).reshape(width)
            episode_successes[active] = np.maximum(episode_successes[active], success[active])
            done |= np.asarray(terminated).reshape(width) | np.asarray(truncated).reshape(width) | (step + 1 >= horizon)
            previous_states = observations
            previous_actions = normalized_actions
            observations = (np.asarray(next_observations, dtype=np.float32) - mean) / std
            if done.all():
                break
        returns.extend(episode_returns[:count])
        successes.extend(episode_successes[:count])
        lengths.extend(episode_lengths[:count])
    returns = np.asarray(returns, dtype=np.float64)
    successes = np.asarray(successes, dtype=np.float64)
    lengths = np.asarray(lengths)
    scores = (returns if family == "bowling" else successes) * 100
    return {
        "return_mean": float(returns.mean()), "return_std": float(returns.std()),
        "success_mean": float(successes.mean() * 100), "success_std": float(successes.std() * 100),
        "normalized_score_mean": float(scores.mean()), "normalized_score_std": float(scores.std()),
        "mean_steps": float(lengths.mean()),
        "throughput": float(successes.mean() * horizon / lengths.mean()),
        "episodes": len(returns),
    }

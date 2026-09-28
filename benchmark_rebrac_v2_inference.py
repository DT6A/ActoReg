"""Benchmark randomly initialized ReBRAC-v2 inference, without training or datasets."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_CONFIGS = (
    "antmaze/large_navigate_singletask_v0.yaml", "antmaze/giant_navigate_singletask_v0.yaml",
    "humanoidmaze/medium_navigate_singletask_v0.yaml", "humanoidmaze/large_navigate_singletask_v0.yaml",
    "antsoccer/arena_navigate_singletask_v0.yaml", "cube/single_play_singletask_v0.yaml",
    "cube/double_play_singletask_v0.yaml", "scene/play_singletask_v0.yaml",
    "puzzle/3x3_play_singletask_v0.yaml", "puzzle/4x4_play_singletask_v0.yaml",
)


def parse_settings(text):
    settings = []
    for item in text.split(","):
        try:
            samples, steps = (int(value) for value in item.split(":"))
        except ValueError as error:
            raise argparse.ArgumentTypeError("Use K:J pairs, e.g. 1:0,32:0,32:2") from error
        if samples < 1 or steps < 0:
            raise argparse.ArgumentTypeError("K must be positive and J nonnegative")
        if (samples, steps) not in settings:
            settings.append((samples, steps))
    return settings


def config_paths(args):
    if args.config:
        return [Path(path).resolve() for path in args.config]
    base = ROOT / "configs/offline/rebrac-v2-ogbench"
    if not base.exists():
        base = ROOT / "configs/offline/rebrac-plus-ogbench"
    return sorted(base.rglob("*.yaml")) if args.all_ogbench else [base / name for name in MAIN_CONFIGS]


def load_config(path, model):
    import yaml

    values = yaml.safe_load(path.read_text())
    known = {field.name for field in fields(model.Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown configuration fields in {path}: {sorted(unknown)}")
    config = model.Config(**values)
    if not config.use_nf or config.nf_flow_type != "affine":
        raise ValueError("This benchmark targets the released affine-flow ReBRAC-v2 recipe")
    if config.use_prev_state or config.use_prev_action:
        raise ValueError("Previous-state/action ablations are not supported by this benchmark")
    if config.nf_eval_select not in {"auto", "q"}:
        raise ValueError("Use the released Q-based candidate selection")
    return config


def discover_dimensions(config, overrides):
    name = config.dataset_name
    if name in overrides:
        dims = overrides[name]
        observation_dim = int(dims["observation_dim"])
        action_dim = int(dims["action_dim"])
        goal_dim = int(dims.get("goal_dim", 0))
        source = "explicit dimensions JSON"
    else:
        if "-singletask-" not in name:
            raise ValueError("For non-singletask environments, supply --dimensions-json including goal_dim")
        import ogbench

        env = ogbench.make_env_and_datasets(name, env_only=True)
        try:
            if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
                raise ValueError("Only vector observations and continuous vector actions are supported")
            observation_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
            goal_dim = 0
        finally:
            env.close()
        source = "ogbench.make_env_and_datasets(env_only=True)"
    if observation_dim < 1 or action_dim < 1 or goal_dim < 0:
        raise ValueError(f"Invalid dimensions for {name}")
    append_goal = config.ogbench_append_goal and "-singletask-" not in name
    if append_goal and goal_dim == 0:
        raise ValueError("The configured goal-conditioned input requires an explicit goal_dim")
    return dict(observation_dim=observation_dim, action_dim=action_dim,
                goal_dim=goal_dim if append_goal else 0,
                state_dim=observation_dim + (goal_dim if append_goal else 0), dimension_source=source)


def initialize_models(model, config, dims, seed):
    import jax
    import jax.numpy as jnp

    actor = model.NFActorFlat(
        action_dim=dims["action_dim"], hidden_dim=config.nf_hidden_dim,
        n_hiddens=config.nf_n_hiddens, num_layers=config.nf_num_layers,
        scale_max=config.nf_scale_max, base_dist=config.nf_base_dist,
        use_plu=config.nf_use_plu, use_layernorm=config.nf_use_layernorm,
        dropout_rate=config.nf_dropout, deterministic_layers=config.nf_det_layers,
        activation=config.activation,
    )
    critic = model.EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=config.num_critics,
        layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens,
        n_classes=config.n_classes if config.use_distributional else 1,
        use_distributional=config.use_distributional, dropout_rate=config.critic_dropout,
        activation=config.activation, residual=config.critic_residual,
    )
    actor_key, critic_key = jax.random.split(jax.random.PRNGKey(seed))
    states = jnp.zeros((1, dims["state_dim"]), dtype=jnp.float32)
    actions = jnp.zeros((1, dims["action_dim"]), dtype=jnp.float32)
    actor_vars = actor.init({"params": actor_key, "mask": actor_key}, states,
                            rng=actor_key, train=False, method=model.nf_actor_sample)
    critic_vars = critic.init(critic_key, states, actions)
    jax.block_until_ready((actor_vars, critic_vars))
    return actor, critic, actor_vars, critic_vars


def make_runners(model, config, actor, critic, samples, steps, support):
    import jax
    import jax.numpy as jnp
    import numpy as np

    if steps and config.q_infer_step_size <= 0:
        raise ValueError("Positive refinement steps require a positive q_infer_step_size")

    @jax.jit
    def sample(actor_vars, states, key):
        if samples == 1:
            actions = actor.apply(actor_vars, states, rng=key, train=False, method=model.nf_actor_sample)
            return actions[:, None, :]
        return actor.apply(actor_vars, states, rng=key, num_samples=samples,
                           z_scale=config.nf_eval_z_scale, z_clip=config.nf_eval_z_clip,
                           train=False, method=model.nf_actor_sample_n)

    @jax.jit
    def values(critic_vars, states, candidates):
        repeated = jnp.repeat(states, samples, axis=0)
        actions = candidates.reshape(-1, candidates.shape[-1])
        logits = critic.apply(critic_vars, repeated, actions, train=False)
        if config.use_distributional:
            scores = model.transform_from_probs(jax.nn.softmax(logits, axis=-1), support).min(axis=0)
        else:
            scores = logits.squeeze(-1).min(axis=0)
        return scores.reshape(states.shape[0], samples)

    @jax.jit
    def refine(critic_vars, states, candidates):
        def objective(actions):
            return values(critic_vars, states, actions).sum()

        def update(step, actions):
            del step
            gradients = jax.grad(objective)(actions)
            norms = jnp.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8
            return jnp.clip(actions + config.q_infer_step_size * gradients / norms, -1.0, 1.0)

        return jax.lax.fori_loop(0, steps, update, candidates)

    @jax.jit
    def fused(actor_vars, critic_vars, states, key):
        next_key, action_key = jax.random.split(key)
        candidates = sample(actor_vars, states, action_key)
        if steps:
            candidates = refine(critic_vars, states, candidates)
        if samples > 1:
            indices = jnp.argmax(values(critic_vars, states, candidates), axis=1)
            actions = candidates[jnp.arange(states.shape[0]), indices]
        else:
            actions = candidates[:, 0]
        return next_key, jnp.clip(actions, -1.0, 1.0)

    def staged_host(actor_vars, critic_vars, states, key):
        states = jnp.asarray(states, dtype=jnp.float32)
        next_key, action_key = jax.random.split(key)
        host_candidates = np.asarray(jax.device_get(sample(actor_vars, states, action_key)))
        if samples == 1 and steps == 0:
            return next_key, np.clip(host_candidates[:, 0], -1.0, 1.0)
        candidates = jnp.asarray(host_candidates)
        if steps:
            candidates = refine(critic_vars, states, candidates)
        if samples > 1:
            indices = np.asarray(jax.device_get(jnp.argmax(values(critic_vars, states, candidates), axis=1)))
            actions = candidates[jnp.arange(states.shape[0]), indices]
        else:
            actions = candidates[:, 0]
        return next_key, np.asarray(jax.device_get(jnp.clip(actions, -1.0, 1.0)))

    return {"staged_host": staged_host, "fused_device": fused,
            "sample": sample, "values": values, "refine": refine}


def measure(runner, actor_vars, critic_vars, inputs, key, warmup, repeats, iterations):
    import jax
    import numpy as np

    started = time.perf_counter_ns()
    key, output = runner(actor_vars, critic_vars, inputs[0], key)
    jax.block_until_ready((key, output))
    first_call_ms = (time.perf_counter_ns() - started) / 1e6
    for index in range(warmup):
        key, output = runner(actor_vars, critic_vars, inputs[index % len(inputs)], key)
        jax.block_until_ready((key, output))
    timings = []
    for repeat in range(repeats):
        for iteration in range(iterations):
            states = inputs[iteration % len(inputs)]
            started = time.perf_counter_ns()
            key, output = runner(actor_vars, critic_vars, states, key)
            jax.block_until_ready((key, output))
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            timings.append(dict(repeat=repeat, iteration=iteration, latency_ms=elapsed_ms))
    actions = np.asarray(jax.device_get(output))
    if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.000001:
        raise ValueError("Nonfinite or out-of-bounds inference output")
    latencies = np.array([row["latency_ms"] for row in timings])
    summary = dict(mean_ms=float(np.mean(latencies)), median_ms=float(np.median(latencies)),
                   p95_ms=float(np.percentile(latencies, 95)), std_ms=float(np.std(latencies)),
                   first_call_ms=first_call_ms, timed_calls=len(timings),
                   actions_per_second=1000.0 * actions.shape[0] / float(np.mean(latencies)))
    return summary, timings


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", help="Specific YAML config(s); defaults to the 10 common OGBench categories")
    parser.add_argument("--all-ogbench", action="store_true", help="Benchmark every released OGBench config separately")
    parser.add_argument("--dimensions-json", type=Path, help="Optional dataset-name to observation_dim/action_dim/goal_dim mapping")
    parser.add_argument("--settings", type=parse_settings, default=parse_settings("1:0,32:0,32:2"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--modes", nargs="+", choices=["staged_host", "fused_device"], default=["staged_host", "fused_device"])
    parser.add_argument("--platform", choices=["gpu", "cpu", "tpu"], default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-envs", action="store_true", help="Only resolve and save dimensions/configuration; do not initialize JAX devices or models")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_benchmark"))
    args = parser.parse_args()
    if args.config and args.all_ogbench:
        parser.error("Use --config or --all-ogbench, not both")
    if min(args.batch_sizes + [args.iterations, args.repeats, args.warmup]) < 1 or args.device_index < 0:
        parser.error("Batch sizes, warmup, iterations, and repeats must be positive; device index must be nonnegative")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "disable")
    import jax
    import jax.numpy as jnp
    import numpy as np

    module_name = "algorithms.rebrac_v2_ogbench"
    if importlib.util.find_spec(module_name) is None:
        module_name = "algorithms.rebrac_cl_plus_ogbench"
    model = importlib.import_module(module_name)
    overrides = json.loads(args.dimensions_json.read_text()) if args.dimensions_json else {}
    paths = config_paths(args)
    if not paths:
        raise ValueError("No configurations found")
    environments = []
    for path in paths:
        config = load_config(path, model)
        dims = discover_dimensions(config, overrides)
        environments.append((config, dims, path))
        print(f"{config.dataset_name}: observation={dims['observation_dim']}, "
              f"goal={dims['goal_dim']}, network_input={dims['state_dim']}, action={dims['action_dim']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = [dict(env_name=config.dataset_name, **dims, config_path=str(path),
                         config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         config=asdict(config)) for config, dims, path in environments]
    (args.output_dir / "environments.json").write_text(json.dumps(descriptions, indent=2, default=str) + "\n")
    if args.list_envs:
        return
    devices = jax.devices(args.platform)
    device = devices[args.device_index]
    versions = {}
    for package in ["jax", "jaxlib", "flax", "numpy", "ogbench"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    system=platform.platform(), machine=platform.machine(), packages=versions,
                    device=str(device), device_kind=device.device_kind, platform=device.platform,
                    model_module=module_name, model_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
                    actor_sha256=hashlib.sha256(Path(importlib.import_module(model.NFActorFlat.__module__).__file__).read_bytes()).hexdigest(),
                    arguments=vars(args), random_initialization=True, dtype="float32",
                    support="synthetic linspace(-1, 1, n_classes+1); no dataset-derived value bounds",
                    state_distribution="standard normal synthetic vectors, fixed pool of 8 batches",
                    timing_scope="No training, optimizer, dataset loading, simulator stepping, or normalization. Warmup excluded; every call synchronized.")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    summaries, raw_timings = [], []
    with jax.default_device(device):
        for env_index, (config, dims, path) in enumerate(environments):
            actor, critic, actor_vars, critic_vars = initialize_models(model, config, dims, args.seed)
            support = jnp.linspace(-1.0, 1.0, config.n_classes + 1, dtype=jnp.float32)
            parameter_counts = dict(actor_parameters=sum(leaf.size for leaf in jax.tree.leaves(actor_vars["params"])),
                                    critic_parameters=sum(leaf.size for leaf in jax.tree.leaves(critic_vars["params"])))
            for batch_size in args.batch_sizes:
                generator = np.random.default_rng(args.seed + env_index)
                host_inputs = [generator.standard_normal((batch_size, dims["state_dim"])).astype(np.float32) for _ in range(8)]
                device_inputs = [jax.device_put(states, device) for states in host_inputs]
                jax.block_until_ready(device_inputs)
                cases = [(setting, mode) for setting in args.settings for mode in args.modes]
                generator.shuffle(cases)
                for (samples, steps), mode in cases:
                    runners = make_runners(model, config, actor, critic, samples, steps, support)
                    inputs = host_inputs if mode == "staged_host" else device_inputs
                    summary, timings = measure(runners[mode], actor_vars, critic_vars, inputs,
                                               jax.random.PRNGKey(args.seed + 1), args.warmup,
                                               args.repeats, args.iterations)
                    identity = dict(env_name=config.dataset_name, batch_size=batch_size,
                                    samples=samples, refinement_steps=steps, mode=mode)
                    summaries.append(dict(**identity, **dims, **parameter_counts,
                                          refinement_step_size=config.q_infer_step_size,
                                          device_kind=device.device_kind, **summary))
                    raw_timings.extend(dict(**identity, **row) for row in timings)
                    print(f"{config.dataset_name} B={batch_size} K={samples} J={steps} {mode}: "
                          f"median={summary['median_ms']:.3f}ms p95={summary['p95_ms']:.3f}ms "
                          f"throughput={summary['actions_per_second']:.1f} actions/s", flush=True)
                    write_csv(args.output_dir / "latency.csv", summaries)
                    write_csv(args.output_dir / "timings.csv", raw_timings)
            del actor_vars, critic_vars, runners, actor, critic, host_inputs, device_inputs, inputs
            jax.clear_caches()
    baselines = {(row["env_name"], row["batch_size"], row["mode"]): row["median_ms"]
                 for row in summaries if (row["samples"], row["refinement_steps"]) == (1, 0)}
    for row in summaries:
        baseline = baselines.get((row["env_name"], row["batch_size"], row["mode"]))
        row["median_overhead_ms_vs_actor_only"] = row["median_ms"] - baseline if baseline else ""
        row["median_ratio_vs_actor_only"] = row["median_ms"] / baseline if baseline else ""
    write_csv(args.output_dir / "latency.csv", summaries)


if __name__ == "__main__":
    main()

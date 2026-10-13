"""ReBRAC-v2 on state-based OCBench, sharing the OGBench model/update code."""

import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "disable")

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from tqdm.auto import trange

from algorithms import rebrac_cl_plus_ogbench as core
from algorithms.ocbench_utils import environment_spec, evaluate_policy, import_ocbench, load_replay


@dataclass
class Config(core.Config):
    name: str = "rebrac-v2-ocbench"
    group: str = "rebrac-v2-ocbench"
    dataset_name: str = "block-single-task1-v0"
    ocbench_dataset_dir: str = "~/.cache/ocbench/datasets"
    ocbench_dataset_path: Optional[str] = None
    ocbench_num_shards: Optional[int] = None
    ocbench_success_only: bool = True
    ocbench_action_normalization: str = "bounds"
    ocbench_action_percentile: float = 1.0
    ocbench_action_clip: float = 5.0
    ocbench_eval_backend: str = "native"
    ocbench_eval_batch_size: int = 50
    updates_per_chunk: int = 10
    eval_checkpoints: str = ""
    output_dir: str = "ocbench_runs"
    save_final_checkpoint: bool = False
    wandb_mode: str = "online"
    v_min: float = 0.0
    v_max: float = 1.0
    eval_every: int = 1000


def evaluation_updates(config):
    total = config.num_epochs * config.num_updates_on_epoch
    if config.track_checkpoint_average and config.eval_checkpoints:
        raise ValueError("Use eval_checkpoints or track_checkpoint_average, not both.")
    if config.track_checkpoint_average:
        requested = core.CHECKPOINT_AVERAGE_UPDATES
    else:
        requested = tuple(int(value.strip()) for value in config.eval_checkpoints.split(",") if value.strip())
    if requested and (len(set(requested)) != len(requested) or any(
        value <= 0 or value > total or value % config.num_updates_on_epoch for value in requested
    )):
        raise ValueError("Evaluation checkpoints must be distinct epoch-aligned updates within the training budget.")
    return set(requested)


def validate_config(config):
    environment_spec(config.dataset_name, config.ocbench_eval_backend)
    for name in ("num_epochs", "num_updates_on_epoch", "updates_per_chunk", "batch_size", "policy_freq",
                 "eval_episodes", "eval_every", "ocbench_eval_batch_size", "n_step", "target_action_k", "num_critics"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.ocbench_num_shards is not None and config.ocbench_num_shards < 1:
        raise ValueError("ocbench_num_shards must be positive or null")
    if config.il_warmup_epochs < 0 or config.critic_warmup_epochs < 0 or (
        config.il_warmup_epochs + config.critic_warmup_epochs > config.num_epochs
    ):
        raise ValueError("Warmups must be nonnegative and fit within num_epochs.")
    if not 0 < config.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not np.isfinite([config.v_min, config.v_max]).all() or config.v_min >= config.v_max:
        raise ValueError("Provide finite v_min < v_max; OCBench defaults to [0, 1].")
    if config.actor_bc_aux_loss not in {"mse", "mae", "sum"}:
        raise ValueError("actor_bc_aux_loss must be mse, mae, or sum")
    if config.target_critic_aggregation not in {"min", "mean"}:
        raise ValueError("target_critic_aggregation must be min or mean, as in the shared OGBench implementation")
    if config.target_action_reduction not in {"min", "mean", "max"}:
        raise ValueError("target_action_reduction must be min, mean, or max")
    if config.critic_bc_coef != 0:
        raise ValueError("OCBench requires critic_bc_coef=0: no demonstrated next action exists at truncation.")
    unsupported = ("use_iql", "actor_reset", "num_refinement_epochs", "use_likelihood_alpha_target",
                   "normalize_reward", "noisy_eval")
    for name in unsupported:
        if getattr(config, name):
            raise ValueError(f"{name} is a legacy OGBench option not supported by this trainer.")
    if not config.actor_prereset_mode:
        raise ValueError("actor_prereset_mode=false requires the unsupported actor-reset experiment.")
    if config.critic_next_state_pred_epochs < 0 or config.q_infer_step_size < 0:
        raise ValueError("Next-state prediction duration and inference step size must be nonnegative.")
    core.resolve_activation(config.activation)
    core.parse_eval_num_samples(config.nf_eval_num_samples)
    core.parse_q_infer_steps(config.q_infer_steps)
    selections = core.parse_nf_eval_select(config.nf_eval_select)
    if not config.use_nf and any(selection in {"likelihood", "mix"} for selection in selections):
        raise ValueError("Likelihood selection requires use_nf=true.")
    if config.ocbench_action_normalization not in {"bounds", "percentile"}:
        raise ValueError("ocbench_action_normalization must be bounds or percentile")
    if not 0 <= config.ocbench_action_percentile < 50 or config.ocbench_action_clip <= 0:
        raise ValueError("Invalid percentile action-normalization settings")
    evaluation_updates(config)


def initialize_models(config, batch):
    key, actor_key, critic_key, dropout_key = jax.random.split(jax.random.PRNGKey(config.train_seed), 4)
    states, actions = batch["states"], batch["actions"]
    actor_inputs = core.build_actor_inputs(states, batch["prev_states"], batch["prev_actions"],
                                          config.use_prev_state, config.use_prev_action)
    if config.use_nf:
        if config.nf_flow_type not in {"affine", "nsf"}:
            raise ValueError("nf_flow_type must be affine or nsf")
        actor_class = core.NFActorFlat if config.nf_flow_type == "affine" else core.NSFActorFlat
        extra = {} if config.nf_flow_type == "affine" else {
            "num_bins": config.nf_num_bins, "tail_bound": config.nf_tail_bound,
        }
        actor_module = actor_class(
            action_dim=actions.shape[-1], hidden_dim=config.nf_hidden_dim, n_hiddens=config.nf_n_hiddens,
            num_layers=config.nf_num_layers, scale_max=config.nf_scale_max, base_dist=config.nf_base_dist,
            use_plu=config.nf_use_plu, use_layernorm=config.nf_use_layernorm, dropout_rate=config.nf_dropout,
            deterministic_layers=config.nf_det_layers, activation=config.activation, **extra,
        )
        actor_variables = actor_module.init({"params": actor_key, "mask": actor_key}, actor_inputs,
                                            rng=actor_key, train=False, method=core.nf_actor_sample)
    else:
        actor_module = core.DetActor(
            action_dim=actions.shape[-1], hidden_dim=config.hidden_dim, layernorm=config.actor_ln,
            featurenorm=config.actor_fn, groupnorm=config.actor_gn, batchnorm=config.actor_bn,
            spectralnorm=config.actor_sn, dropout_rate=config.actor_dropout,
            n_hiddens=config.actor_n_hiddens, activation=config.activation,
        )
        actor_variables = actor_module.init(actor_key, actor_inputs, False)
    total = config.num_epochs * config.num_updates_on_epoch
    actor_lr = config.actor_learning_rate
    if config.decay_schedule == "cosine":
        actor_lr = optax.cosine_decay_schedule(actor_lr, total)
    elif config.decay_schedule == "linear":
        actor_lr = optax.linear_schedule(actor_lr, actor_lr / 10, total)
    elif config.decay_schedule == "exp":
        actor_lr = optax.exponential_decay(actor_lr, total, 0.99)
    elif config.decay_schedule is not None:
        raise ValueError("decay_schedule must be null, cosine, linear, or exp")
    if config.optimizer_type == "kron":
        from algorithms.kron import kron
        actor_optimizer = kron(learning_rate=actor_lr, weight_decay=config.actor_wd)
        critic_optimizer = kron(learning_rate=config.critic_learning_rate, weight_decay=config.critic_wd)
    elif config.optimizer_type == "adan":
        actor_optimizer = optax.adan(learning_rate=actor_lr, weight_decay=config.actor_wd)
        critic_optimizer = optax.adan(learning_rate=config.critic_learning_rate, weight_decay=config.critic_wd)
    elif config.optimizer_type in {"adam", "adamw"}:
        actor_optimizer = core.adamw_elastic(actor_lr, weight_decay=config.actor_wd, l1_ratio=config.l1_ratio)
        critic_optimizer = (optax.adam(config.critic_learning_rate) if config.optimizer_type == "adam" else
                            optax.adamw(config.critic_learning_rate, weight_decay=config.critic_wd))
    else:
        raise ValueError("optimizer_type must be kron, adan, adam, or adamw")
    parameters = actor_variables["params"]
    statistics, constants = actor_variables.get("batch_stats", {}), actor_variables.get("constants", {})
    actor = core.ActorTrainState.create(
        apply_fn=actor_module.apply, params=parameters, target_params=parameters, ema_params=parameters,
        batch_stats=statistics, target_batch_stats=statistics, ema_batch_stats=statistics,
        constants=constants, target_constants=constants, ema_constants=constants,
        dropout_key=dropout_key, tx=actor_optimizer,
    )
    classes = config.n_classes if config.use_distributional else 1
    critic_module = core.EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=config.num_critics, layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens, n_classes=classes, use_distributional=config.use_distributional,
        dropout_rate=config.critic_dropout, activation=config.activation, residual=config.critic_residual,
    )
    lower, upper = config.v_min, config.v_max
    expand = (upper - lower) * config.v_expand
    if config.v_expand_mode == "both":
        lower, upper = lower - expand / 2, upper + expand / 2
    elif config.v_expand_mode == "min":
        lower -= expand
    elif config.v_expand_mode == "max":
        upper += expand
    else:
        raise ValueError("v_expand_mode must be both, min, or max")
    critic_variables = critic_module.init(critic_key, states, actions)
    critic = core.CriticTrainState.create(
        apply_fn=critic_module.apply, params=critic_variables, target_params=critic_variables,
        support=jnp.linspace(lower, upper, classes + 1), sigma=config.sigma_frac * (upper - lower) / classes,
        tx=critic_optimizer,
    )
    return key, actor, critic


METRICS = (
    "critic_loss", "critic_next_state_loss", "q_min", "actor_loss", "nll", "aux_bc",
    "actor_loss_bc_term", "actor_loss_aux_term", "actor_loss_q_term", "actor_loss_lmbda",
    "actor_loss_is_finite", "critic_loss_is_finite", "actor_grad_norm", "actor_grad_norm_after_noise",
    "critic_grad_norm", "critic_grad_norm_after_noise", "action_mse", "bc_mse_policy", "bc_mse_random",
)


def make_update(config, stage):
    critic_options = dict(
        gamma=config.gamma, critic_bc_coef=config.critic_bc_coef, tau=config.tau,
        policy_noise=config.policy_noise, noise_clip=config.noise_clip,
        critic_objective_noise=config.critic_objective_noise, critic_grad_noise=config.critic_grad_noise,
        use_prev_state=config.use_prev_state, use_prev_action=config.use_prev_action,
        use_target_actor=config.use_target_actor, use_nf=config.use_nf,
        use_distributional=config.use_distributional, target_critic_aggregation=config.target_critic_aggregation,
        next_state_pred_epochs=config.critic_next_state_pred_epochs, use_likelihood_alpha_target=False,
        likelihood_alpha_eps=config.likelihood_alpha_eps, likelihood_logprob_min=jnp.float32(0),
        likelihood_logprob_max=jnp.float32(1), target_action_k=config.target_action_k,
        target_action_reduction=config.target_action_reduction,
    )
    full_update = partial(
        core.update_td3, **critic_options, actor_bc_coef=config.actor_bc_coef,
        actor_bc_aux_weight=config.actor_bc_aux_weight, actor_bc_aux_loss=config.actor_bc_aux_loss,
        actor_ema_tau=config.actor_ema_tau, normalize_q=config.normalize_q,
        actor_input_noise=config.actor_input_noise, actor_bc_noise=config.actor_bc_noise,
        actor_grad_noise=config.actor_grad_noise,
    )
    critic_update = partial(core.update_critic_warmup, **critic_options)
    delayed_update = partial(core.update_td3_no_targets, **critic_options, actor_bc_coef=config.actor_bc_coef)
    bc_update = partial(
        core.update_actor_bc, beta=config.actor_bc_coef, aux_weight=config.actor_bc_aux_weight,
        aux_loss=config.actor_bc_aux_loss, tau=config.tau, ema_tau=config.actor_ema_tau,
        input_noise=config.actor_input_noise, bc_noise=config.actor_bc_noise, grad_noise=config.actor_grad_noise,
        use_prev_state=config.use_prev_state, use_prev_action=config.use_prev_action, use_nf=config.use_nf,
    )

    @jax.jit
    def update(carry, batches, epoch, offset):
        def body(current, inputs):
            batch, index = inputs
            key, actor, critic, metrics = current
            if stage == "il":
                key, actor, metrics = bc_update(key=key, actor=actor, batch=batch, metrics=metrics)
                return (key, actor, critic, metrics), None
            options = dict(key=key, actor=actor, critic=critic, metrics=metrics, batch=batch, epoch=epoch)
            if stage == "critic":
                return critic_update(**options), None
            result = jax.lax.cond((offset + index) % config.policy_freq == 0,
                                  lambda: full_update(**options), lambda: delayed_update(**options))
            return result, None
        indices = jnp.arange(batches["rewards"].shape[0])
        result, _ = jax.lax.scan(body, carry, (batches, indices))
        return result
    return update


def make_action_selector(config, actor, critic, samples, steps, selection):
    samples = samples if config.use_nf else 1

    @jax.jit
    def select(parameters, statistics, constants, critic_parameters, states, previous_states, previous_actions, key):
        actor_states = core.build_actor_inputs(states, previous_states, previous_actions,
                                               config.use_prev_state, config.use_prev_action)
        variables = {"params": parameters, "constants": constants}
        if config.use_nf:
            candidates = actor.apply_fn(variables, actor_states, rng=key, num_samples=samples,
                                        z_scale=config.nf_eval_z_scale, z_clip=config.nf_eval_z_clip,
                                        train=False, method=core.nf_actor_sample_n)
        else:
            candidates = actor.apply_fn({"params": parameters, "batch_stats": statistics}, actor_states, False)[0][:, None]
        repeated_states = jnp.repeat(states, samples, axis=0)
        flat_actions = candidates.reshape(-1, candidates.shape[-1])

        def scores(actions):
            logits = critic.apply_fn(critic_parameters, repeated_states, actions, False)
            if config.use_distributional:
                return core.transform_from_probs(jax.nn.softmax(logits, axis=-1), critic.support).min(0)
            return jnp.squeeze(logits, -1).min(0)

        def refine(index, actions):
            gradients = jax.grad(lambda values: scores(values).sum())(actions)
            gradients /= jnp.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8
            return jnp.clip(actions + config.q_infer_step_size * gradients, -1, 1)

        if steps > 0 and config.q_infer_step_size > 0:
            flat_actions = jax.lax.fori_loop(0, steps, refine, flat_actions)
        if samples == 1:
            return flat_actions
        if selection in {"likelihood", "mix"}:
            likelihoods = actor.apply_fn(variables, flat_actions, jnp.repeat(actor_states, samples, axis=0),
                                         train=False, method=core.nf_actor_log_prob).reshape(-1, samples)
            if selection == "likelihood":
                values = likelihoods
            else:
                values = scores(flat_actions).reshape(-1, samples) * (
                    likelihoods - likelihoods.min(axis=-1, keepdims=True) + 1e-8
                )
        else:
            values = scores(flat_actions).reshape(-1, samples)
        return flat_actions.reshape(candidates.shape)[jnp.arange(states.shape[0]), jnp.argmax(values, axis=-1)]
    return select


def save_checkpoint(path, actor, critic, replay, updates):
    payload = dict(
        actor=flax.serialization.to_state_dict(actor), critic=flax.serialization.to_state_dict(critic),
        observation_mean=replay.mean, observation_std=replay.std,
        action_center=replay.action_transform.center, action_scale=replay.action_transform.scale,
        updates=updates,
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(flax.serialization.to_bytes(payload))
    temporary.replace(path)


def create_eval_env(config, replay):
    name, _, backend, _ = environment_spec(config.dataset_name, config.ocbench_eval_backend)
    width = min(config.ocbench_eval_batch_size, config.eval_episodes) if backend == "mjwarp" else 1
    options = {"nworld": width} if backend == "mjwarp" else {}
    env = import_ocbench().make(name, **options)
    try:
        observations, _ = env.reset(seed=config.eval_seed)
        shape = np.asarray(observations).shape
        expected = (width, replay.observations.shape[-1]) if backend == "mjwarp" else (replay.observations.shape[-1],)
        if shape != expected:
            raise ValueError(f"Evaluation observations {shape} do not match replay {expected}.")
        action_space = env.single_action_space if backend == "mjwarp" else env.action_space
        if action_space.shape != (7,) or not np.allclose(action_space.low, -1) or not np.allclose(action_space.high, 1):
            raise ValueError("Expected OCBench's seven-dimensional [-1, 1] action space.")
    except BaseException:
        env.close()
        raise
    return env


def run_training(config):
    validate_config(config)
    if environment_spec(config.dataset_name, config.ocbench_eval_backend)[2] == "mjwarp":
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("Native OCBench evaluation requires CUDA. Use ocbench_eval_backend=cpu for a CPU smoke test.")
    replay, paths = load_replay(config)
    key, actor, critic = initialize_models(config, jax.device_put(replay.batch(np.array([0]))))
    eval_name, _, _, _ = environment_spec(config.dataset_name, config.ocbench_eval_backend)
    run_dir = Path(config.output_dir).expanduser() / config.name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.yaml").open("w") as output:
        pyrallis.dump(config, output)
    (run_dir / "dataset.json").write_text(json.dumps(dict(
        training_paths=paths, transitions=replay.size, episodes=len(replay.episodes),
        eval_environment=eval_name, devices=[str(device) for device in jax.devices()],
    ), indent=2) + "\n")
    print(f"Run output: {run_dir}", flush=True)
    eval_env = None
    run = wandb.init(project=config.project, group=config.group, name=config.name, config=asdict(config),
                     mode=config.wandb_mode, dir=str(run_dir))
    checkpoints = evaluation_updates(config)
    history = {}
    selectors = {}
    update_functions = {stage: make_update(config, stage) for stage in ("il", "critic", "rl")}
    sample_counts = core.parse_eval_num_samples(config.nf_eval_num_samples)
    refinement_steps = core.parse_q_infer_steps(config.q_infer_steps)
    requested_selections = core.parse_nf_eval_select(config.nf_eval_select)
    il_end = config.il_warmup_epochs
    critic_end = il_end + config.critic_warmup_epochs
    total_updates = config.num_epochs * config.num_updates_on_epoch
    try:
        eval_env = create_eval_env(config, replay)
        for epoch in trange(config.num_epochs, desc="ReBRAC-v2 OCBench"):
            stage = "il" if epoch < il_end else "critic" if epoch < critic_end else "rl"
            carry = (key, actor, critic, core.Metrics.create(METRICS))
            for offset in range(0, config.num_updates_on_epoch, config.updates_per_chunk):
                chunk_size = min(config.updates_per_chunk, config.num_updates_on_epoch - offset)
                batches = jax.device_put(replay.sample(chunk_size, config.batch_size))
                carry = update_functions[stage](carry, batches, jnp.int32(epoch), jnp.int32(offset))
                jax.block_until_ready(carry)
            key, actor, critic, metrics = carry
            updates = (epoch + 1) * config.num_updates_on_epoch
            metrics = {f"ReBRACv2/{name}": float(np.asarray(value).item()) for name, value in metrics.compute().items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite training metrics at update {updates}")
            metrics.update(epoch=epoch, updates=updates, stage=stage)
            should_evaluate = updates in checkpoints if checkpoints else (epoch + 1) % config.eval_every == 0
            if should_evaluate or updates == total_updates:
                selections = ["likelihood" if stage == "il" and config.use_nf else "q"] if requested_selections == ["auto"] else requested_selections
                if stage == "il" and config.use_nf:
                    selections = list(dict.fromkeys("likelihood" if value == "q" or value == "mix" else value for value in selections))
                parameters = actor.ema_params if config.use_actor_ema else actor.params
                statistics = actor.ema_batch_stats if config.use_actor_ema else actor.batch_stats
                constants = actor.ema_constants if config.use_actor_ema else actor.constants
                for samples in sample_counts:
                    for steps in refinement_steps:
                        for selection in selections:
                            effective_steps = 0 if stage == "il" else steps
                            selector_key = (samples, effective_steps, selection)
                            if selector_key not in selectors:
                                selectors[selector_key] = make_action_selector(config, actor, critic, *selector_key)
                            selector = selectors[selector_key]
                            eval_key = jax.random.PRNGKey(config.eval_seed)

                            def policy(states, previous_states, previous_actions):
                                nonlocal eval_key
                                eval_key, action_key = jax.random.split(eval_key)
                                return selector(parameters, statistics, constants, critic.params, states,
                                                previous_states, previous_actions, action_key)

                            suffix = (f"_ns_{samples}" if len(sample_counts) > 1 else "")
                            suffix += f"_qs_{steps}" if len(refinement_steps) > 1 else ""
                            suffix += f"_sel_{selection}" if len(selections) > 1 else ""
                            scores = evaluate_policy(eval_env, policy, replay.action_transform, config, replay.mean, replay.std)
                            metrics.update({f"eval/{name}{suffix}": value for name, value in scores.items()})
                            if updates in checkpoints:
                                for name, value in scores.items():
                                    if name.endswith("_mean"):
                                        history.setdefault(f"{name}{suffix}", []).append(value)
                            print(f"{updates}: {suffix or 'default'} score={scores['normalized_score_mean']:.2f}", flush=True)
                metrics.update({f"eval/checkpoint_average_{name}": float(np.mean(values)) for name, values in history.items()})
                if history:
                    metrics["eval/checkpoint_count"] = max(map(len, history.values()))
            run.log(metrics, step=updates)
            with (run_dir / "metrics.jsonl").open("a") as output:
                output.write(json.dumps(metrics, allow_nan=False) + "\n")
        if config.save_final_checkpoint:
            save_checkpoint(run_dir / "final.msgpack", actor, critic, replay, total_updates)
    finally:
        if eval_env is not None:
            eval_env.close()
        run.finish()
    return run_dir


@pyrallis.wrap()
def train(config: Config):
    run_training(config)


if __name__ == "__main__":
    train()

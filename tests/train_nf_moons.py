import argparse
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms.nf_policy import NFActor


def make_moons(n_samples: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    t_out = rng.rand(n_out) * np.pi
    t_in = rng.rand(n_in) * np.pi

    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    inner = np.stack([1.0 - np.cos(t_in), 1.0 - np.sin(t_in) - 0.5], axis=1)
    data = np.concatenate([outer, inner], axis=0)
    data += noise * rng.randn(*data.shape)

    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs
    data = np.clip(data, -1.0, 1.0)
    return data.astype(np.float32)


@dataclass
class TrainConfig:
    n_samples: int = 2048
    batch_size: int = 256
    steps: int = 2000
    lr: float = 1e-4
    noise: float = 0.05
    seed: int = 0
    log_every: int = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--base-dist", type=str, default="normal", choices=["normal", "uniform"])
    parser.add_argument("--plot-path", type=str, default="tests/moons_nf_samples.png")
    parser.add_argument("--plot-samples", type=int, default=2000)
    parser.add_argument("--use-transformer", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        noise=args.noise,
        seed=args.seed,
        log_every=args.log_every,
    )

    data = make_moons(cfg.n_samples, cfg.noise, cfg.seed)
    data = jnp.asarray(data)

    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)

    model = NFActor(
        action_dim=1,
        chunk_len=2,
        hidden_dim=64,
        n_hiddens=2,
        num_layers=6,
        use_transformer=args.use_transformer,
        scale_max=1.0,
        base_dist=args.base_dist,
        use_plu=True,
        use_layernorm=True,
        dropout_rate=0.0,
    )

    dummy_state = jnp.zeros((cfg.batch_size, 1), dtype=jnp.float32)
    dummy_prev = jnp.zeros((cfg.batch_size, 0, 1), dtype=jnp.float32)
    variables = model.init(
        {"params": init_key, "dropout": init_key, "mask": init_key},
        dummy_state,
        dummy_prev,
        rng=init_key,
        train=True,
        method=NFActor.sample,
    )
    params = variables["params"]
    constants = variables.get("constants", {})

    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch, rng):
        def loss_fn(p):
            logp = model.apply(
                {"params": p, "constants": constants},
                batch,
                dummy_state[: batch.shape[0]],
                dummy_prev[: batch.shape[0]],
                train=True,
                method=NFActor.log_prob,
                rngs={"dropout": rng},
            )
            return -jnp.mean(logp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(cfg.steps):
        key, batch_key, step_key = jax.random.split(key, 3)
        idx = jax.random.randint(batch_key, (cfg.batch_size,), 0, data.shape[0])
        batch = data[idx]
        batch = batch[:, :, None]  # [B, 2, 1]

        params, opt_state, loss = step(params, opt_state, batch, step_key)

        if (i + 1) % cfg.log_every == 0:
            print(f"step {i+1} | nll {float(loss):.4f}")

    key, sample_key = jax.random.split(key)
    sample_n = args.plot_samples
    sample_state = jnp.zeros((sample_n, 1), dtype=jnp.float32)
    sample_prev = jnp.zeros((sample_n, 0, 1), dtype=jnp.float32)
    samples = model.apply(
        {"params": params, "constants": constants},
        sample_state,
        sample_prev,
        rng=sample_key,
        train=False,
        method=NFActor.sample,
    )
    samples = np.asarray(samples[:, :, 0])

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
        axes[0].set_title("Data")
        axes[0].set_xlim(-1.1, 1.1)
        axes[0].set_ylim(-1.1, 1.1)
        axes[1].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        axes[1].set_title("NF Samples")
        axes[1].set_xlim(-1.1, 1.1)
        axes[1].set_ylim(-1.1, 1.1)
        fig.tight_layout()
        fig.savefig(args.plot_path, dpi=150)
        print(f"saved plot to {args.plot_path}")
    except Exception as exc:
        print(f"plotting skipped: {exc}")

    print("done")


if __name__ == "__main__":
    main()

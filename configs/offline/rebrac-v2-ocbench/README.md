# ReBRAC-v2 on OCBench (state observations)

## Scope

`algorithms/rebrac_v2_ocbench.py` reuses the networks, flow implementation, mixed BC loss, categorical critic updates, warmups, target updates, and actor EMA from `algorithms/rebrac_cl_plus_ogbench.py`. The existing OGBench trainer is unchanged. OCBench-specific dataset loading and evaluation live in `algorithms/ocbench_utils.py`.

There are **28 main benchmark tasks**, not 55 unrelated tasks. The **55 config files** cover those 28 full-horizon tasks and their 27 optional lite variants. Bowling has no lite variant. You do not need to run both suites unless you want to study both settings. CPU evaluation is an option, not another set of tasks. No image observations or visual datasets are used. Every configured task was checked against the upstream registry and the released state-dataset manifest.

These are **starting configurations for tuning, not tuned OCBench results**. Architecture, optimizer, regularizers, and warmup settings are transferred from the existing OGBench recipe. Initial BC weights are `actor_bc_aux_weight=0.03` and `actor_bc_coef=0.0003`; they are not task-specific selections.

**Current experiment set: 16 selected lite tasks.** Use `tools/download_ocbench_lite_datasets.py` and `configs/sweeps/rebrac_v2_ocbench_lite_bc_grid.yaml`. They cover exactly the same state-only tasks: Block Single/Double (3), Chamber Easy/Medium (5), Switch 3x3/4x4 (4), and Hanoi Single/Double (4). All task IDs from `block-lite-quadruple`, `chamber-lite-hard`, `block-lite-triple`, `hanoi-lite-triple`, and `switch-lite-5x5` are excluded from downloads and the lite sweep because of dataset size. Their config files remain available but inactive. Bowling has no lite version. Full-horizon configs are not part of this sweep.

## All environments

Replace `N` in each ID with one of the listed task numbers. This enumerates all 28 full-horizon task IDs across five families:

| Environment ID | Task numbers | Count |
| --- | --- | ---: |
| `block-single-taskN-v0` | 1 | 1 |
| `block-double-taskN-v0` | 1, 2 | 2 |
| `block-triple-taskN-v0` | 1, 2 | 2 |
| `block-quadruple-taskN-v0` | 1, 2, 3 | 3 |
| `chamber-easy-taskN-v0` | 1, 2, 3 | 3 |
| `chamber-medium-taskN-v0` | 1, 2 | 2 |
| `chamber-hard-taskN-v0` | 1, 2 | 2 |
| `switch-3x3-taskN-v0` | 1, 2 | 2 |
| `switch-4x4-taskN-v0` | 1, 2 | 2 |
| `switch-5x5-taskN-v0` | 1, 2 | 2 |
| `hanoi-single-taskN-v0` | 1, 2 | 2 |
| `hanoi-double-taskN-v0` | 1, 2 | 2 |
| `hanoi-triple-taskN-v0` | 1, 2 | 2 |
| `bowling-taskN-v0` | 1 | 1 |

The benchmark's 27 lite IDs are the same list except Bowling, with `-lite` inserted after the family, e.g. `block-lite-double-task2-v0` and `chamber-lite-easy-task3-v0`. Only the 16 selected lite IDs are active in this experiment. The lite-only downloader's `--list` prints those 16 exact IDs; the general downloader's `--suite all --list` prints 44 IDs (28 full-horizon plus the 16 selected lite tasks).

## Installation

Use the existing ReBRAC-v2 JAX environment on a Linux NVIDIA machine. Add OCBench and its GPU simulator:

```bash
python -m pip install -r requirements-ocbench.txt
```

The dependency file pins the inspected [OCBench repository](https://github.com/seohongpark/ocbench) at `6939a1e84f7ce9dbdd20c9f80d4cf335e8f83b49` (package version 1.0.1). Its dataset loader pins the release manifest to `fd771640f7a47879d8b06657fe05389641b5bff9`. The inspection clone in `tmp/ocbench-reference` is not a runtime dependency.

For CPU-only testing, install without the `mjwarp` extra:

```bash
python -m pip install "ocbench @ git+https://github.com/seohongpark/ocbench.git@6939a1e84f7ce9dbdd20c9f80d4cf335e8f83b49"
```

The trainer defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not reserve GPU memory needed by MJWarp, and `MUJOCO_GL=disable` because state-only rollouts need no renderer. Use the project's JAX/Flax environment, not the old root `requirements.txt` intended for the original experiments. Installing the additive OCBench requirements may update its simulator dependencies if your environment does not already satisfy them.

## Download datasets first

`tools/download_ocbench_datasets.py` uses the official OCBench downloader without initializing a simulator or loading datasets into RAM. Downloading requires only base OCBench and PyYAML, not CUDA/JAX/MJWarp. Task selection comes from the checked-in configs, and visual datasets are excluded.

From the repository root, preview the required space, then download:

```bash
python tools/download_ocbench_lite_datasets.py --dry-run
python tools/download_ocbench_lite_datasets.py
```

The lite-only entrypoint rejects full-horizon and excluded lite task IDs and does not accept a `--suite` override. It defaults to all shards for the 16 selected lite datasets. List the exact tasks or choose a destination:

```bash
python tools/download_ocbench_lite_datasets.py --list
python tools/download_ocbench_lite_datasets.py --dataset-dir /path/to/ocbench/datasets
```

The general-purpose `tools/download_ocbench_datasets.py` remains available with `--suite full|lite|all`, but is not needed for the lite-only experiment. It respects the same lite exclusions, including when tasks are requested explicitly.

The selected 16 lite datasets total approximately **34.56 GB**, down from **320.37 GB** for all 27 lite datasets, including all training shards, validation companions, and metadata (decimal units, excluding temporary/cache overhead). The separate full-horizon suite is still approximately 2.11 TB and is not selected by the lite-only script. `--dry-run` fetches only the small manifest if needed and reports the selected file sizes and existing same-size files. `--list` needs no network access or OCBench import.

Download a subset or a small pilot dataset:

```bash
python tools/download_ocbench_lite_datasets.py \
  --env-name block-lite-single-task1-v0 \
  --env-name chamber-lite-easy-task2-v0 \
  --num-shards 1
```

With `--num-shards 1`, also pass `--ocbench_num_shards=1` to training (or set it in the training config/sweep), otherwise training will download the remaining shards. The default destination is `~/.cache/ocbench/datasets`, matching the trainer. For another location, pass the same path as training's `--ocbench_dataset_dir`. Downloads reuse the official cache and can be rerun after an interruption. The official API includes validation and metadata companions; training still uses training shards only. No dataset download was launched when this script was added.

## Run a task

Run commands from the repository root:

```bash
PYTHONPATH=. python algorithms/rebrac_v2_ocbench.py \
  --config_path=configs/offline/rebrac-v2-ocbench/full/block/block-single-task1-v0.yaml \
  --train_seed=0
```

For a lite task:

```bash
PYTHONPATH=. python algorithms/rebrac_v2_ocbench.py \
  --config_path=configs/offline/rebrac-v2-ocbench/lite/chamber/chamber-lite-easy-task2-v0.yaml \
  --train_seed=0
```

Useful overrides:

| Flag | Meaning |
| --- | --- |
| `--ocbench_dataset_dir=/path/to/data` | Download/cache location. |
| `--ocbench_dataset_path=/path/to/shards` | Local `.npz` file or directory; `*-val.npz` is excluded. No downloads. |
| `--ocbench_num_shards=1` | Explicitly restrict the training data to the first shard. Defaults to all released shards. |
| `--ocbench_success_only=false` | Keep failed demonstrations too; changes the training-data protocol. |
| `--ocbench_eval_backend=cpu` | CPU MuJoCo evaluation instead of native GPU MJWarp. |
| `--ocbench_eval_batch_size=10` | Fewer parallel GPU rollouts to reduce evaluation VRAM. |
| `--eval_episodes=500` | More evaluation episodes, e.g. for final reporting. |
| `--updates_per_chunk=5` | Smaller minibatch transfer/JIT blocks if memory is tight. |
| `--save_final_checkpoint=true` | Opt in to saving the final actor/critic checkpoint; disabled by default. |
| `--wandb_mode=disabled` | Local-only smoke test; JSONL metrics are still saved. |

The default backend is **native MJWarp** and never silently falls back to CPU. CUDA availability is checked before downloading data, and the simulator is initialized and its dimensions checked before training. Although upstream shares dataset files between these IDs, CPU and GPU dynamics are not numerically identical. The actual evaluation ID is recorded in `dataset.json`; use the native backend for comparisons to GPU baselines.

**Memory:** replay stays in host RAM in OCBench's compact format; only sampled minibatches go to the accelerator. This avoids placing the whole dataset and duplicated next/history states in VRAM. Full-horizon, multi-shard datasets can still require substantial host RAM and disk. The loader does not silently subsample them. For a first smoke run, explicitly set `--ocbench_num_shards=1`; retain the same data budget across compared methods. Upstream downloads validation files together with training shards, but this trainer never loads or uses validation data for training.

## Per-task coefficient tuning

Two ordinary W&B grid sweeps are provided:

- `configs/sweeps/rebrac_v2_ocbench_full_bc_grid.yaml`: all 28 full-horizon tasks.
- `configs/sweeps/rebrac_v2_ocbench_lite_bc_grid.yaml`: the 16 selected lite tasks.

Both use the previous 4-by-4 coefficient grid and seeds **0, 1, 2, 3**:

```yaml
actor_bc_aux_weight: [0.03, 0.1, 0.5, 1.0]
actor_bc_coef: [0.0001, 0.001, 0.003, 0.01]
```

```bash
wandb sweep --entity YOUR_ENTITY --project ReBRAC2 \
  configs/sweeps/rebrac_v2_ocbench_lite_bc_grid.yaml
wandb agent YOUR_ENTITY/ReBRAC2/SWEEP_ID
```

The full sweep has **1,792 runs** and the selected lite sweep **1,024 runs**, each with 1M updates. They are provided, **not launched**. For a smaller job or one task at a time, copy the YAML and shorten `parameters.config_path.values`; one task is 64 runs. Extend coefficients or tune other model fields by adding parameters using the same CLI names. Selection should be performed **separately for each task by averaging its four seeds**, not by selecting W&B's globally best individual run. Use fresh seeds, e.g. 10-14, for confirmation. A sweep includes multiple tasks but does not optimize one shared average across tasks.

## Data and evaluation decisions

- Each episode has T transitions and T+1 observations. Sampling respects these offsets and never pairs observations/actions from adjacent episodes.
- Bellman termination is `1 - masks`: environment termination stops bootstrapping; time limits do not. Episode boundaries still stop n-step accumulation and reset action/state history.
- Configs keep successful demonstrations only, matching the published BC filtering choice. Bowling keeps all episodes because its rewards are fractional. Action-normalization statistics, if enabled, are fitted before filtering.
- Actions use the environment's existing `[-1, 1]` bounds by default, preserving ReBRAC-v2's action coordinates. Optional `--ocbench_action_normalization=percentile` uses OCBench's 1st/99th percentile interval and clip 5, then divides those coordinates by 5 so the **entire** clipped interval fits the bounded flow; gripper coordinates remain unchanged. The inverse transform is applied before stepping the environment. This is not silently clipping the benchmark's `[-5, 5]` policy coordinates to `[-1, 1]`. Coefficients, action noise, and inference step sizes are coordinate-dependent and may need retuning when this option changes.
- The categorical support starts at `[0, 1]`, expanded using the existing `v_expand` setting. OCBench's undiscounted task return is bounded by one, including Bowling. `v_min` and `v_max` can be overridden; they are not estimated across concatenated episodes.
- Warmups are inside the 1M-update budget: 100k BC, 200k critic, 700k joint training. Default evaluation is **only at the final checkpoint**, using 50 episodes. No initial rollout is performed.
- `eval/normalized_score_mean` is success percentage for manipulation tasks and mean return times 100 for Bowling. Raw return, success, episode length, and OCBench's success-per-normalized-horizon throughput are also logged. Standard deviations here are across evaluation episodes within one run, not across training seeds.
- Native evaluation uses fixed per-episode seeds, handles incomplete final batches, masks finished worlds, and enforces the registry time limit. Inference retains sampling, minimum-critic selection, and normalized critic-gradient refinement. It supports `nf_eval_num_samples="1,32"` and `q_infer_steps="0,2"` for evaluation grids without retraining.
- No action chunking, image encoder, or online data collection is introduced. This is the existing single-action ReBRAC-v2 recipe on a new benchmark.

The defaults are **not the published OCBench BC evaluation protocol** (2M updates, 500 episodes, averaging five late checkpoints). To evaluate that budget/checkpoint schedule without intermediate rollouts:

```bash
PYTHONPATH=. python algorithms/rebrac_v2_ocbench.py \
  --config_path=configs/offline/rebrac-v2-ocbench/full/block/block-single-task1-v0.yaml \
  --num_epochs=2000 --eval_episodes=500 \
  --eval_checkpoints=1800000,1850000,1900000,1950000,2000000
```

Warmups remain 100k/200k updates unless changed. Exact-checkpoint mode overrides periodic evaluation; final evaluation is always included. `eval/checkpoint_average_normalized_score_mean` averages only explicitly requested checkpoints, while `eval/normalized_score_mean` retains the current checkpoint score. The inherited `--track_checkpoint_average=true` selects 800k/900k/1M. It is mutually exclusive with `eval_checkpoints`.

## Outputs and supported options

Each run writes `ocbench_runs/<unique-run-name>/config.yaml`, `dataset.json`, and `metrics.jsonl`. Checkpoint saving is disabled by default to limit disk usage; pass `--save_final_checkpoint=true` to also write `final.msgpack`. The optional checkpoint contains actor/critic training states, observation statistics, action transform, and update count; it is intended for subsequent inference, not exact training resumption (host sampling RNG state is not saved). The checkpoint does not include the dataset. Disabling saving does not affect evaluation or checkpoint-score averaging, which use the in-memory models, and does not delete existing checkpoint files. The W&B project argument is respected; no entity/profile is hardcoded.

All normal ReBRAC-v2 architecture, BC, regularizer, optimizer, EMA, n-step, and inference fields use the OGBench names. `adam` preserves the existing actor weight decay / plain-Adam critic behavior; `adamw` explicitly applies weight decay to both. The inherited IQL, actor-reset, refinement-training, likelihood-alpha-target, reward-rescaling, and noisy-evaluation experiments are rejected instead of silently ignored. `critic_bc_coef` must remain zero because truncated datasets have no demonstrated action at the final bootstrap state. OGBench-specific goal/task flags are irrelevant here: task identity is in `dataset_name` and no artificial goal is appended.

## Regeneration and tests

```bash
PYTHONPATH=. python tools/make_ocbench_configs.py
PYTHONPATH=. python -m unittest discover -s tests -p test_ocbench.py -v
PYTHONPATH=. python -m unittest discover -s tests -p test_ocbench_download.py -v
```

Regeneration overwrites the 55 starting configs and two sweeps from the current OGBench base recipe. It preserves the five lite-group exclusions in the generated lite sweep, using the same selection rule as the downloaders. Do not run it over tuned configs you want to preserve. Unit tests cover compact transitions, filtering, timeouts, n-step targets, action transforms, rollout masking, all three optimization stages, checkpoints, config coverage, and W&B-style CLI overrides. GPU MJWarp execution must additionally be checked on the target CUDA machine; CPU tests do not validate that backend's kernels.

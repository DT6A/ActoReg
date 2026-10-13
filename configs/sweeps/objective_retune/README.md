# Coefficient-retuned objective ablations

Entry point: `configs/sweeps/rebrac_v2_objective_retune.yaml`.
Run commands from the repository root, with the ReBRAC-v2 environment active:

```bash
wandb sweep --entity tarasovd --project ReBRAC2 configs/sweeps/rebrac_v2_objective_retune.yaml
wandb agent tarasovd/ReBRAC2/SWEEP_ID
```

Replace `SWEEP_ID` with the ID returned by the first command. Authenticate with
`wandb login`; do not put API keys in the YAML or launch scripts. These are
standard [W&B sweep configurations](https://docs.wandb.ai/models/sweeps/define-sweep-configuration).

## Budget and fixed protocol

- Three BC objectives, eight distinct coefficient settings each.
- Six environments, using their original default single tasks:
  AntMaze Giant Navigate (task 1), AntSoccer Arena Navigate (task 4),
  Cube Double Play (task 2), HumanoidMaze Large Navigate (task 1),
  Puzzle 4x4 Play (task 4), and Scene Play (task 2).
- Four selection seeds: **0, 1, 2, 3**.
- **576 training runs**, 192 per objective and 96 per environment.
- One million update-loop iterations per run, including the unchanged
  100k BC warmup and 200k critic warmup, followed by 700k joint training.
- Final-checkpoint evaluation: 50 episodes, 32 action candidates, two
  Q-gradient refinement steps, step size 0.01, evaluation seed 42.
- The existing evaluator additionally logs an initial evaluation at epoch 0;
  ignore it for selection. With `eval_every: 1000`, subsequent selection scores
  come from epoch 999 (one million updates). Checkpoint averaging is disabled.

Architecture, optimizer, regularizers, inference settings, and training schedule
are copied from the shared recipe used by all six existing configurations.
Only the two BC weights differ among candidates. The Q-maximization term remains
active during joint training in **every** variant: "NLL-only" means NLL-only
*behavior regularization*, not pure behavioral cloning without RL.

The weights also affect the existing BC warmup objective. The warmup duration is
unchanged; no separate warmup-only coefficients are introduced.

## Candidate grids

`actor_bc_coef` weights flow NLL; `actor_bc_aux_weight` weights MSE + MAE.
`actor_bc_aux_loss: sum` is fixed throughout. The same eight candidates for each
objective are used in all six environments.

### Full: NLL + MSE + MAE

| Candidate | NLL coefficient | MSE + MAE coefficient |
| --- | ---: | ---: |
| 01 | 0.0001 | 0.03 |
| 02 | 0.0001 | 0.1 |
| 03 | 0.001 | 0.1 |
| 04 | 0.001 | 0.5 |
| 05 | 0.003 | 0.1 |
| 06 | 0.003 | 1.0 |
| 07 | 0.01 | 0.03 |
| 08 | 0.01 | 1.0 |

This is a targeted eight-point subset of the original 16-point grid. It contains
all five distinct coefficient pairs selected in the current six environment
configurations, plus three upper-range candidates. It is not the full Cartesian
product, and its choice benefits from the earlier full-objective search.

### NLL-only

Set `actor_bc_aux_weight = 0`; sweep `actor_bc_coef` over:

`[0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]`.

### MSE + MAE-only

Set `actor_bc_coef = 0`; sweep `actor_bc_aux_weight` over:

`[0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0]`.

The single-term searches extend below and above the original active-coefficient
ranges to allow compensation for the missing loss. These are proposed bounded
retuning grids, not experimentally verified optimal ranges. If a winner is at a
boundary, flag it before drawing conclusions; any expansion requires additional
budget and should be reported.

## Why the preset files exist

The single sweep enumerates 24 tiny files under `presets/` via the existing
`--config_path` argument. Each supplies an objective group and one coefficient
pair. All shared settings, environment names, and seeds are supplied by the
sweep as CLI overrides. Keep the preset directory when copying the sweep.

This avoids crossing independent coefficient grids: there are no duplicate
runs that only change the coefficient of a disabled term, and no candidate with
both coefficients zero. No training-code changes or custom launcher are needed.

## Select, then confirm

For **each environment and objective separately**, choose the coefficient pair
with the highest mean final `eval/normalized_score_mean` across all four seeds.
Do not use the best individual seed, an intermediate checkpoint, or a single
global sweep winner across environments/objectives. Require all four seeds to
finish before comparing candidates; rerun missing/failed seeds rather than
silently averaging fewer runs. Resolve exact ties deterministically by candidate
number, without consulting confirmation results.

Then freeze the 18 selected configurations (6 environments x 3 objectives) and
evaluate new training runs on seeds **10-14**, using the same final-checkpoint
protocol. That would require **90 additional runs**, not included in this
selection sweep. The selected values must be known before preparing that sweep.

Report the resulting comparison as **coefficient-retuned objective ablations**.
The new selection-run counts are equal across objectives, but the full objective
has prior development history; this does not equalize historical tuning compute
or establish superiority under arbitrary joint hyperparameter retuning.

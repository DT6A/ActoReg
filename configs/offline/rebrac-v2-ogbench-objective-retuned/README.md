# Retuned objective ablations: held-out-seed evaluation

The 24 YAML files contain one selected configuration per ablation and environment.
They preserve the settings logged by the tuning runs, including the default task,
1M training updates, 50 evaluation episodes, and final-checkpoint reporting.
The confirmation sweep uses training seeds **10, 11, 12, 13, 14**, disjoint from
the tuning seeds **0, 1, 2, 3**. The rollout `eval_seed` remains 42, as in tuning.
This is a fresh-training confirmation experiment, not evaluation of saved models.

## Sources and selection

All source sweeps are in `tarasovd/ReBRAC2`:

| Variant | Source sweep | Coefficient settings per environment |
| --- | --- | ---: |
| NLL only, no auxiliary BC | `jgvk1yc6` | 8 |
| Auxiliary MSE + MAE only, no NLL | `cxrxx9qa` | 8 |
| NLL + auxiliary MSE | `vfj4v2bg` | 16 |
| NLL + auxiliary MAE | `6veatu8l` | 16 |

Selection maximizes the arithmetic mean of `eval/normalized_score_mean` across
all four tuning seeds, separately for each variant and environment. Only finished
runs with a finite score at 1M updates are eligible. A coefficient setting needs
all four valid seeds; no partial-seed means or best-checkpoint scores are used.
Exact ties are broken by smaller auxiliary weight, then smaller NLL coefficient.
In particular, NLL-only AntMaze and HumanoidMaze have tied zero-score candidates;
their selected coefficients are deterministic tie choices, not evidence of an
improvement over the other tied settings.

All 1,152 scheduled runs are present and marked finished. However, these five
runs only logged evaluation at 1,000 updates, not at 1M updates:

| Variant | Environment | Seed | Run |
| --- | --- | ---: | --- |
| NLL + MAE | Scene | 2 | `0xkdp1nv` |
| NLL + MAE | Cube Double | 3 | `5ty1ll1s` |
| NLL + MSE | Puzzle 4x4 | 0 | `zac4vnhs` |
| NLL only | Scene | 1 | `wu3fv5jc` |
| NLL only | HumanoidMaze Large | 3 | `9l1b1pz2` |

Their five coefficient settings are excluded from selection rather than treating
the initial zero scores as final results. This leaves 283 eligible settings out
of 288. Every selected configuration has four valid final-checkpoint scores.

## Run

From the repository root, in the training environment:

```bash
wandb sweep --entity tarasovd --project ReBRAC2 configs/sweeps/rebrac_v2_retuned_objective_eval.yaml
wandb agent tarasovd/ReBRAC2/<returned-sweep-id>
```

This one grid contains **24 configurations x 5 seeds = 120 runs**. Each config
sets a variant-specific `abl-retuned-*` logging group. There are no coefficient
overrides in the sweep, so each run uses the values stored in its selected YAML.
It covers the original six default tasks only, not all five evaluation task IDs
of every environment. No runs were launched when preparing these files.

## Audit and regeneration

- `results_data/rebrac2_objective_retune_source.json`: source sweep definitions,
  logged run configurations, and evaluation summaries.
- `results_data/rebrac2_objective_retune_incomplete_history.json`: evaluation
  histories confirming the five missing final evaluations.
- `results_data/rebrac2_objective_retune_candidates.csv`: all 288 candidates,
  eligibility, per-seed scores/run IDs, and selected flags.
- `results_data/rebrac2_objective_retune_selection.csv`: the 24 selected rows.
- `results_data/rebrac2_objective_retune_excluded_runs.csv`: the five unusable runs.

Scores in these files are **selection scores**, not held-out confirmation results.
To regenerate the configs and sweep from the saved snapshot without network access:

```bash
python prepare_retuned_objective_eval.py
```

Regeneration requires PyYAML and overwrites the generated configs, sweep, and
selection CSVs. The source snapshot and original tuning configs remain unchanged.

"""Generate state-only task configs and ordinary W&B coefficient grids."""

import argparse
import json
from pathlib import Path

import gymnasium
import ocbench
import yaml


ROOT = Path(__file__).resolve().parents[1]


def state_tasks():
    return sorted(
        spec.id.replace("-cpu-", "-", 1)
        for spec in gymnasium.envs.registry.values()
        if isinstance(spec.entry_point, str) and spec.entry_point.startswith("ocbench.envs.")
        and not spec.id.startswith("visual-")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    base = yaml.safe_load((ROOT / "configs/offline/rebrac-plus-ogbench/scene/play_singletask_v0.yaml").read_text())
    base.update(
        project="ReBRAC2", group="rebrac-v2-ocbench", name="rebrac-v2-ocbench",
        actor_bc_aux_weight=0.03, actor_bc_coef=0.0003,
        ocbench_dataset_dir="~/.cache/ocbench/datasets", ocbench_num_shards=None,
        ocbench_action_normalization="bounds", ocbench_eval_backend="native",
        ocbench_eval_batch_size=50, updates_per_chunk=10, num_updates_on_epoch=1000,
        v_min=0.0, v_max=1.0, critic_bc_coef=0.0, normalize_states=False,
        save_final_checkpoint=True,
    )
    available = None
    if args.manifest:
        available = {
            entry["path"].split("/")[1]
            for line in args.manifest.read_text().splitlines()
            if (entry := json.loads(line))["kind"] == "data" and entry["split"] == "train"
        }
    groups = {"full": [], "lite": []}
    for task in state_tasks():
        if available is not None and task not in available:
            raise ValueError(f"No released data in the supplied manifest for {task}")
        family = task.split("-", 1)[0]
        category = "lite" if "-lite-" in task else "full"
        config = dict(base, dataset_name=task, ocbench_success_only=family != "bowling")
        relative = Path("configs/offline/rebrac-v2-ocbench") / category / family / f"{task}.yaml"
        destination = args.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
        groups[category].append(str(relative))
    for category, paths in groups.items():
        sweep = dict(
            program="algorithms/rebrac_v2_ocbench.py",
            command=["${env}", "PYTHONPATH=.", "python3", "${program}", "${args}"],
            method="grid", metric=dict(goal="maximize", name="eval/normalized_score_mean"),
            parameters=dict(
                config_path=dict(values=paths),
                actor_bc_aux_weight=dict(values=[0.03, 0.1, 0.5, 1.0]),
                actor_bc_coef=dict(values=[0.0001, 0.001, 0.003, 0.01]),
                train_seed=dict(values=[0, 1, 2, 3]),
                group=dict(value=f"rebrac-v2-ocbench-{category}-bc-grid"),
            ),
        )
        destination = args.output_root / f"configs/sweeps/rebrac_v2_ocbench_{category}_bc_grid.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(sweep, sort_keys=False))
        print(f"{category}: {len(paths)} configs, {len(paths) * 16 * 4} sweep runs")


if __name__ == "__main__":
    main()

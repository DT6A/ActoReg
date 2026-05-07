#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import wandb


def _parse_seeds(raw) -> List[int]:
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if raw is None:
        return [0, 1, 2, 3]
    text = str(raw).strip()
    if not text:
        return [0, 1, 2, 3]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_items(raw) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _normalize_arg_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return "dataset_name"
    return text[2:] if text.startswith("--") else text


def _strip_overrides(args: Iterable[str], blocked_keys: Iterable[str]) -> List[str]:
    blocked_prefixes = tuple(f"--{_normalize_arg_name(key)}" for key in blocked_keys)
    filtered = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if any(a == p for p in blocked_prefixes):
            skip_next = True
            continue
        if any(a.startswith(p + "=") for p in blocked_prefixes):
            continue
        filtered.append(a)
    return filtered


def _read_summary_metric(wandb_dir: Path, metric_key: str) -> float:
    summary_files = sorted(
        wandb_dir.glob("**/wandb-summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not summary_files:
        raise RuntimeError(f"No wandb-summary.json found in {wandb_dir}")

    summary_path = summary_files[-1]
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    if metric_key not in summary:
        raise RuntimeError(
            f"Metric '{metric_key}' not found in {summary_path}. "
            f"Available keys (sample): {list(summary.keys())[:20]}"
        )

    try:
        return float(summary[metric_key])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Metric '{metric_key}' is not numeric: {summary[metric_key]!r}") from exc


def _run_child(
    python_bin: str,
    program: str,
    base_args: List[str],
    seed: int,
    dataset: Optional[str],
    dataset_arg_name: str,
    group: str,
    parent_id: str,
    metric_key: str,
    base_wandb_dir: Path,
) -> float:
    dataset_suffix = f"-dataset-{_sanitize_name(dataset)}" if dataset is not None else ""
    child_dir = base_wandb_dir / f"seed_{seed}{dataset_suffix}"
    child_dir.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["WANDB_DIR"] = str(child_dir)

    # Prevent child runs from attaching to the parent sweep run.
    child_env.pop("WANDB_SWEEP_ID", None)
    child_env.pop("WANDB_RUN_ID", None)

    child_name = f"ms-{parent_id}-seed-{seed}{dataset_suffix}"
    cmd = [
        python_bin,
        program,
        *base_args,
        f"--train_seed={seed}",
        f"--group={group}",
        f"--name={child_name}",
    ]
    if dataset is not None:
        cmd.append(f"--{_normalize_arg_name(dataset_arg_name)}={dataset}")

    print("[multiseed] launching:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"[seed={seed}] {line}")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(
            f"Child run failed for seed={seed}, dataset={dataset!r} with exit code {ret}"
        )

    return _read_summary_metric(child_dir, metric_key)


def _sanitize_name(value: Optional[str]) -> str:
    if value is None:
        return "default"
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one sweep trial over multiple seeds and aggregate metrics.")
    parser.add_argument("--program", default="algorithms/rebrac_cl_plus.py")
    parser.add_argument("--python_bin", default="python3")
    parser.add_argument("--multiseed_seeds", default="0,1,2,3")
    parser.add_argument("--multiseed_datasets", default="")
    parser.add_argument("--multiseed_dataset_arg", default="dataset_name")
    parser.add_argument("--metric_key", default="eval/normalized_score_mean")
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    args, passthrough = parser.parse_known_args()

    run = wandb.init()
    cfg = dict(run.config)

    seeds = _parse_seeds(cfg.get("multiseed_seeds", args.multiseed_seeds))
    datasets = _parse_items(cfg.get("multiseed_datasets", args.multiseed_datasets))
    dataset_arg_name = _normalize_arg_name(
        cfg.get("multiseed_dataset_arg", args.multiseed_dataset_arg)
    )
    metric_key = str(cfg.get("metric_key", args.metric_key))
    aggregate = str(cfg.get("aggregate", args.aggregate))

    blocked_keys = ["train_seed", "group", "name"]
    if datasets:
        blocked_keys.append(dataset_arg_name)
    base_args = _strip_overrides(passthrough, blocked_keys)

    parent_id = run.id or f"manual-{int(time.time())}"
    child_group = f"multiseed-{parent_id}"
    base_wandb_dir = Path(os.environ.get("MULTISEED_CHILD_WANDB_DIR", f"/tmp/wandb-multiseed-{parent_id}"))
    base_wandb_dir.mkdir(parents=True, exist_ok=True)

    dataset_values: List[Optional[str]] = datasets or [None]
    scores = []
    per_dataset_scores = {}
    for dataset in dataset_values:
        dataset_scores = []
        for seed in seeds:
            score = _run_child(
                python_bin=args.python_bin,
                program=args.program,
                base_args=base_args,
                seed=seed,
                dataset=dataset,
                dataset_arg_name=dataset_arg_name,
                group=child_group,
                parent_id=parent_id,
                metric_key=metric_key,
                base_wandb_dir=base_wandb_dir,
            )
            if not np.isfinite(score):
                raise RuntimeError(f"Non-finite metric for seed={seed}, dataset={dataset!r}: {score}")
            status = "ok"
            scores.append(score)
            dataset_scores.append(score)
            metric_prefix = (
                f"multiseed/dataset_{_sanitize_name(dataset)}/seed_{seed}"
                if dataset is not None
                else f"multiseed/seed_{seed}"
            )
            run.log(
                {
                    f"{metric_prefix}/{metric_key}": score,
                    f"{metric_prefix}/status": status,
                }
            )
        if dataset is not None:
            per_dataset_scores[dataset] = dataset_scores

    values = np.asarray(scores, dtype=np.float64)
    if aggregate == "median":
        agg = float(np.median(values))
    else:
        agg = float(np.mean(values))

    std = float(np.std(values))
    log_payload = {
        metric_key: agg,
        "multiseed/metric_mean": float(np.mean(values)),
        "multiseed/metric_median": float(np.median(values)),
        "multiseed/metric_std": std,
        "multiseed/num_seeds": len(seeds),
        "multiseed/num_datasets": len(datasets) if datasets else 1,
    }
    for dataset, dataset_scores in per_dataset_scores.items():
        dataset_values_np = np.asarray(dataset_scores, dtype=np.float64)
        dataset_key = _sanitize_name(dataset)
        log_payload[f"multiseed/dataset_{dataset_key}/metric_mean"] = float(
            np.mean(dataset_values_np)
        )
        log_payload[f"multiseed/dataset_{dataset_key}/metric_median"] = float(
            np.median(dataset_values_np)
        )
        log_payload[f"multiseed/dataset_{dataset_key}/metric_std"] = float(
            np.std(dataset_values_np)
        )
    run.log(log_payload)
    run.summary[metric_key] = agg
    run.summary["multiseed/metric_std"] = std
    run.summary["multiseed/seeds"] = seeds
    run.summary["multiseed/datasets"] = datasets
    run.summary["multiseed/dataset_arg"] = dataset_arg_name
    run.summary["multiseed/child_group"] = child_group

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Download the OGBench datasets used by the ReBRAC-plus OGBench configs.

Examples:
  PYTHONPATH=. python3 tools/download_ogbench_datasets.py
  PYTHONPATH=. python3 tools/download_ogbench_datasets.py --include-task-variants
  PYTHONPATH=. python3 tools/download_ogbench_datasets.py --dataset-dir ~/.ogbench/data
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


DEFAULT_DATASETS = [
    "antmaze-large-navigate-singletask-v0",
    "antmaze-giant-navigate-singletask-v0",
    "humanoidmaze-medium-navigate-singletask-v0",
    "humanoidmaze-large-navigate-singletask-v0",
    "antsoccer-arena-navigate-singletask-v0",
    "cube-single-play-singletask-v0",
    "cube-double-play-singletask-v0",
    "scene-play-singletask-v0",
    "puzzle-3x3-play-singletask-v0",
    "puzzle-4x4-play-singletask-v0",
]


def _import_ogbench():
    try:
        import ogbench  # type: ignore

        return ogbench
    except ImportError:
        ogbench_repo = os.path.expanduser("~/ogbench")
        if os.path.isdir(ogbench_repo) and ogbench_repo not in sys.path:
            sys.path.append(ogbench_repo)
        import ogbench  # type: ignore

        return ogbench


def _task_variants(dataset_name: str) -> List[str]:
    if "-singletask-" not in dataset_name:
        return [dataset_name]
    return [dataset_name.replace("-singletask-", f"-singletask-task{task_id}-", 1) for task_id in range(1, 6)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OGBench datasets used by local ReBRAC-plus configs.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="~/.ogbench/data",
        help="Target OGBench dataset directory.",
    )
    parser.add_argument(
        "--include-task-variants",
        action="store_true",
        help="Also trigger downloads for singletask task1-task5 environment names.",
    )
    parser.add_argument(
        "--env-name",
        action="append",
        default=[],
        help="Optional extra OGBench environment name. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ogbench = _import_ogbench()

    dataset_dir = os.path.expanduser(args.dataset_dir)
    env_names = list(DEFAULT_DATASETS)
    if args.env_name:
        env_names.extend(args.env_name)

    expanded_env_names: List[str] = []
    for env_name in env_names:
        if args.include_task_variants:
            expanded_env_names.extend(_task_variants(env_name))
        else:
            expanded_env_names.append(env_name)

    # Preserve order while avoiding duplicate downloads.
    deduped_env_names = list(dict.fromkeys(expanded_env_names))

    print(f"dataset_dir: {dataset_dir}")
    print(f"num_env_names: {len(deduped_env_names)}")
    for env_name in deduped_env_names:
        print(f"[download] {env_name}")
        ogbench.make_env_and_datasets(env_name, dataset_dir=dataset_dir)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

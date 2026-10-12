"""Download state-only OCBench datasets selected from the ReBRAC-v2 configs."""

import argparse
import json
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/offline/rebrac-v2-ocbench"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def configured_environments(config_root, suite, requested=()):
    categories = ("full", "lite") if suite == "all" else (suite,)
    names = set()
    for category in categories:
        paths = sorted((config_root / category).rglob("*.yaml"))
        if not paths:
            raise ValueError(f"No {category} configs found under {config_root}")
        for path in paths:
            config = yaml.safe_load(path.read_text())
            name = config["dataset_name"].replace("-cpu-", "-", 1)
            if name.startswith("visual-"):
                raise ValueError(f"Visual dataset is not allowed: {path}")
            if ("-lite-" in name) != (category == "lite"):
                raise ValueError(f"Task is in the wrong suite directory: {path}")
            names.add(name)
    if requested:
        requested = {name.replace("-cpu-", "-", 1) for name in requested}
        unknown = requested - names
        if unknown:
            raise ValueError(f"Tasks not configured for --suite {suite}: {', '.join(sorted(unknown))}")
        names = requested
    return sorted(names)


def load_manifest(dataset_dir):
    from huggingface_hub import hf_hub_download
    from ocbench.dataset_utils import DATASET_REPO, DATASET_REVISION

    destination = dataset_dir / ".cache/ocbench" / DATASET_REVISION
    path = destination / "manifest.jsonl"
    if not path.is_file():
        hf_hub_download(DATASET_REPO, "manifest.jsonl", repo_type="dataset",
                         revision=DATASET_REVISION, local_dir=str(destination))
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def selected_files(manifest, names, num_shards):
    result = {}
    for name in names:
        category = "lite" if "-lite-" in name else "full"
        directory = f"{category}/{name}"
        entries = {entry["path"]: entry for entry in manifest if entry["path"].rsplit("/", 1)[0] == directory}
        train_paths = sorted(path for path, entry in entries.items() if entry["kind"] == "data" and entry["split"] == "train")
        if num_shards is not None:
            train_paths = train_paths[:num_shards]
        if not train_paths:
            raise ValueError(f"No released training shards for {name}")
        selected = []
        for path in train_paths:
            stem = path.removesuffix(".npz")
            for suffix in (".npz", "-val.npz", "-metadata.pkl", "-val-metadata.pkl"):
                companion = stem + suffix
                if companion not in entries:
                    raise ValueError(f"Missing file in release manifest: {companion}")
                selected.append(entries[companion])
        result[name] = (len(train_paths), selected)
    return result


def main(argv=None, fixed_suite=None):
    description = __doc__ if fixed_suite is None else f"Download only the state-based OCBench {fixed_suite} datasets."
    parser = argparse.ArgumentParser(description=description)
    if fixed_suite is None:
        parser.add_argument("--suite", choices=("full", "lite", "all"), default="full",
                            help="28 main tasks, 27 optional lite variants, or both. Default: full.")
    else:
        parser.set_defaults(suite=fixed_suite)
    parser.add_argument("--dataset-dir", type=Path, default=Path("~/.cache/ocbench/datasets"))
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--env-name", action="append", default=[],
                        help="Restrict to this task within the selected suite; repeat for multiple tasks.")
    parser.add_argument("--num-shards", type=positive_int,
                        help="First N training shards per task plus companions; default: all shards.")
    parser.add_argument("--list", action="store_true", help="List task IDs only; no network access or OCBench import.")
    parser.add_argument("--dry-run", action="store_true", help="Read the small release manifest and show sizes without downloading datasets.")
    args = parser.parse_args(argv)
    try:
        names = configured_environments(args.config_root.expanduser(), args.suite, args.env_name)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.list:
        print("\n".join(names))
        return 0
    try:
        import ocbench
    except ImportError:
        parser.error("Install OCBench first; downloading needs only the base package, not CUDA or mjwarp.")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest = load_manifest(dataset_dir)
    files = selected_files(manifest, names, args.num_shards)
    total_bytes = 0
    existing_bytes = 0
    print(f"Destination: {dataset_dir}")
    for name, (shards, entries) in files.items():
        size = sum(entry["bytes"] for entry in entries)
        total_bytes += size
        for entry in entries:
            path = dataset_dir / entry["path"]
            if path.is_file() and path.stat().st_size == entry["bytes"]:
                existing_bytes += entry["bytes"]
        print(f"{name}: {shards} training shards, {size / 1e9:.2f} GB including validation/metadata")
    print(f"Total: {len(names)} datasets, {total_bytes / 1e9:.2f} GB ({total_bytes / 1e12:.3f} TB).")
    print(f"Files absent or with a different size: {(total_bytes - existing_bytes) / 1e9:.2f} GB.")
    print("The official downloader validates cached files; temporary download/cache space may also be needed.")
    if args.dry_run:
        print("Dry run: no dataset files downloaded.")
        return 0
    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        try:
            ocbench.download_datasets(name, dataset_root=str(dataset_dir), num_shards=args.num_shards)
        except Exception as error:
            print(f"Download failed for {name}: {error}\nRerun the same command to reuse the download cache.", file=sys.stderr)
            return 1
    print("All selected datasets are ready. No environments were created or data loaded into RAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

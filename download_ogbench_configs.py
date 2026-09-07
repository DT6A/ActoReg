#!/usr/bin/env python3
"""
Download exactly the OGBench datasets required by the rebrac-plus-ogbench configs.

It scans configs/offline/rebrac-plus-ogbench/**/*.yaml, reads each `dataset_name`,
and derives the underlying base dataset (OGBench singletask datasets reuse the
base goal-conditioned .npz files -- the `-singletask`/`-task<N>` suffix only
changes the reward relabeling, not the downloaded data). Each base dataset has a
train file `<base>.npz` and a validation file `<base>-val.npz`.

This stays in sync with whatever configs exist, so it never downloads visual or
powderworld data unless you actually added configs for them.

Usage:
    python download_ogbench_configs.py                 # download all configs' datasets
    python download_ogbench_configs.py --list          # just print the base datasets
    python download_ogbench_configs.py --dataset_dir /path/to/data
    python download_ogbench_configs.py --no_skip       # re-download existing files
"""

import argparse
import os
import re
import urllib.error
import urllib.request

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
CONFIGS_DIR = "configs/offline/rebrac-plus-ogbench"


def base_dataset(dataset_name):
    """'antmaze-large-navigate-singletask-task2-v0' -> 'antmaze-large-navigate-v0'."""
    name = dataset_name.strip()
    name = re.sub(r"-singletask", "", name)
    name = re.sub(r"-task\d+", "", name)
    return name


def collect_base_datasets(configs_dir):
    """Read dataset_name from every config yaml, return the sorted set of bases."""
    bases = set()
    for root, _, files in os.walk(configs_dir):
        for fn in files:
            if not fn.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(root, fn)) as f:
                for line in f:
                    m = re.match(r"\s*dataset_name:\s*(\S+)", line)
                    if m:
                        bases.add(base_dataset(m.group(1).strip().strip('"\'')))
                        break
    return sorted(bases)


def _open(url, retries=3):
    """urlopen with a browser-like User-Agent and simple retry/backoff.
    Some hosts 403 the default 'Python-urllib' agent; transient 5xx/connection
    errors are retried."""
    import time
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ogbench-downloader",
        "Accept": "*/*",
    })
    last = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 404):  # not transient -- don't hammer the host
                raise
        except Exception as e:  # noqa: BLE001 - connection resets etc.
            last = e
        time.sleep(2 * (attempt + 1))
    raise last


def download_file(url, dest_path):
    response = _open(url)
    total = getattr(response, "length", None)
    tmp_path = dest_path + ".tmp"
    filename = os.path.basename(dest_path)
    if HAS_TQDM:
        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=filename, miniters=1) as bar:
            with open(tmp_path, "wb") as f:
                for chunk in response:
                    f.write(chunk)
                    bar.update(len(chunk))
    else:
        with open(tmp_path, "wb") as f:
            for chunk in response:
                f.write(chunk)
    os.rename(tmp_path, dest_path)


def download_datasets(base_names, dataset_dir, skip_existing=True, base_url=DATASET_URL):
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    files = []
    for name in base_names:
        files.append(f"{name}.npz")
        files.append(f"{name}-val.npz")
    total = len(files)
    errors = []
    for i, filename in enumerate(files, 1):
        dest = os.path.join(dataset_dir, filename)
        if skip_existing and os.path.exists(dest):
            print(f"[{i}/{total}] skip {filename} (exists)")
            continue
        url = f"{base_url.rstrip('/')}/{filename}"
        print(f"[{i}/{total}] downloading {filename}")
        try:
            download_file(url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}")
            errors.append((filename, str(e)))
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for fn, e in errors:
            print(f"  {fn}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs_dir", default=CONFIGS_DIR)
    p.add_argument("--dataset_dir", default="~/.ogbench/data")
    p.add_argument("--list", action="store_true", help="print the base datasets and exit")
    p.add_argument("--no_skip", action="store_true", help="re-download even if the file exists")
    p.add_argument("--start-from", metavar="NAME", default=None,
                   help="skip datasets whose name sorts before NAME (e.g. 'pointmaze' keeps "
                        "pointmaze/puzzle/scene and drops antmaze/antsoccer/cube/humanoidmaze)")
    p.add_argument("--url", default=DATASET_URL,
                   help=f"base URL to download from (default: {DATASET_URL}); point at a mirror "
                        "if the default host 403s")
    p.add_argument("--via-ogbench", action="store_true",
                   help="delegate the fetching to ogbench.download_datasets instead of the "
                        "built-in downloader (same datasets, ogbench's own logic)")
    args = p.parse_args()

    bases = collect_base_datasets(args.configs_dir)
    if not bases:
        raise SystemExit(f"No dataset_name entries found under {args.configs_dir}")
    if args.start_from:
        before = len(bases)
        bases = [b for b in bases if b >= args.start_from]
        print(f"--start-from {args.start_from!r}: kept {len(bases)} of {before} datasets")

    if args.list:
        for b in bases:
            print(b)
        print(f"\n{len(bases)} base dataset(s) ({len(bases) * 2} files)")
        return

    print(f"Downloading {len(bases)} base dataset(s) ({len(bases) * 2} files) to {args.dataset_dir}")
    if args.via_ogbench:
        import ogbench
        ogbench.download_datasets(bases, dataset_dir=os.path.expanduser(args.dataset_dir))
    else:
        download_datasets(bases, args.dataset_dir, skip_existing=not args.no_skip, base_url=args.url)
    print("Done.")


if __name__ == "__main__":
    main()

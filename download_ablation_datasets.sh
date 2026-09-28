#!/usr/bin/env bash
set -euo pipefail

datasets=(
  antmaze-giant-navigate-v0
  antsoccer-arena-navigate-v0
  cube-double-play-v0
  humanoidmaze-large-navigate-v0
  puzzle-4x4-play-v0
  scene-play-v0
)

if [[ $# -gt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: bash %s [DATASET_DIRECTORY | --list]\n' "$0"
  printf 'Downloads only the six ablation datasets, including train and validation files.\n'
  printf 'Default directory: ~/.ogbench/data. Existing nonempty files are skipped.\n'
  if [[ $# -gt 1 ]]; then exit 2; fi
  exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${datasets[@]}"
  exit 0
fi

dataset_dir="${1:-$HOME/.ogbench/data}"
if [[ "$dataset_dir" == '~/'* ]]; then
  dataset_dir="$HOME/${dataset_dir:2}"
fi
command -v curl >/dev/null || { printf 'Error: curl is required.\n' >&2; exit 1; }
mkdir -p -- "$dataset_dir"
base_url='https://rail.eecs.berkeley.edu/datasets/ogbench'
partial_file=''
trap 'if [[ -n "$partial_file" ]]; then rm -f -- "$partial_file"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for dataset in "${datasets[@]}"; do
  for suffix in '' '-val'; do
    filename="${dataset}${suffix}.npz"
    target="$dataset_dir/$filename"
    if [[ -s "$target" ]]; then
      printf 'Skipping %s (already exists)\n' "$filename"
      continue
    fi
    printf 'Downloading %s\n' "$filename"
    partial_file=$(mktemp "$target.part.XXXXXX")
    curl --fail --location --proto '=https' --retry 3 --connect-timeout 30 \
      --progress-bar --output "$partial_file" "$base_url/$filename"
    if [[ ! -s "$partial_file" ]]; then
      printf 'Error: received an empty file for %s\n' "$filename" >&2
      exit 1
    fi
    mv -- "$partial_file" "$target"
    partial_file=''
  done
done

printf 'All six datasets (12 train/validation files) are available in %s\n' "$dataset_dir"

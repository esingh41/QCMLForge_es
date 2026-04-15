#!/usr/bin/env bash
# Convert one or more pandas-2 pkl files to pandas-3 compatible pkl files.
# Workflow per file:
#   1. qcml env  (pandas 2) : pkl  -> parquet
#   2. psi4 env  (pandas 3) : parquet -> pkl   (overwrites the original)
#
# Usage:
#   ./roundtrip_pkl_parquet.sh file1.pkl [file2.pkl ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    printf "Usage: %s <file1.pkl> [file2.pkl] ...\n" "$0" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    printf "conda is not available in PATH.\n" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for INPUT_PKL in "$@"; do
    # Resolve to absolute path before any env switches
    INPUT_PKL="$(realpath "$INPUT_PKL")"

    if [[ ! -f "$INPUT_PKL" ]]; then
        printf "[SKIP] Not found: %s\n" "$INPUT_PKL" >&2
        continue
    fi

    printf "[START] %s\n" "$INPUT_PKL"

    TMP_PKL="$TMP_DIR/input.pkl"
    TMP_PARQUET="$TMP_DIR/input.parquet"

    cp "$INPUT_PKL" "$TMP_PKL"

    # Step 1: qcml (pandas 2) — read old pkl, write parquet
    conda activate qcml
    PYTHONPATH="$SCRIPT_DIR" python - "$TMP_PKL" "$TMP_PARQUET" <<'PY'
import sys
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])  # no-op, PYTHONPATH already set
from convert_pkl_parquet import pkl_to_parquet
import os

pkl_path = sys.argv[1]
out_parquet = sys.argv[2]

pkl_to_parquet(pkl_path)

# pkl_to_parquet writes next to the source; move to our desired tmp path
auto_parquet = pkl_path.rsplit(".", 1)[0] + ".parquet"
if auto_parquet != out_parquet:
    os.replace(auto_parquet, out_parquet)
PY

    # Step 2: psi4 (pandas 3) — read parquet, write pandas-3 pkl
    conda activate psi4
    PYTHONPATH="$SCRIPT_DIR" python - "$TMP_PARQUET" "$INPUT_PKL" <<'PY'
import sys, os
from convert_pkl_parquet import parquet_to_pkl

parquet_path = sys.argv[1]
final_pkl    = sys.argv[2]

parquet_to_pkl(parquet_path)

# parquet_to_pkl writes next to the source; move to the original path
auto_pkl = parquet_path.rsplit(".", 1)[0] + ".pkl"
if auto_pkl != final_pkl:
    os.replace(auto_pkl, final_pkl)
PY

    printf "[DONE]  %s\n" "$INPUT_PKL"
done

printf "\nAll done.\n"

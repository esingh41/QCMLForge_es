#!/usr/bin/env bash
# Convert one or more pickle files between arbitrary pandas environments.
# Workflow per file:
#   1. source env : pkl     -> parquet
#   2. target env : parquet -> pkl     (overwrites the original)
#
# Usage:
#   ./pandas_v2_to_v3.sh [--from-env <env>] [--to-env <env>] file1.pkl [file2.pkl ...]
#
# Defaults:
#   --from-env qcml
#   --to-env   psi4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM_ENV="qcml"
TO_ENV="psi4"

usage() {
    printf "Usage: %s [--from-env <env>] [--to-env <env>] <file1.pkl> [file2.pkl] ...\n" "$0" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-env|-f)
            if [[ $# -lt 2 ]]; then
                printf "Missing value for %s\n" "$1" >&2
                usage
                exit 1
            fi
            FROM_ENV="$2"
            shift 2
            ;;
        --to-env|-t)
            if [[ $# -lt 2 ]]; then
                printf "Missing value for %s\n" "$1" >&2
                usage
                exit 1
            fi
            TO_ENV="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            printf "Unknown option: %s\n" "$1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    printf "conda is not available in PATH.\n" >&2
    exit 1
fi

for ENV_NAME in "$FROM_ENV" "$TO_ENV"; do
    if ! conda run -n "$ENV_NAME" python -c "import pandas as pd" >/dev/null; then
        printf "Cannot run python/pandas in conda env: %s\n" "$ENV_NAME" >&2
        exit 1
    fi
done

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

    # Step 1: source env — read pkl, write parquet
    PYTHONPATH="$SCRIPT_DIR" PKL_PATH="$TMP_PKL" OUT_PARQUET="$TMP_PARQUET" \
        conda run -n "$FROM_ENV" python -c "import os, shutil; from convert_pkl_parquet import pkl_to_parquet; pkl_path=os.environ['PKL_PATH']; out_parquet=os.environ['OUT_PARQUET']; pkl_to_parquet(pkl_path); auto_parquet=pkl_path.rsplit('.', 1)[0]+'.parquet'; shutil.move(auto_parquet, out_parquet) if auto_parquet != out_parquet else None"

    # Step 2: target env — read parquet, write pkl
    PYTHONPATH="$SCRIPT_DIR" PARQUET_PATH="$TMP_PARQUET" FINAL_PKL="$INPUT_PKL" \
        conda run -n "$TO_ENV" python -c "import os, shutil; from convert_pkl_parquet import parquet_to_pkl; parquet_path=os.environ['PARQUET_PATH']; final_pkl=os.environ['FINAL_PKL']; parquet_to_pkl(parquet_path); auto_pkl=parquet_path.rsplit('.', 1)[0]+'.pkl'; shutil.move(auto_pkl, final_pkl) if auto_pkl != final_pkl else None"

    printf "[DONE]  %s\n" "$INPUT_PKL"
done

printf "\nAll done.\n"

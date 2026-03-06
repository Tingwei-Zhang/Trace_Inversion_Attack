#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
python "${SCRIPT_DIR}/run_inversion_eval.py" --preset surrogate_r1_on_chatgpt "$@"

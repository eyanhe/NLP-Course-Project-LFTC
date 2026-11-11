#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash ops/run.sh CONFIG.yaml [additional lftc run arguments]" >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"
shift
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
exec "${LFTC_PYTHON:-python}" -m lftc run --config "$config" "$@"

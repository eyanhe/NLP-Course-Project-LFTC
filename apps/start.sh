#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
exec "${LFTC_PYTHON:-python}" -m streamlit run "$project_root/apps/streamlit_app.py" "$@"

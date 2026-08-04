#!/usr/bin/env bash
# Launch the experiment comparison UI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  # Prefer the project's RL env when available.
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    :
  elif conda env list | awk '{print $1}' | grep -qx 'rl_lab'; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rl_lab
  fi
fi

exec streamlit run tools/experiment_compare/app.py --server.headless true "$@"

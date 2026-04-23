#!/usr/bin/env bash
# Usage:  source scripts/use.sh <extra>
# Example: source scripts/use.sh cogvlm
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: source scripts/use.sh <extra>" >&2
    return 1 2>/dev/null || exit 1
fi
ENV="$1"
mkdir -p .venvs
[ -d ".venvs/$ENV" ] || uv venv ".venvs/$ENV" --python 3.10
# shellcheck disable=SC1090
source ".venvs/$ENV/bin/activate"
uv sync --extra "$ENV" --active

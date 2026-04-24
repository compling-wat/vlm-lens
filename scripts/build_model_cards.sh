#!/usr/bin/env bash
# Orchestrate the model-card probe across every configs/models/<arch>/*.yaml
# checkpoint. Activates the appropriate per-arch uv venv, runs
# `python -m src.main -c <config> -l`, and records results to
# docs/_data/cards/. Idempotent: probes whose JSON already exists are skipped
# unless --force is passed.
#
# Usage:
#   bash scripts/build_model_cards.sh              # probe all
#   bash scripts/build_model_cards.sh --force      # re-probe all
#   bash scripts/build_model_cards.sh clip qwen    # only these arch dirs
#
# NOTE: must be run with scripts/use.sh's requirements satisfied (uv on PATH).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIGS="$ROOT/configs/models"
CARDS="$ROOT/docs/_data/cards"

FORCE=0
ARCH_FILTER=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '1,15p' "$0"; exit 0 ;;
        *) ARCH_FILTER+=("$arg") ;;
    esac
done

# arch dir -> uv extras name.
declare -A EXTRA=(
    [aya-vision]=base
    [blip2]=base
    [clip]=base
    [dino]=base
    [llava]=base
    [llavanext]=base
    [qwen]=base
    [cogvlm]=cogvlm
    [glamm]=glamm
    [internlm-xc]=internlm-xcomposer
    [internvl]=internvl
    [janus]=janus
    [minicpm-o]=minicpm-o
    [minicpm-V2]=minicpm-v
    [molmo]=molmo
    [paligemma]=paligemma
    [pixtral]=pixtral
    [plm]=plm
)

contains() {
    local needle="$1"; shift
    for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}

model_path_of() {
    # Extract `model_path:` from a YAML config (simple single-line form).
    grep -E '^model_path:' "$1" | head -1 | awk '{print $2}' | tr -d '"'
}

cd "$ROOT" || exit 1

current_extra=""
for arch_dir in "$CONFIGS"/*/; do
    arch=$(basename "$arch_dir")
    if [[ ${#ARCH_FILTER[@]} -gt 0 ]] && ! contains "$arch" "${ARCH_FILTER[@]}"; then
        continue
    fi
    extra="${EXTRA[$arch]:-}"
    if [[ -z "$extra" ]]; then
        echo "[skip] $arch — no extras mapping known" >&2
        continue
    fi

    if [[ "$extra" != "$current_extra" ]]; then
        echo "[env] switching to extras=$extra"
        # shellcheck disable=SC1091
        source "$ROOT/scripts/use.sh" "$extra" || {
            echo "[fail] could not activate $extra" >&2
            continue
        }
        current_extra="$extra"
    fi

    for cfg in "$arch_dir"*.yaml; do
        [[ -e "$cfg" ]] || continue
        mp=$(model_path_of "$cfg")
        if [[ -z "$mp" ]]; then
            echo "[warn] $(basename "$cfg"): no model_path, skipping" >&2
            continue
        fi
        card_json="$CARDS/$mp.json"
        if [[ "$FORCE" -eq 0 && -f "$card_json" ]]; then
            echo "[skip] $mp (card exists)"
            continue
        fi
        echo "[probe] $arch / $(basename "$cfg") -> $mp"
        uv run --active python -m src.main -c "$cfg" -l || {
            echo "[fail] $cfg" >&2
        }
    done
done

echo "[done] rendering RST cards..."
# renderer is pure-python; run it in whatever venv is active
uv run --active python "$ROOT/scripts/build_model_cards.py"

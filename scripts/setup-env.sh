#!/usr/bin/env bash
# Create or update the Assay development environment.
#
# Uses a dedicated conda environment when conda is available, so project
# dependencies never mix with other environments; otherwise falls back to a
# uv-managed .venv. Dependencies always come from uv.lock.
#
# Usage: scripts/setup-env.sh [--env-name NAME] [--all-extras] [--no-conda]
set -euo pipefail

env_name="assay"
all_extras=false
use_conda=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) env_name="$2"; shift 2 ;;
        --all-extras) all_extras=true; shift ;;
        --no-conda) use_conda=false; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

command -v uv >/dev/null || {
    echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

python_version="$(head -n1 .python-version | tr -d '[:space:]')"
sync_args=(sync --locked --extra app)
$all_extras && sync_args+=(--all-extras)

find_conda() {
    if command -v conda >/dev/null; then command -v conda; return; fi
    for dist in miniforge3 miniconda3 anaconda3; do
        for candidate in "$HOME/$dist/bin/conda" "$HOME/$dist/Scripts/conda.exe"; do
            [[ -x "$candidate" ]] && { echo "$candidate"; return; }
        done
    done
}

conda=""
$use_conda && conda="$(find_conda || true)"

env_prefix() {
    # Prints the prefix of the named env, or nothing. JSON escapes Windows
    # backslashes, so unescape before matching.
    "$conda" env list --json | tr -d '\r' | grep -oE '"[^"]+"' | tr -d '"' \
        | sed 's#\\\\#\\#g' | grep -E "[/\\\\]envs[/\\\\]${env_name}$" | head -n1 || true
}

if [[ -n "$conda" ]]; then
    prefix="$(env_prefix)"
    if [[ -z "$prefix" ]]; then
        echo "Creating conda env '$env_name' (Python $python_version)..."
        "$conda" create -n "$env_name" "python=$python_version" -y
        prefix="$(env_prefix)"
    else
        echo "Using existing conda env '$env_name' at $prefix"
    fi

    # Make uv install into the conda env whenever it is activated.
    "$conda" env config vars set -n "$env_name" "UV_PROJECT_ENVIRONMENT=$prefix" >/dev/null
    export UV_PROJECT_ENVIRONMENT="$prefix"

    if [[ -x "$prefix/bin/python" ]]; then env_python="$prefix/bin/python"; else env_python="$prefix/python.exe"; fi
    # --inexact keeps packages conda installed that uv.lock does not list.
    sync_args+=(--inexact --python "$env_python")
else
    echo "conda not found or disabled; using a uv-managed .venv"
    sync_args+=(--python "$python_version")
fi

echo "Installing locked dependencies..."
uv "${sync_args[@]}"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example; add your provider keys."
fi

echo
echo "Done."
if [[ -n "$conda" ]]; then echo "Activate with: conda activate $env_name"; else echo "Run commands with: uv run <command>"; fi

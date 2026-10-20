#!/usr/bin/env bash
# Run the same checks as CI, locally: formatting, lint, type-check, tests
# (excluding real LLM calls) and the GitHub Actions security audit.
# Stops at the first failing step.
#
# Usage: scripts/check.sh [--fix] [--no-sync]
set -euo pipefail

fix=false
sync=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix) fix=true; shift ;;
        --no-sync) sync=false; shift ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

passed=0

step() {
    local name="$1"; shift
    printf '\033[36m==> %s\033[0m\n' "$name"
    local code=0
    "$@" || code=$?
    if [[ $code -ne 0 ]]; then
        printf '\033[31mFAILED: %s\033[0m\n' "$name" >&2
        exit "$code"
    fi
    passed=$((passed + 1))
}

if $sync; then
    # --inexact keeps conda-installed packages when syncing into a conda env.
    sync_args=(sync --locked --extra app --extra bedrock --extra openai)
    [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]] && sync_args+=(--inexact)
    step "Sync dependencies" uv "${sync_args[@]}"
fi

if $fix; then
    step "Format (apply)" uv run --no-sync ruff format .
    step "Lint (apply fixes)" uv run --no-sync ruff check --fix .
fi

step "Format" uv run --no-sync ruff format --check .
step "Lint" uv run --no-sync ruff check .
step "Type-check" uv run --no-sync mypy
step "Tests" uv run --no-sync pytest -m "not llm"
step "Workflow audit" uvx zizmor --offline --quiet .github

printf '\n\033[32mAll %d checks passed.\033[0m\n' "$passed"

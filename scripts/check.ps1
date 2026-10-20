<#
.SYNOPSIS
    Run the same checks as CI, locally.

.DESCRIPTION
    Formatting, lint, type-check, tests (excluding real LLM calls) and the
    GitHub Actions security audit. Stops at the first failing step.

.EXAMPLE
    ./scripts/check.ps1           # verify, like CI
    ./scripts/check.ps1 -Fix      # apply formatting and safe lint fixes first
    ./scripts/check.ps1 -NoSync   # skip dependency sync
#>
[CmdletBinding()]
param(
    [switch]$Fix,
    [switch]$NoSync
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$script:passed = @()

function Invoke-Step {
    param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string[]]$Command)
    Write-Host "==> $Name" -ForegroundColor Cyan
    $exe, $rest = $Command
    & $exe @rest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $script:passed += $Name
}

if (-not $NoSync) {
    # --inexact keeps conda-installed packages when syncing into a conda env.
    $sync = @("uv", "sync", "--locked", "--extra", "app")
    if ($env:UV_PROJECT_ENVIRONMENT) { $sync += "--inexact" }
    Invoke-Step "Sync dependencies" $sync
}

if ($Fix) {
    Invoke-Step "Format (apply)" @("uv", "run", "--no-sync", "ruff", "format", ".")
    Invoke-Step "Lint (apply fixes)" @("uv", "run", "--no-sync", "ruff", "check", "--fix", ".")
}

Invoke-Step "Format" @("uv", "run", "--no-sync", "ruff", "format", "--check", ".")
Invoke-Step "Lint" @("uv", "run", "--no-sync", "ruff", "check", ".")
Invoke-Step "Type-check" @("uv", "run", "--no-sync", "mypy")
Invoke-Step "Tests" @("uv", "run", "--no-sync", "pytest", "-m", "not llm")
Invoke-Step "Workflow audit" @("uvx", "zizmor", "--offline", "--quiet", ".github")

Write-Host ""
Write-Host "All $($script:passed.Count) checks passed." -ForegroundColor Green

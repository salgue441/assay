<#
.SYNOPSIS
    Create or update the Assay development environment.

.DESCRIPTION
    Uses a dedicated conda environment when conda is available, so project
    dependencies never mix with other environments; otherwise falls back to a
    uv-managed .venv. Dependencies always come from uv.lock.

.EXAMPLE
    ./scripts/setup-env.ps1
    ./scripts/setup-env.ps1 -AllExtras
    ./scripts/setup-env.ps1 -NoConda
#>
[CmdletBinding()]
param(
    [string]$EnvName = "assay",
    [switch]$AllExtras,
    [switch]$NoConda
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Native {
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -like "*.exe") { return $cmd.Source }
    foreach ($dist in "miniforge3", "miniconda3", "anaconda3") {
        $candidate = Join-Path $env:USERPROFILE "$dist\Scripts\conda.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"
}

$pythonVersion = (Get-Content (Join-Path $root ".python-version") -TotalCount 1).Trim()
$syncArgs = @("sync", "--locked", "--extra", "app")
if ($AllExtras) { $syncArgs += "--all-extras" }

$conda = if ($NoConda) { $null } else { Find-Conda }

function Get-EnvPrefix {
    # -AsHashtable: conda's JSON can repeat a key with different drive-letter casing.
    $envs = (& $conda env list --json | ConvertFrom-Json -AsHashtable)["envs"]
    return $envs | Where-Object { (Split-Path -Leaf $_) -eq $EnvName } | Select-Object -First 1
}

if ($conda) {
    $prefix = Get-EnvPrefix

    if (-not $prefix) {
        Write-Host "Creating conda env '$EnvName' (Python $pythonVersion)..." -ForegroundColor Cyan
        Invoke-Native $conda @("create", "-n", $EnvName, "python=$pythonVersion", "-y")
        $prefix = Get-EnvPrefix
    } else {
        Write-Host "Using existing conda env '$EnvName' at $prefix" -ForegroundColor Cyan
    }

    # Make uv install into the conda env whenever it is activated.
    Invoke-Native $conda @("env", "config", "vars", "set", "-n", $EnvName, "UV_PROJECT_ENVIRONMENT=$prefix")
    $env:UV_PROJECT_ENVIRONMENT = $prefix

    # --inexact keeps packages conda installed that uv.lock does not list.
    $syncArgs += @("--inexact", "--python", (Join-Path $prefix "python.exe"))
} else {
    Write-Host "conda not found or disabled; using a uv-managed .venv" -ForegroundColor Yellow
    $syncArgs += @("--python", $pythonVersion)
}

Write-Host "Installing locked dependencies..." -ForegroundColor Cyan
Invoke-Native uv $syncArgs

if (-not (Test-Path (Join-Path $root ".env"))) {
    Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env")
    Write-Host "Created .env from .env.example; add your provider keys." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
if ($conda) { Write-Host "Activate with: conda activate $EnvName" }
else { Write-Host "Run commands with: uv run <command>" }

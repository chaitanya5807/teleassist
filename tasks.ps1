param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "data", "index", "sft-data", "train", "eval", "report", "serve", "test", "lint", "docker-build")]
    [string]$Task = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Task {
    param([string]$Name)

    switch ($Name) {
        "setup" {
            & $Python -m pip install -r requirements.txt
            & $Python -m pip install -e .
        }
        "test" { & $Python -m pytest }
        "lint" { & $Python -m ruff check . }
        "data" {
            $DownloadArgs = @(
                "-m", "teleassist.ingestion.download",
                "--output-dir", (Join-Path $PSScriptRoot "data\raw"),
                "--bundle-path", (Join-Path $PSScriptRoot "data\fallback\wikipedia_telecom.jsonl")
            )
            & $Python @DownloadArgs
            $ChunkArgs = @(
                "-m", "teleassist.ingestion.chunking",
                "--input-dir", (Join-Path $PSScriptRoot "data\fallback\wikipedia_telecom.jsonl"),
                "--manual-dir", (Join-Path $PSScriptRoot "data\raw\manual"),
                "--output", (Join-Path $PSScriptRoot "data\processed\chunks.jsonl"),
                "--config", (Join-Path $PSScriptRoot "configs\default.yaml")
            )
            & $Python @ChunkArgs
        }
        "index" { Write-Host "not implemented until Phase 3" }
        "sft-data" { Write-Host "not implemented until Phase 5" }
        "train" { Write-Host "not implemented until Phase 6" }
        "eval" { Write-Host "not implemented until Phase 7" }
        "report" { Write-Host "not implemented until Phase 7" }
        "serve" { Write-Host "not implemented until Phase 8" }
        "docker-build" { Write-Host "not implemented until Phase 8" }
        default { throw "Unknown task: $Name" }
    }
}

if ($Task) {
    Invoke-Task -Name $Task
} else {
    Write-Host "Usage: .\tasks.ps1 <setup|data|index|sft-data|train|eval|report|serve|test|lint|docker-build>"
    exit 1
}

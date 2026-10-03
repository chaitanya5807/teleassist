param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "data", "index", "sft-data", "train", "eval", "report", "serve", "test", "lint", "docker-build")]
    [string]$Task = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Task {
    param([string]$Name)

    switch ($Name) {
        "setup" {
            python -m pip install -r requirements.txt
            python -m pip install -e .
        }
        "test" { python -m pytest }
        "lint" { python -m ruff check . }
        "data" { Write-Host "not implemented until Phase 2" }
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

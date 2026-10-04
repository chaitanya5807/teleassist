param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "data", "build-index", "search", "ask", "retrieval-eval", "sft-data", "train", "eval", "report", "serve", "test", "lint", "docker-build")]
    [string]$Task = "",
    [Parameter(Position = 1)]
    [string]$Query = "",
    [ValidateSet("bm25", "dense", "hybrid", "hybrid_rerank")]
    [string]$Mode = "hybrid_rerank",
    [string]$EvalSetPath = "",
    [switch]$NoRetrieval,
    [string]$Config = "configs/default.yaml"
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
                "--config", (Join-Path $PSScriptRoot "configs\default.yaml")
            )
            & $Python @DownloadArgs
            $ChunkArgs = @(
                "-m", "teleassist.ingestion.chunking",
                "--input-dir", (Join-Path $PSScriptRoot "data\fallback\wikipedia_telecom.jsonl"),
                "--manual-dir", (Join-Path $PSScriptRoot "data\raw\manual"),
                "--manifest", (Join-Path $PSScriptRoot "data\raw\MANIFEST.json"),
                "--output", (Join-Path $PSScriptRoot "data\processed\chunks.jsonl"),
                "--config", (Join-Path $PSScriptRoot "configs\default.yaml")
            )
            & $Python @ChunkArgs
        }
        "build-index" {
            & $Python -m teleassist.retrieval.pipeline build-index `
                --chunks (Join-Path $PSScriptRoot "data\processed\chunks.jsonl") `
                --index (Join-Path $PSScriptRoot "data\processed\dense_index.npz") `
                --config (Join-Path $PSScriptRoot "configs\default.yaml")
        }
        "search" {
            if (-not $Query) { throw 'Usage: .\tasks.ps1 search "query" -Mode hybrid_rerank' }
            & $Python -m teleassist.retrieval.pipeline search $Query --mode $Mode `
                --chunks (Join-Path $PSScriptRoot "data\processed\chunks.jsonl") `
                --index (Join-Path $PSScriptRoot "data\processed\dense_index.npz") `
                --config (Join-Path $PSScriptRoot "configs\default.yaml")
        }
        "ask" {
            if (-not $Query) { throw 'Usage: .\tasks.ps1 ask "question" -Mode hybrid_rerank [-NoRetrieval] [-Config configs/local_cpu.yaml]' }
            $AskMode = if ($NoRetrieval) { "no_retrieval" } else { $Mode }
            & $Python -m teleassist.generation.rag --query $Query --mode $AskMode `
                --chunks (Join-Path $PSScriptRoot "data\processed\chunks.jsonl") `
                --index (Join-Path $PSScriptRoot "data\processed\dense_index.npz") `
                --config (Join-Path $PSScriptRoot $Config)
        }
        "retrieval-eval" {
            if (-not $EvalSetPath) {
                throw 'Usage: .\tasks.ps1 retrieval-eval -EvalSetPath data/eval/smoke_eval.jsonl'
            }
            & $Python -m teleassist.evaluation.run_retrieval_eval $EvalSetPath --mode $Mode `
                --chunks (Join-Path $PSScriptRoot "data\processed\chunks.jsonl") `
                --index (Join-Path $PSScriptRoot "data\processed\dense_index.npz") `
                --config (Join-Path $PSScriptRoot "configs\default.yaml")
        }
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
    Write-Host "Usage: .\tasks.ps1 <setup|data|build-index|search|retrieval-eval|sft-data|train|eval|report|serve|test|lint|docker-build>"
    exit 1
}

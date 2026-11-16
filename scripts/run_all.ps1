# 一键运行：数据生成 -> 调制识别训练 -> 遥测异常检测训练 -> 评测
# 用法: powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-EpochsMod 15] [-EpochsAE 30]

param(
    [int]$EpochsMod = 15,
    [int]$EpochsAE = 30,
    [int]$TrainPerClass = 1200,
    [int]$EvalPerClass = 200
)

$ErrorActionPreference = "Stop"
$conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
$root = Split-Path -Parent $PSScriptRoot

function Run-Step([string]$step, [string[]]$extra) {
    $args = @("run", "--live-stream", "--cwd", $root, "-n", "signalai", "python", "-m", "signalai", $step) + $extra
    & $conda @args
    if ($LASTEXITCODE -ne 0) { throw "步骤失败: $step" }
}

Run-Step "gen-data" @("--n-train-per-class", "$TrainPerClass", "--n-eval-per-class", "$EvalPerClass")
Run-Step "train-mod" @("--epochs", "$EpochsMod")
Run-Step "train-ae" @("--epochs", "$EpochsAE")
Run-Step "eval" @()
Write-Host "完成。结果见 artifacts\ 与 docs\report.md"

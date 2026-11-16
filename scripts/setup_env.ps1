# 创建 signalai conda 环境（国内网络使用清华镜像）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
$conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
if (-not (Test-Path $conda)) {
    throw "未找到 conda: $conda，请先安装 Miniconda 或修改本脚本中的路径。"
}

& $conda create -n signalai python=3.11 -y `
    --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

& $conda run -n signalai python -m pip install `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
    numpy scipy scikit-learn matplotlib torch pytest

$root = Split-Path -Parent $PSScriptRoot
& $conda run --cwd $root -n signalai python -m pip install -e . --no-build-isolation --no-deps

Write-Host "环境就绪：conda activate signalai"

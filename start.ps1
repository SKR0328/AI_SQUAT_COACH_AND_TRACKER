$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$streamlitConfig = Join-Path $projectRoot ".streamlit"
$matplotlibConfig = Join-Path $projectRoot "work\matplotlib"
$coachRuntime = Join-Path $projectRoot ".runtime\llama\llama-mtmd-cli.exe"
$coachModel = Join-Path $projectRoot "models\Qwen3VL-2B-Instruct-Q4_K_M.gguf"
$visionProjector = Join-Path $projectRoot "models\mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
$ffmpeg = Join-Path $projectRoot ".runtime\ffmpeg\bin\ffmpeg.exe"

if (-not (Test-Path $python)) {
    Write-Error "Project environment not found. Create it with: python -m venv .venv; then install requirements.txt."
}

Push-Location $projectRoot
try {
    New-Item -ItemType Directory -Force -Path $streamlitConfig, $matplotlibConfig | Out-Null
    $env:STREAMLIT_CONFIG_DIR = $streamlitConfig
    $env:MPLCONFIGDIR = $matplotlibConfig
    if (-not ((Test-Path $coachRuntime) -and (Test-Path $coachModel) -and (Test-Path $visionProjector) -and (Test-Path $ffmpeg))) {
        Write-Warning "The local Qwen video-coaching runtime is incomplete. Run '.\.venv\Scripts\python.exe scripts\download_local_video_coach.py' to download it. Pose analysis will still work without that optional runtime."
    }
    & $python -m streamlit run app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
}
finally {
    Pop-Location
}

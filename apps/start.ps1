param([string]$Python = "", [int]$Port = 8501)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $researchPython = Join-Path $projectRoot '.venv-research/Scripts/python.exe'
    $Python = if (Test-Path -LiteralPath $researchPython) { $researchPython } else { 'python' }
}
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Join-Path $projectRoot "src") + [IO.Path]::PathSeparator + $oldPythonPath
    Push-Location -LiteralPath $projectRoot
    try {
        & $Python -m streamlit run (Join-Path $PSScriptRoot "streamlit_app.py") --server.port $Port
        if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }
    } finally { Pop-Location }
} finally { $env:PYTHONPATH = $oldPythonPath }

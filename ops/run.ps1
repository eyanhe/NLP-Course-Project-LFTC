param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$Python = "python",
    [string]$ArtifactsDir = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = (Resolve-Path -LiteralPath $Config).Path
$oldPythonPath = $env:PYTHONPATH
$oldArtifacts = $env:LFTC_ARTIFACTS_DIR
try {
    $env:PYTHONPATH = (Join-Path $projectRoot "src") + [IO.Path]::PathSeparator + $oldPythonPath
    if ($ArtifactsDir) { $env:LFTC_ARTIFACTS_DIR = [IO.Path]::GetFullPath($ArtifactsDir) }
    Push-Location -LiteralPath $projectRoot
    try {
        & $Python -m lftc run --config $configPath
        if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
    } finally { Pop-Location }
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:LFTC_ARTIFACTS_DIR = $oldArtifacts
}

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$authRoot = Join-Path $projectRoot '.superpowers\publish-20261108\auth'
$env:GH_CONFIG_DIR = Join-Path $authRoot 'github'
$env:HF_HOME = Join-Path $authRoot 'huggingface'
$env:PATH = (Join-Path $projectRoot '.tools\git\cmd') + ';' + $env:PATH
$ghExecutable = Join-Path $projectRoot '.tools\gh\bin\gh.exe'
$hfExecutable = Join-Path $projectRoot '.venv-publish\Scripts\hf.exe'
Write-Host '1/2: GitHub - sign in as eyanhe or another account with push access.'
$ErrorActionPreference = 'Continue'
& $ghExecutable auth status 2>$null
$authStatus = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($authStatus -ne 0) {
    & $ghExecutable auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) { throw 'GitHub login did not finish.' }
}
Write-Host '2/2: Hugging Face - sign in as Eyanhe with write access to LFTC_Dataset_And_Checkpoints.'
Write-Host 'Create/select a write token at https://huggingface.co/settings/tokens and enter it only in the hidden terminal prompt.'
& $hfExecutable auth login
if ($LASTEXITCODE -ne 0) { throw 'Hugging Face login did not finish.' }
Write-Host 'Both logins finished. Tell Codex that the repository logins are complete.'

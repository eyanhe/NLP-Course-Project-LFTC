$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gitPath = Join-Path $projectRoot '.tools/git/cmd'
$ghExecutable = Join-Path $projectRoot '.tools/gh/bin/gh.exe'
$hfExecutable = Join-Path $projectRoot '.venv-publish/Scripts/hf.exe'
if (!(Test-Path -LiteralPath $ghExecutable) -or !(Test-Path -LiteralPath $hfExecutable)) {
    throw 'Publishing tools are not ready yet. Please retry after the preparation finishes.'
}
$env:PATH = $gitPath + ';' + $env:PATH
Write-Host '1/2 GitHub: sign in with an account that can push to eyanhe/NLP-Course-Project-LFTC.'
& $ghExecutable auth login --hostname github.com --git-protocol https --web
if ($LASTEXITCODE -ne 0) { throw 'GitHub login did not finish.' }
& $ghExecutable auth setup-git --hostname github.com
if ($LASTEXITCODE -ne 0) { throw 'GitHub Git credential setup did not finish.' }
Write-Host '2/2 Hugging Face: enter a write token for Eyanhe/LFTC_Dataset_And_Checkpoints at the hidden prompt.'
Write-Host 'Token settings: https://huggingface.co/settings/tokens'
& $hfExecutable auth login
if ($LASTEXITCODE -ne 0) { throw 'Hugging Face login did not finish.' }
Write-Host 'Both logins finished. Return to the chat and say the logins are complete.'

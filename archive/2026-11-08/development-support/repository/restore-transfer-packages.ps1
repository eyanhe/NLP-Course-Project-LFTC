param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$workspaceRoot = [IO.Directory]::GetParent($projectRoot).FullName
$manifestPath = Join-Path $projectRoot 'docs\repository-archive-manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
function Resolve-Within([string]$basePath, [string]$relativePath) {
    if ([IO.Path]::IsPathRooted($relativePath)) { throw "Absolute path rejected: $relativePath" }
    $resolved = [IO.Path]::GetFullPath((Join-Path $basePath $relativePath))
    $prefix = $basePath.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Path escapes root: $relativePath" }
    return $resolved
}
# Validate every entry before changing any path.
$actions = foreach ($entry in $manifest.files) {
    $target = Resolve-Within $projectRoot $entry.original_path
    $source = Resolve-Within $workspaceRoot $entry.archive_path
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Different existing target: $target" }
        continue
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Archive missing: $source" }
    if ((Get-Item -LiteralPath $source).Length -ne $entry.bytes) { throw "Size mismatch: $source" }
    Write-Host "Checking $($entry.original_path)"
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "SHA256 mismatch: $source" }
    [pscustomobject]@{ Source=$source; Target=$target }
}
if ($VerifyOnly) { Write-Host "Verified $(@($manifest.files).Count) packages; no files moved."; return }
foreach ($action in $actions) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($action.Target)) | Out-Null
    Move-Item -LiteralPath $action.Source -Destination $action.Target
}
Write-Host "Restored $(@($actions).Count) packages to their original project paths."

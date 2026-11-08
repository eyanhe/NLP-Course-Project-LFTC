param([switch]$SkipDownload, [switch]$AppOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
if ($AppOnly) {
    python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
    exit $LASTEXITCODE
}
if (-not $SkipDownload) {
    python -u download_assets.py
    if ($LASTEXITCODE -ne 0) { throw 'Asset download failed' }
}
python -u pipeline.py all
if ($LASTEXITCODE -ne 0) { throw 'Experiment pipeline failed' }
python -u extension.py all
if ($LASTEXITCODE -ne 0) { throw 'V2 experiments failed' }
python -u hard_benchmark.py all
if ($LASTEXITCODE -ne 0) { throw 'MultiNLI experiments failed' }
python -u imbalance.py all
if ($LASTEXITCODE -ne 0) { throw 'Imbalance experiments failed' }
python -u make_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
python -u make_report_v2.py
if ($LASTEXITCODE -ne 0) { throw 'V2 report generation failed' }
python -u make_report_hard.py
if ($LASTEXITCODE -ne 0) { throw 'MultiNLI report generation failed' }
python -u make_report_imbalance.py
if ($LASTEXITCODE -ne 0) { throw 'Imbalance report generation failed' }
python -m unittest -v test_project test_extension test_hard test_imbalance
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }

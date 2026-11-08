"""Create a compact code/results archive; keep multi-GB weights in the local project."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent

def main():
    required=['results/evaluation.json','results/benchmark.json','report/experiment-report.md',
              'report/project-presentation.pptx','report/narration-script-zh.md','README.md',
              'results-v2/evaluation.json','results-v2/benchmark.json','report/classification-report-v2.md',
              'report/classification-presentation-v2.pptx','results-hard/evaluation.json',
              'results-hard/benchmark.json','report/hard-benchmark-report.md','report/hard-benchmark-supplement.pptx',
              'results-imbalance/evaluation.json','results-imbalance/comparison.csv','report/imbalance-report.md',
              'report/final-report.md','report/final-synthesis-zh.md','report/final-presentation.pptx']
    missing=[name for name in required if not (ROOT/name).is_file()]
    if missing: raise FileNotFoundError(f'Complete deliverables first: {missing}')
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file(): continue
        relative=p.relative_to(ROOT)
        if relative.parts[0] in {'.cache','.venv','__pycache__','assets','.streamlit'}: continue
        if '__pycache__' in relative.parts: continue
        if p.name.startswith('~$'): continue  # Office/WPS locks are not deliverables.
        if relative.parts[0] in {'models','models-v2','models-hard','models-imbalance'} and p.name!='metadata.json': continue
        if relative.parts[0] in {'data','data-v2','data-hard','data-imbalance'} and p.suffix=='.jsonl': continue
        if p.name in {'run.log','run-v2.log','run-hard.log','run-imbalance.log','bundle-manifest.json'} or p.suffix in {'.pyc','.zip','.tmp'}: continue
        files.append(p)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Compact code, results and presentation package. Raw assets, prepared texts and model weights remain in local project; archive users must download assets and reproduce training.',
        'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]}
    manifest_path=ROOT/'bundle-manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),'utf-8');files.append(manifest_path)
    archive=ROOT.parent/'sentiment-project-submission.zip'
    pending=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files): z.write(p,'sentiment-project/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(pending) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f'Archive verification failed: {bad}')
    pending.replace(archive)
    print(f'Created {archive.name}: {len(files)} files, {archive.stat().st_size:,} bytes; CRC verification passed.')

if __name__=='__main__':main()

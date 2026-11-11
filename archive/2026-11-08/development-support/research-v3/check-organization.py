"""Check organization without loading models, evaluating Q, or training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import re
from urllib.parse import unquote

PROJECT=Path(__file__).resolve().parents[2]
STATE=PROJECT/'.superpowers/repo-organization-20261108'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def protected():
    analysis=PROJECT/'research-v3/reports/overnight/analysis-v2/analysis-data.json'
    data=json.loads(analysis.read_text(encoding='utf-8'));paths={PROJECT/p for p in data['source_sha256']}
    paths.add(analysis)
    locks=[PROJECT/'research-v3/reports/overnight/convergence-v1/cluster-snapshot/training/convergence-v1/scoped-evaluation.freeze.json']
    locks += [PROJECT/f'research-v3/reports/overnight/supplement-q-v1/cluster-snapshot/reports/overnight/supplement-q-v1/{stage}/evaluation.freeze.json' for stage in ['S1','S3']]
    for lock in locks:
        body=json.loads(lock.read_text(encoding='utf-8'));paths.add(lock)
        paths.update(PROJECT/'research_v3'/p for p in body['source_inventory'])
        paths.update(PROJECT/'research-v3'/p for p in body['files'])
    return {str(p.resolve().relative_to(PROJECT)):digest(p) for p in paths if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['snapshot','verify']);args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True);baseline=STATE/'protected-before.json'
    if args.mode=='snapshot':
        if baseline.exists():raise FileExistsError('Keep the original organization baseline')
        value={'at_utc':datetime.now(timezone.utc).isoformat(),'files':protected()}
        baseline.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'protected_files':len(value['files'])}));return
    before=json.loads(baseline.read_text(encoding='utf-8'))['files']
    changed=[p for p,h in before.items() if not (PROJECT/p).is_file() or digest(PROJECT/p)!=h]
    docs=['README.md','research-v3/README.md','research-v3/RESEARCH.md','research_v3/README.md',
        'docs/superpowers/README.md','docs/PROJECT-HISTORY.md','report/README.md','outputs/README.md',
        'research-v3/reports/README.md','research-v3/reports/overnight/README.md',
        'research-v3/reports/overnight/convergence-v1/README.md',
        'research-v3/reports/overnight/supplement-1-4/README.md',
        'research-v3/reports/overnight/supplement-q-v1/README.md',
        'research-v3/reports/overnight/analysis-v2/README.md',
        'research-v3/reports/overnight/results-analysis-v1-zh.md','development-support/research-v3/README.md',
        'research-v3/reports/overnight/complete-atlas-v1/README.md','docs/RESEARCH-CONTINUITY.md']
    broken=[];links=0
    for relative in docs:
        file=PROJECT/relative
        if not file.exists():broken.append((relative,'missing document'));continue
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            target=target.strip().strip('<>')
            if re.match(r'^(?:https?|mailto|data):',target) or target.startswith('#'):continue
            target=unquote(target.split('#')[0]);links+=1
            resolved=Path(target) if re.match(r'^[A-Za-z]:',target) else file.parent/target
            if not resolved.exists():broken.append((relative,target))
    inventory=json.loads((PROJECT/'research-v3/reports/overnight/convergence-v1/experiment-inventory-current.json').read_text(encoding='utf-8'))
    assert inventory['complete_total']==428 and inventory['current_research_complete']==232
    assert inventory['current_research_with_V']==226 and inventory['formal_test_complete']==42
    result={'protected_files':len(before),'changed_protected_files':changed,'checked_documents':len(docs),
        'local_links':links,'broken_links':broken,'training_records':232,'formal_Q_models':42}
    (STATE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    assert not changed and not broken


if __name__=='__main__':main()

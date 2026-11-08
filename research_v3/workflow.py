"""Post-anchor commands, isolated from the source of an active anchor worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .batch import run_resolved_batch
from .config import PROJECT_ROOT,load_protocol,output_root
from .diagnostics import diagnostics_code_hash
from .execution import authorize_phase
from .freeze import _verify_pilot,freeze_stage,verify_anchor_choice,verify_freeze
from .matrix import verify_design_lock
from .orchestrator import _ids_hash
from .registry import build_contrast_manifest,build_run_manifest
from .storage import atomic_write_json,bounded_path,read_json
from .train import learner_code_hash


def _save_immutable(root,name,value):
    path=root/name
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Existing resolved evidence differs: {name}')
    else:atomic_write_json(path,value,root)


def finalize_selection(p: dict,root: Path):
    root=Path(root).resolve();lock=verify_design_lock(root)
    design=read_json(root/'matrix/design_manifest.json')
    if lock['bindings']['protocol_hash']!=p['_protocol_hash'] or design['protocol_hash']!=p['_protocol_hash']:
        raise ValueError('Selection protocol/design identity changed')
    if (root/'selection.lock.json').exists():return verify_freeze('selection',root)
    selections=read_json(root/'selections.json');choices=selections.get('choices',{})
    if selections.get('design_hash')!=design['design_hash']:raise ValueError('Anchor selection design identity changed')
    pilots=selections.get('discovery_pilots',{})
    for node in design['discovery_pilot_nodes']:
        if pilots.get(node['pilot_id'],{}).get('status')!='complete':
            raise PermissionError(f'Real human D pilot is pending: {node["pilot_id"]}')
    files={}
    for node in design['selection_nodes']:
        if node['node_id'] not in choices:raise PermissionError('All declared anchor selections must finish')
        verify_anchor_choice(root,node,choices[node['node_id']],files)
    for node in design['discovery_pilot_nodes']:
        _verify_pilot(root,node,pilots[node['pilot_id']],choices,files)
    prepared=read_json(root/'data/prepared_manifest.json');assets=read_json(root/'assets/assets.lock.json')
    import pyarrow.parquet as pq
    select_hashes={}
    for dataset,meta in prepared['datasets'].items():
        ids=pq.read_table(bounded_path(root/meta['partitions']['select']['file'],root),columns=['sample_id']).column(0).to_pylist()
        select_hashes[dataset]=_ids_hash(ids)
    manifest=build_run_manifest(design,selections,prepared,p,learner_code_hash(),
        diagnostics_code_hash(),select_hashes,asset_hash=assets['asset_hash'])
    contrasts=build_contrast_manifest(design,manifest,prepared,p,pilots)
    _save_immutable(root,'run_manifest.json',manifest)
    _save_immutable(root,'contrasts.json',contrasts)
    return freeze_stage('selection',root,selections)


def execute_workflow(command,p,root,authorization=None):
    root=Path(root).resolve()
    if command=='finalize-selection':return finalize_selection(p,root)
    if authorization is None:raise PermissionError('An execution authorization file is required')
    if command=='train-batch':return run_resolved_batch(p,root,authorization)
    if command=='freeze-evaluation':
        authorize_phase(p,root,authorization,'main')
        if (root/'evaluation.freeze.json').exists():return verify_freeze('evaluation',root)
        from .thresholds import freeze_thresholds
        freeze_thresholds(root,read_json(root/'run_manifest.json'),read_json(root/'data/prepared_manifest.json'))
        return freeze_stage('evaluation',root,{})
    raise ValueError('Unknown post-anchor workflow command')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['finalize-selection','train-batch','freeze-evaluation'])
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/research-v3.json')
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args(argv)
    try:
        p=load_protocol(args.config)
        result=execute_workflow(args.command,p,output_root(p),args.authorization)
        print(json.dumps({k:result[k] for k in ['stage','lock_hash','complete','status',
            'completed_fit_count','total_fit_count'] if k in result},ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'blocked' if isinstance(exc,PermissionError) else 'failed',
            'error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())

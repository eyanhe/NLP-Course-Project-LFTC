"""Independent, exploratory CPU supplements; frozen learners and requests stay intact.

Only the worker entrypoint may do heavy work, and it requires a Slurm allocation.
No official-test loader or human-review writer is called here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json, hash_file, read_json
from .types import RunSpec

RELATIVE = Path('reports/overnight/supplement-1-4')
START_CUTOFF = datetime(2026,11,8,1,0,tzinfo=timezone.utc)
HARD_STOP = datetime(2026,11,8,2,0,tzinfo=timezone.utc)


def relocated_assets(locked,project):
    from .overnight_training import runtime_assets
    return runtime_assets(locked,project)


def cpu_entry(entry, supplement, **overrides):
    result=copy.deepcopy(entry)
    source=result['run']
    if source['strict_no_val']:raise ValueError('Supplement cannot change a strict fewshot request')
    hp={**source['hparams'],'device':'cpu','cpu_threads':8,'micro_batch':4,
        'supplement_id':supplement,**overrides}
    if hp['role_ids_hash']['discovery']!=source['hparams']['role_ids_hash']['discovery']:
        raise ValueError('Supplement changed fitting roles')
    run=replace(RunSpec(**source),run_id='',hparams=hp)
    run=replace(run,run_id=canonical_hash(asdict(run)))
    result.update(run=asdict(run),run_id=run.run_id,artifact_id=run.run_id,
                  expected_model_request_hash=canonical_hash(asdict(run)),
                  source_run_id=source['run_id'],supplement_id=supplement)
    return result


def paired_trajectory(small, large):
    def keyed(rows):
        result={row['sample_id']:row for row in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate trajectory sample IDs')
        return result
    a=keyed(small);b=keyed(large)
    if set(a)!=set(b):raise ValueError('Trajectory sample IDs differ')
    counts=dict(persistent_error=0,corrected=0,new_error=0,always_correct=0);rows=[]
    for sid in sorted(a):
        x,y=a[sid],b[sid]
        if x['group_id']!=y['group_id'] or set(x['dataset_labels'])!=set(y['dataset_labels']):
            raise ValueError('Trajectory group/gold changed')
        ac=set(x['prediction'])==set(x['dataset_labels']);bc=set(y['prediction'])==set(y['dataset_labels'])
        key=('always_correct' if bc else 'new_error') if ac else ('corrected' if bc else 'persistent_error')
        counts[key]+=1
        rows.append(dict(sample_id=sid,group_id=x['group_id'],dataset_labels=x['dataset_labels'],
                         small_prediction=x['prediction'],large_prediction=y['prediction'],trajectory=key))
    return dict(counts=counts,rows=rows,definition='exact dataset-label set at fixed zero margin/logit threshold; not semantic human judgment')


def check_draft(pair):
    from .interventions import english_words
    source=english_words(pair['original_text']);a=english_words(pair['intervention_text']);b=english_words(pair['control_text'])
    if not 5<=len(source)<=60 or len(source)!=len(a) or len(source)!=len(b):
        raise ValueError('Word count/substitution-only constraint failed')
    aa=[(x,y) for x,y in zip(source,a) if x!=y];bb=[(x,y) for x,y in zip(source,b) if x!=y]
    cues=set(pair['cue_words'])
    if not 1<=len(aa)<=3 or len(aa)!=len(bb):raise ValueError('Unmatched edit count')
    if any(x not in cues for x,y in aa) or any(x in cues for x,y in bb):
        raise ValueError('Cue/noncue arm constraint failed')
    # This only checks structure. It never asserts label preservation or content-word status.
    return dict(edit_count=len(aa),cue_replacements=aa,control_replacements=bb,
                human_semantic_review='pending')


def prepare_manifest(root):
    root=Path(root);base=read_json(root/'training/overnight/run-plan.json')
    extension=read_json(root/'training/overnight/extension-run-plan.json')
    design={r['condition_id']:r for r in read_json(root/'matrix/design_manifest.json')['conditions']}
    primary=[row for row in base['runs'] if row['priority']==0 and row['data_seed']==2026
             and design[row['entry']['condition_id']]['experiment_id']=='E1']
    fits=[]
    for dataset,n in [('banking77',1000),('go_emotions',6000)]:
        for model,rates,budgets in [('bilstm',[.001,.0003],[10,20]),('distilbert',[2e-5,5e-5],[5,10])]:
            candidates=[r for r in primary if r['dataset_id']==dataset and r['model_id']==model and r['training_rows']==n]
            if len(candidates)!=1:raise ValueError('Primary fitting source not unique')
            for lr in rates:
                for epochs in reversed(budgets):
                    entry=cpu_entry(candidates[0]['entry'],'S1',lr=lr,max_epochs=epochs,epochs=epochs)
                    fits.append(dict(supplement_id='S1',dataset_id=dataset,model_id=model,training_rows=n,
                                     lr=lr,epochs=epochs,entry=entry))
    m3=[r for r in extension['runs'] if r['dataset_id']=='banking77' and r['model_id'] in {'fixed_tfidf_svm','frozen_generic_distilbert_head'}]
    if len(m3)!=8:raise ValueError('Expected complete M3 representation by 2x2 factorial')
    for row in m3:
        meta=design[row['entry']['condition_id']]['metadata']
        fits.append(dict(supplement_id='S3',dataset_id='banking77',model_id=row['model_id'],
                         training_rows=row['training_rows'],support=meta['support'],risk=meta['risk'],
                         entry=cpu_entry(row['entry'],'S3')))
    sources=[r for r in primary if r['model_id'] in {'tfidf_svm','bilstm','distilbert'} and r['training_rows'] in {1000,6000}]
    if len(sources)!=24:raise ValueError('Expected 4x3x2 matched checkpoint requests')
    body=dict(scope='supplement_1_4_exploratory',official_test_predictions_allowed=False,
              training_seed=42,data_seed=2026,single_seed_limitation=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
              start_cutoff_utc=START_CUTOFF.isoformat(),hard_stop_utc=HARD_STOP.isoformat(),
              original_manifest_sha256=hash_file(root/'training/fixed-main-run-manifest.json'),
              source_plan_sha256=hash_file(root/'training/overnight/run-plan.json'),
              source_extension_sha256=hash_file(root/'training/overnight/extension-run-plan.json'),
              driver_sha256=hash_file(Path(__file__)),fits=fits,trajectory_sources=sources,
              interpretation='S1 budget includes its matching linear decay schedule; V-selected exploratory sensitivity, not convergence proof or Q generalization')
    manifest={**body,'manifest_hash':canonical_hash(body)}
    dest=root/RELATIVE/'manifest.json'
    if dest.exists():
        old=read_json(dest)
        if old['driver_sha256']!=body['driver_sha256']:raise ValueError('Existing supplemental driver changed')
        return old
    atomic_write_json(dest,manifest,root)
    return manifest


def score_checkpoint_cpu(output,inputs):
    """Authenticate artifacts, then override only the in-memory execution device."""
    from .train import _verified_bundle,TRADITIONAL,_neural_scores
    status,bundle=_verified_bundle(Path(output))
    if bundle['model_id'] in TRADITIONAL:
        return bundle['model'].scores(bundle['features'].transform(inputs)),status
    import torch
    from .models import build_model
    hp={**bundle['hparams'],'device':'cpu','micro_batch':4,'cpu_threads':8}
    # Runtime path relocation leaves authenticated bytes and declared asset_hash intact.
    assets=relocated_assets(bundle['assets'],Path(__file__).resolve().parents[1]) if 'files' in bundle['assets'] else copy.deepcopy(bundle['assets'])
    torch.set_num_threads(8);model_id=bundle['model_id'];tokenizer=None
    if model_id=='bilstm':
        from safetensors.torch import load_file
        model=build_model(model_id,bundle['task'],bundle['label_count'],assets,hp,bundle['features'])
        model.load_state_dict(load_file(str(Path(output)/'checkpoint/model.safetensors')))
    else:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        model=AutoModelForSequenceClassification.from_pretrained(Path(output)/'checkpoint',local_files_only=True,attn_implementation='sdpa')
        family='modernbert' if model_id=='modernbert' else 'distilbert'
        tokenizer=AutoTokenizer.from_pretrained(assets['models'][family]['directory'],local_files_only=True)
    model.to('cpu')
    scores=_neural_scores(model,model_id,tuple(inputs),bundle['features'],tokenizer,hp,bundle['constants'])
    del model,bundle;gc.collect()
    return scores,status


def score_trajectories(root,manifest):
    from .orchestrator import load_case
    from .train import target_matrix
    from .metrics import metric_bundle
    folder=root/RELATIVE;frame=read_json(root/'reports/overnight/failure-frame-plan.json')
    prepared=read_json(root/'data/prepared_manifest.json')['datasets'];records={};missing=[];metrics=[]
    prediction_dir=folder/'S2-predictions';prediction_dir.mkdir(exist_ok=True,parents=True)
    for source in manifest['trajectory_sources']:
        ds=source['dataset_id'];n=source['training_rows'];model=source['model_id'];entry=source['entry'];rid=entry['run_id']
        status_path=root/'runs'/rid/'status.json'
        if not status_path.exists() or read_json(status_path).get('status')!='complete':
            missing.append(dict(dataset_id=ds,model_id=model,n=n,run_id=rid,reason='checkpoint_not_complete'));continue
        ids=frame['frames'][ds]['sample_ids'];meta=prepared[ds]
        f=frame['frames'][ds]
        if hash_file(root/f['source_file'])!=f['source_sha256']:raise ValueError('D frame source bytes changed')
        output=prediction_dir/f'{rid}.json'
        if output.exists():
            saved=read_json(output)
            if saved['status_sha256']!=hash_file(status_path) or saved['frame_ids_hash']!=canonical_hash(ids):
                raise ValueError('Cached inference checkpoint or D frame changed')
            rows=saved['rows'];metric=saved['metrics']
        else:
            data,inputs=load_case(root,meta,(),selection_ids=(),discovery_ids=ids)
            byid={row.sample_id:row for row in data.partitions['discovery']};ordered=tuple(inputs[sid] for sid in ids)
            if {byid[sid].group_id for sid in ids}!=set(f['group_ids']):raise ValueError('D frame groups changed')
            started=time.monotonic();scores,status=score_checkpoint_cpu(root/'runs'/rid,ordered)
            if status['run']!=entry['run']:raise ValueError('Frozen checkpoint request changed')
            rows=[]
            for i,sid in enumerate(ids):
                gold=list(byid[sid].labels)
                prediction=[int(scores[i].argmax())] if data.task=='single_label' else np.flatnonzero(scores[i]>=0).astype(int).tolist()
                rows.append(dict(sample_id=sid,group_id=byid[sid].group_id,dataset_labels=gold,prediction=prediction,
                    false_negative_labels=sorted(set(gold)-set(prediction)),false_positive_labels=sorted(set(prediction)-set(gold)),
                    scores=scores[i].tolist(),input_text_hash=inputs[sid].text_hash))
            y=target_matrix(tuple(byid[sid] for sid in ids),len(meta['label_names']),data.task)
            metric=metric_bundle(y,scores,data.task,status['score_kind'])
            atomic_write_json(output,dict(run_id=rid,source_partition='discovery',inference_device='cpu',
                original_training_device=entry['run']['hparams']['device'],cpu_gpu_bitwise_equivalence_claimed=False,
                status_sha256=hash_file(status_path),frame_ids_hash=canonical_hash(ids),rows=rows,metrics=metric,
                seconds=time.monotonic()-started,human_review_status='pending'),root)
        records[(ds,model,n)]=rows
        metrics.append(dict(dataset_id=ds,model_id=model,n=n,run_id=rid,metrics=metric))
        print(json.dumps(dict(event='S2_scored',dataset_id=ds,model_id=model,n=n,rows=len(rows))),flush=True)
    trajectories=[];overlaps=[]
    for ds in prepared:
        for model in ['tfidf_svm','bilstm','distilbert']:
            if (ds,model,1000) in records and (ds,model,6000) in records:
                result=paired_trajectory(records[(ds,model,1000)],records[(ds,model,6000)])
                trajectories.append(dict(dataset_id=ds,model_id=model,**result))
        for n in [1000,6000]:
            available=[m for m in ['tfidf_svm','bilstm','distilbert'] if (ds,m,n) in records]
            if not available:continue
            keyed={m:{r['sample_id']:r for r in records[(ds,m,n)]} for m in available}
            ids=set(next(iter(keyed.values())))
            if any(set(rows)!=ids for rows in keyed.values()):raise ValueError('Cross-model D IDs differ')
            pattern=Counter()
            for sid in sorted(ids):
                values=[keyed[m][sid] for m in available]
                if len({(v['group_id'],tuple(sorted(v['dataset_labels']))) for v in values})!=1:
                    raise ValueError('Cross-model group/gold changed')
                pattern[tuple(m for m in available if set(keyed[m][sid]['prediction'])!=set(keyed[m][sid]['dataset_labels']))]+=1
            overlaps.append(dict(dataset_id=ds,n=n,available_models=available,patterns=[dict(error_models=list(k),count=v) for k,v in sorted(pattern.items())]))
    summary=dict(status='complete' if not missing else 'partial_missing_checkpoints',missing=missing,
                 complete_checkpoint_predictions=len(records),matched_trajectories=trajectories,cross_model_overlap=overlaps,metrics=metrics,
                 scope='fixed unreserved D frames, BANKING77 only23; conditional exploratory results, no population inference',
                 multilabel_note='Exact-set trajectories use fixed threshold0; AP is also reported separately and is threshold-free',
                 official_test_predictions_generated=False,human_semantic_reviews_completed=0)
    atomic_write_json(folder/'S2-summary.json',summary,root)
    return summary


def run_fits(root,manifest,worker):
    from .batch import fit_declared_run
    prepared=read_json(root/'data/prepared_manifest.json');assets=relocated_assets(read_json(root/'assets/assets.lock.json'),root.parent)
    if worker=='bank':
        selected=[r for r in manifest['fits'] if r['supplement_id']=='S3']+[r for r in manifest['fits'] if r['supplement_id']=='S1' and r['dataset_id']=='banking77']
    else:selected=[r for r in manifest['fits'] if r['supplement_id']=='S1' and r['dataset_id']=='go_emotions']
    folder=root/RELATIVE;done=[]
    # A shared worker lease avoids cross-node ambiguity in per-run PID lease checks.
    from .execution import run_lease
    with run_lease(folder/f'worker-{worker}',manifest['manifest_hash']):
        for row in selected:
            if datetime.now(timezone.utc)>=START_CUTOFF:
                atomic_write_json(folder/f'worker-{worker}-status.json',dict(status='start_cutoff_reached',completed=done),root);return
            rid=row['entry']['run_id']
            event=dict(status='fitting',supplement_id=row['supplement_id'],dataset_id=row['dataset_id'],model_id=row['model_id'],
                       run_id=rid,completed=done,job_id=os.environ['SLURM_JOB_ID'],updated_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_write_json(folder/f'worker-{worker}-status.json',event,root);print(json.dumps(event),flush=True)
            result=fit_declared_run(row['entry'],root,prepared,assets)
            if result['status']!='complete' or not result.get('reload_verified'):raise ValueError('Supplement fit failed verification')
            done.append(rid);print(json.dumps(dict(event='fit_complete',run_id=rid,validation=result['validation'],seconds=result.get('total_seconds'))),flush=True)
            write_fit_summary(root,manifest);gc.collect()
        atomic_write_json(folder/f'worker-{worker}-status.json',dict(status='complete',completed=done,job_id=os.environ['SLURM_JOB_ID']),root)


def write_fit_summary(root,manifest):
    records=[]
    for request in manifest['fits']:
        path=root/'runs'/request['entry']['run_id']/'status.json'
        value=read_json(path) if path.exists() else {'status':'not_started'}
        if value.get('status')=='complete' and value.get('run')!=request['entry']['run']:
            raise ValueError('Supplement result request identity changed')
        record={k:v for k,v in request.items() if k!='entry'}
        record.update(run_id=request['entry']['run_id'],source_run_id=request['entry']['source_run_id'],
                      status=value['status'],validation=value.get('validation'),selected_epoch=value.get('selected_epoch'),
                      epochs_completed=value.get('epochs_completed'),optimizer_updates=value.get('optimizer_updates'),
                      sample_weight_sum=value.get('sample_weight_sum'),reference_n=value.get('reference_n'),
                      error=value.get('error'),seconds=value.get('total_seconds'))
        records.append(record)
    atomic_write_json(root/RELATIVE/'fit-summary.json',dict(records=records,counts=dict(Counter(r['status'] for r in records)),
                      updated_at_utc=datetime.now(timezone.utc).isoformat(),official_test_predictions_generated=False),root)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','summarize'])
    parser.add_argument('--root',type=Path,default=Path('research-v3'));parser.add_argument('--worker',choices=['bank','emotion'])
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':
        m=prepare_manifest(root);print(json.dumps(dict(fits=len(m['fits']),trajectory_checkpoints=len(m['trajectory_sources']),manifest_hash=m['manifest_hash'])));return
    manifest=read_json(root/RELATIVE/'manifest.json')
    if manifest['manifest_hash']!=canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'}):raise ValueError('Supplement manifest changed')
    if hash_file(Path(__file__))!=manifest['driver_sha256']:raise ValueError('Supplement driver changed')
    if hash_file(root/'training/fixed-main-run-manifest.json')!=manifest['original_manifest_sha256']:raise ValueError('Original requests changed')
    if args.action=='summarize':write_fit_summary(root,manifest);return
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Heavy supplement work requires Slurm')
    if args.worker=='bank':score_trajectories(root,manifest)
    run_fits(root,manifest,args.worker)


if __name__=='__main__':main()

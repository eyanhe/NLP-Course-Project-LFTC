"""Official-test inference only after the globally complete evaluation freeze."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import canonical_hash
from .freeze import verify_freeze
from .metrics import metric_bundle
from .storage import atomic_write_json,bounded_path,hash_file,read_json
from .train import score_saved
from .types import Example,PredictionRow,RunSpec,VisibleInput


def test_exposure_status(dataset_id: str) -> str:
    if dataset_id in {'imdb','rotten_tomatoes','yelp','ag_news'}:
        return 'historical_Q_seen_exploratory_replication'
    if dataset_id in {'go_emotions','banking77','yahoo_answers_topics'}:
        return 'new_source_first_v3_evaluation'
    return 'fixture_or_external_source'


def validate_prediction_rows(rows: tuple[PredictionRow,...],label_count: int,task: str) -> None:
    seen=set();multilabel=task in {'multilabel','multi_label'}
    for row in rows:
        key=row.run_id,row.sample_id
        if key in seen:raise ValueError('Prediction rows must have unique run/sample IDs; duplicate found')
        seen.add(key)
        if not row.labels or any(c<0 or c>=label_count for c in row.labels):
            raise ValueError('Prediction record has empty or invalid gold labels')
        if tuple(sorted(set(row.labels)))!=tuple(row.labels) or tuple(sorted(set(row.prediction)))!=tuple(row.prediction):
            raise ValueError('Gold/predicted labels must be unique canonical indices')
        if any(c<0 or c>=label_count for c in row.prediction) or (not multilabel and len(row.prediction)!=1):
            raise ValueError('Invalid predicted label set')
        if row.score_kind not in {'probability','logit','svm_margin'}:
            raise ValueError('Unknown prediction score kind')
        scores=np.asarray(row.scores,dtype=float)
        if scores.shape!=(label_count,) or not np.isfinite(scores).all():
            raise ValueError('Prediction score vector has incorrect full-label shape or nonfinite values')
        if row.score_kind=='probability' and ((scores<0).any() or (scores>1).any()):
            raise ValueError('Probability score outside [0,1]')
        if not row.run_id or not row.sample_id or not row.group_id or not row.input_hash:
            raise ValueError('Prediction provenance is incomplete')


def _context(run,examples,inputs,root):
    import pyarrow.parquet as pq
    design=read_json(root/'matrix/design_manifest.json')
    conditions=[row for row in design['conditions'] if row['condition_id']==run.condition_id]
    if len(conditions)!=1:raise ValueError('Official evaluation requires a declared design condition')
    dataset=conditions[0]['dataset_id'];meta=read_json(root/'data/prepared_manifest.json')['datasets'][dataset]
    if run.input_hash!=meta['inputs']['input_hash']:raise ValueError('Run common-input identity changed')
    source=pq.read_table(bounded_path(root/meta['partitions']['test']['file'],root),
                         columns=['sample_id','labels','group_id']).to_pylist()
    by_id={row['sample_id']:row for row in source}
    if len(examples)!=meta['partitions']['test']['rows'] or len({row.sample_id for row in examples})!=len(examples) or {row.sample_id for row in examples}!=set(by_id):
        raise ValueError('Official evaluation must contain every unique frozen Q sample')
    frozen=pq.read_table(bounded_path(root/meta['inputs']['file'],root),
        columns=['sample_id','text','start_char','end_char','content_tokens','full_tokens','text_hash'],
        filters=[('partition','=','test')]).to_pylist()
    expected={row['sample_id']:VisibleInput(**row) for row in frozen}
    ordered=tuple(sorted(examples,key=lambda row:row.sample_id));visible=[]
    for row in ordered:
        if row.official_split!='test' or tuple(row.labels)!=tuple(by_id[row.sample_id]['labels']) or row.group_id!=by_id[row.sample_id]['group_id']:
            raise ValueError('Official evaluation sample/gold/group identity changed')
        value=inputs[row.sample_id]
        if value!=expected.get(row.sample_id) or value.text_hash!=hashlib.sha256(value.text.encode()).hexdigest():
            raise ValueError('Common visible input changed from frozen source')
        visible.append(value)
    return {**meta,'dataset_id':dataset},ordered,tuple(visible)


def _read_predictions(path):
    import pyarrow.parquet as pq
    return tuple(PredictionRow(row['run_id'],row['sample_id'],row['group_id'],tuple(row['labels']),
        tuple(row['prediction']),tuple(row['scores']),row['score_kind'],row['input_hash'])
        for row in pq.read_table(path).to_pylist())


def _metrics(rows,meta):
    scores=np.asarray([row.scores for row in rows],dtype=np.float64)
    if meta['task']=='single_label':y=np.asarray([row.labels[0] for row in rows])
    else:
        y=np.zeros_like(scores)
        for i,row in enumerate(rows):y[i,list(row.labels)]=1
    return metric_bundle(y,scores,meta['task'],rows[0].score_kind,
        label_mask=meta['partitions']['test'].get('ap_mask') if meta['task']!='single_label' else None)


def predict_run(run: RunSpec,examples: tuple[Example,...],inputs: dict[str,VisibleInput],root: Path) -> tuple[PredictionRow,...]:
    root=Path(root).resolve();freeze=verify_freeze('evaluation',root)
    manifest=read_json(root/'run_manifest.json')
    if run.run_id not in {row.get('artifact_id',row['run_id']) for row in manifest['runs']}:
        raise ValueError('This checkpoint is not part of the declared globally frozen evaluation')
    output=bounded_path(root/'runs'/run.run_id,root/'runs')
    status=read_json(output/'status.json')
    if canonical_hash(status['run'])!=canonical_hash(asdict(run)):
        raise ValueError('Model request identity differs from the frozen checkpoint')
    meta,ordered,visible=_context(run,examples,inputs,root)
    if not ordered:raise ValueError('Official evaluation rows cannot be empty')
    identity=canonical_hash({'run':asdict(run),'evaluation_freeze_hash':freeze['lock_hash'],
        'rows':[(row.sample_id,row.group_id,row.labels,value.text_hash) for row,value in zip(ordered,visible)]})
    path=bounded_path(root/'predictions'/f'{run.run_id}.parquet',root)
    info_path=path.with_suffix('.manifest.json')
    if info_path.exists():
        info=read_json(info_path)
        if info['request_hash']!=identity:raise ValueError('Prediction cache input/request identity changed')
        if not path.is_file() or hash_file(path)!=info['file_sha256']:raise ValueError('Prediction cache bytes changed')
        result=_read_predictions(path);validate_prediction_rows(result,len(meta['label_names']),meta['task'])
        if len(result)!=len(ordered) or canonical_hash(_metrics(result,meta))!=info['metrics_hash']:
            raise ValueError('Saved prediction metrics/row counts do not reconstruct exactly')
        return result
    journal=root/'evaluation-started.json'
    now=datetime.now(timezone.utc).isoformat()
    history=read_json(journal) if journal.exists() else {
        'first_new_official_prediction_started_at_utc':now,
        'evaluation_freeze_hash':freeze['lock_hash'],'first_run_id':run.run_id,'datasets':{}}
    if history['evaluation_freeze_hash']!=freeze['lock_hash']:
        raise ValueError('Official-test exposure history belongs to a different evaluation freeze')
    history.setdefault('datasets',{}).setdefault(meta['dataset_id'],{
        'first_v3_model_prediction_at_utc':now,'exposure_status':test_exposure_status(meta['dataset_id'])})
    atomic_write_json(journal,history,root)
    inference_started=time.perf_counter()
    scores=score_saved(output,visible)
    inference_seconds=time.perf_counter()-inference_started
    if scores.shape!=(len(ordered),len(meta['label_names'])):raise ValueError('Checkpoint lost the full declared score space')
    kind=status['score_kind'];threshold=.5 if kind=='probability' else 0.
    multilabel=meta['task'] in {'multilabel','multi_label'}
    result=tuple(PredictionRow(run.run_id,row.sample_id,row.group_id,tuple(row.labels),
        tuple(int(x) for x in np.flatnonzero(score>=threshold)) if multilabel else (int(score.argmax()),),
        tuple(float(x) for x in score),kind,run.input_hash) for row,score in zip(ordered,scores))
    validate_prediction_rows(result,len(meta['label_names']),meta['task'])
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=pa.schema([('run_id',pa.string()),('sample_id',pa.string()),('group_id',pa.string()),
        ('labels',pa.list_(pa.int32())),('prediction',pa.list_(pa.int32())),
        ('scores',pa.list_(pa.float64())),('score_kind',pa.string()),('input_hash',pa.string())])
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.parquet.part')
    pq.write_table(pa.Table.from_pylist([asdict(row) for row in result],schema=schema),temporary,compression='zstd')
    temporary.replace(path)
    # Float64 storage preserves source classifier values, including margins.
    saved=_read_predictions(path)
    if saved!=result:raise ValueError('Prediction values changed while saving')
    metrics=_metrics(saved,meta)
    atomic_write_json(info_path,{'status':'complete','request_hash':identity,
        'evaluation_freeze_hash':freeze['lock_hash'],'run_id':run.run_id,
        'file_sha256':hash_file(path),'rows':len(result),'score_dtype':'float64',
        'dataset_id':meta['dataset_id'],'test_exposure_status':test_exposure_status(meta['dataset_id']),
        'prediction_completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint_to_scores_seconds':inference_seconds,
        'inference_timing_scope':'checkpoint_verification_load_tokenization_feature_transform_forward_and_cpu_scores; excludes_global_freeze_verification_source_load_and_output_write',
        'metrics':metrics,'metrics_hash':canonical_hash(metrics),
        'official_test_predictions_generated':True},root)
    return saved

"""Full-source preparation with persisted support and duplicate-isolation proofs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, canonical_hash
from .datasets import load_examples
from .inputs import write_inputs
from .partitions import prepare_partitions
from .storage import atomic_write_json, hash_file, read_json
from .types import Example, PreparedDataset

PURPOSES = ('train', 'select', 'discovery', 'test', 'quarantine')


def preparation_code_hash() -> str:
    return canonical_hash({name: hash_file(Path(__file__).parent / name)
                           for name in ['preparation.py','partitions.py','datasets.py','inputs.py','types.py']})


def save_prepared_dataset(data: PreparedDataset, root: Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([('sample_id',pa.string()),('text',pa.string()),
                        ('labels',pa.list_(pa.int32())),('official_split',pa.string()),
                        ('group_id',pa.string()),('fields_json',pa.string())])
    result = {'dataset_id':data.dataset_id,'task':data.task,'label_names':list(data.label_names),
              'manifest_hash':data.manifest_hash,'partitions':{}}
    for purpose in PURPOSES:
        rows = data.partitions[purpose]
        path = Path(root) / 'data' / data.dataset_id / f'{purpose}.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.parquet.part')
        support = [0] * len(data.label_names)
        reasons: Counter[str] = Counter()
        with pq.ParquetWriter(temporary,schema,compression='zstd') as writer:
            for start in range(0,len(rows),4096):
                batch=[]
                for row in rows[start:start+4096]:
                    for label in row.labels:
                        support[label]+=1
                    if purpose=='quarantine':
                        reasons[row.fields.get('quarantine_reason','unspecified')]+=1
                    batch.append({'sample_id':row.sample_id,'text':row.text,'labels':list(row.labels),
                                  'official_split':row.official_split,'group_id':row.group_id,
                                  'fields_json':json.dumps(row.fields,ensure_ascii=False)})
                writer.write_table(pa.Table.from_pylist(batch,schema=schema))
        temporary.replace(path)
        result['partitions'][purpose]={'file':path.relative_to(root).as_posix(),
            'sha256':hash_file(path),'bytes':path.stat().st_size,'rows':len(rows),
            'label_support':support,'ap_mask':[c for c,n in enumerate(support) if 0<n<len(rows)],
            'zero_positive_labels':[c for c,n in enumerate(support) if n==0],
            'all_positive_labels':[c for c,n in enumerate(support) if n==len(rows) and len(rows)>0],
            'quarantine_reasons':dict(reasons)}
    return result


def load_prepared_dataset(root: Path, metadata: dict[str,Any], include_text: bool=False,
                          train_only: bool=False) -> PreparedDataset:
    import pyarrow.parquet as pq
    partitions={}
    for name in PURPOSES:
        record=metadata['partitions'][name]
        path=Path(root)/record['file']
        if not path.is_file() or hash_file(path)!=record['sha256']:
            raise ValueError(f'Prepared partition hash changed: {path}')
        rows=[]
        if not train_only or name=='train':
            columns=['sample_id','labels','official_split','group_id']
            if include_text:
                columns+=['text','fields_json']
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=columns):
                for row in batch.to_pylist():
                    rows.append(Example(metadata['dataset_id'],row['sample_id'],row.get('text',''),
                        tuple(row['labels']),row['official_split'],row['group_id'],
                        json.loads(row.get('fields_json','{}'))))
        partitions[name]=tuple(rows)
    return PreparedDataset(metadata['dataset_id'],metadata['task'],tuple(metadata['label_names']),
                           partitions,metadata['manifest_hash'])


def audit_global_partition_groups(data: dict[str,PreparedDataset]) -> dict[str,Any]:
    owner={}
    counts=Counter()
    for dataset in data.values():
        for purpose in PURPOSES[:-1]:
            for row in dataset.partitions[purpose]:
                if row.group_id in owner and owner[row.group_id]!=purpose:
                    raise ValueError(f'Global duplicate group overlap: {row.group_id}')
                owner[row.group_id]=purpose
                counts[purpose]+=1
    return {'global_group_overlap_count':0,'unique_retained_groups':len(owner),
            'retained_rows_by_purpose':dict(counts),'quarantine_excluded_from_overlap_proof':True}


def _legacy_ids() -> dict[str,tuple[str,...]]:
    result={}
    for dataset in ['imdb','rotten_tomatoes','yelp','ag_news']:
        folder='data-v2' if dataset=='ag_news' else 'data'
        with (PROJECT_ROOT/folder/dataset/'validation.jsonl').open(encoding='utf-8-sig') as stream:
            result[dataset]=tuple(json.loads(line)['sample_id'] for line in stream if line.strip())
    return result


def prepare_all(p: dict[str,Any], root: Path, assets: dict[str,Any]) -> dict[str,Any]:
    import psutil
    root=Path(root)
    code_hash=preparation_code_hash()
    identity={'protocol_hash':p['_protocol_hash'],'asset_hash':assets['asset_hash'],
              'preparation_code_hash':code_hash,
              'legacy_validation_files':{name:hash_file(PROJECT_ROOT/folder/name/'validation.jsonl')
                for name,folder in [('imdb','data'),('rotten_tomatoes','data'),('yelp','data'),('ag_news','data-v2')]}}
    manifest_path=root/'data/prepared_manifest.json'
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old.get('identity')==identity and old.get('complete'):
            for meta in old['datasets'].values():
                for rec in meta['partitions'].values():
                    if hash_file(root/rec['file'])!=rec['sha256']:
                        raise ValueError('Prepared partition changed; refusing stale completed state')
                rec=meta['inputs']
                if hash_file(root/rec['file'])!=rec['file_sha256']:
                    raise ValueError('Visible input cache changed; refusing stale completed state')
            return {'status':'verified_existing','prepared_hash':old['prepared_hash'],
                    'datasets':len(old['datasets']),'actual_model_training_run':False}
    def progress(event):
        event={**event,'pid':os.getpid(),'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
               'rss_gib':round(psutil.Process().memory_info().rss/2**30,3)}
        atomic_write_json(root/'data/progress.json',event,root)
        print(json.dumps(event,ensure_ascii=False),flush=True)
    all_rows=[]
    counts={}
    for dataset in p['scope']['single_text_datasets']:
        rows=load_examples(dataset,assets)
        counts[dataset]=dict(Counter(r.official_split for r in rows))
        expected=assets['datasets'][dataset].get('official_counts')
        if expected and counts[dataset]!=expected:
            raise ValueError(f'Official source counts changed: {dataset}: {counts[dataset]} != {expected}')
        all_rows.extend(rows)
        progress({'stage':'load_official_sources','dataset':dataset,'rows':len(rows),'loaded_total':len(all_rows)})
    protocol={**p,'_dataset_registry':assets['datasets'],'_legacy_validation_ids':_legacy_ids(),
              '_preparation_workspace':root/'data/.work','_progress_callback':progress}
    prepared=prepare_partitions(tuple(all_rows),protocol)
    del all_rows,rows
    gc.collect()
    overlap=audit_global_partition_groups(prepared)
    metadata={}
    for name,data in prepared.items():
        progress({'stage':'persist_partitions','dataset':name})
        metadata[name]=save_prepared_dataset(data,root)
        metadata[name]['official_source_counts']=counts[name]
        metadata[name]['post_isolation_holdout_counts_are_actual']=True
        if data.task=='multi_label' and not metadata[name]['partitions']['select']['ap_mask']:
            raise ValueError(f'{name}: selection AP mask is empty')
    # Arrow has already been loaded before the tokenizer/model libraries.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(assets['models']['distilbert']['directory'],
                                           local_files_only=True,use_fast=True)
    for name in list(prepared):
        data=prepared[name]
        audit=write_inputs(data,tokenizer,root/'inputs'/f'{name}.parquet',
                           p['preparation']['tokenizer_batch_rows'],progress)
        audit['file']=Path(audit['file']).relative_to(root).as_posix()
        metadata[name]['inputs']=audit
        atomic_write_json(root/'data'/name/'manifest.json',metadata[name],root)
        del prepared[name],data
        gc.collect()
    result={'schema_version':1,'complete':True,'identity':identity,'datasets':metadata,
            'overlap_audit':overlap,'near_duplicate_protocol':{
                'normalization':'HTML removal, whitespace collapse, Unicode casefold',
                'shingle_words':5,'minimum_words':10,'minhash_permutations':128,
                'bands':32,'rows_per_band':4,'exact_jaccard_threshold':0.9,
                'candidate_recall_is_probabilistic':True},
            'actual_model_training_run':False,'model_predictions_generated':False}
    result['prepared_hash']=canonical_hash(result)
    atomic_write_json(manifest_path,result,root)
    progress({'stage':'prepared_complete','datasets':len(metadata),'prepared_hash':result['prepared_hash']})
    return {'status':'complete','datasets':len(metadata),'prepared_hash':result['prepared_hash'],
            'actual_model_training_run':False}

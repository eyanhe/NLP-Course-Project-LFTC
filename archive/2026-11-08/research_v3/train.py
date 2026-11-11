"""Exact-condition fitting, train-only features, checkpoint reload and T/V metrics."""
from __future__ import annotations

from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import joblib
import numpy as np

from .config import canonical_hash
from .features import fit_features,visible_texts
from .metrics import metric_bundle
from .models import build_model
from .storage import atomic_write_json,bounded_path,hash_file,read_json
from .types import PreparedDataset,RunSpec,VisibleInput

TRADITIONAL={'tfidf_lr','tfidf_svm','w2v_svm','fixed_tfidf_svm'}


def accumulation_groups(order,effective_batch: int,micro_batch: int):
    if micro_batch<1 or effective_batch<1 or micro_batch>effective_batch:
        raise ValueError('Invalid effective/micro batch size')
    return [[order[j:min(j+micro_batch,i+effective_batch)] for j in range(i,min(i+effective_batch,len(order)),micro_batch)]
            for i in range(0,len(order),effective_batch)]


def learner_code_hash() -> str:
    names=['features.py','models.py','metrics.py','train.py','types.py','inputs.py','resources.py']
    return canonical_hash({name:hash_file(Path(__file__).parent/name) for name in names})


def _condition(run: RunSpec,data: PreparedDataset,root: Path) -> tuple[str,...]:
    specific=root/'training/conditions'/f'{run.condition_id}.json'
    if specific.exists():
        record=read_json(specific)
    else:
        design=read_json(root/'matrix/design_manifest.json')
        matches=[node for node in design['conditions'] if node['condition_id']==run.condition_id]
        if len(matches)!=1:raise ValueError('Missing or duplicate condition identity')
        record=matches[0]
    if record['dataset_id']!=data.dataset_id or record['metadata']['pool_manifest_hash']!=data.manifest_hash:
        raise ValueError('Condition training pool identity changed')
    if 'train_ids_file' in record:
        path=bounded_path(root/record['train_ids_file'],root)
        if hash_file(path)!=record['train_ids_file_sha256']:raise ValueError('Condition ID file hash changed')
        with gzip.open(path,'rt',encoding='utf-8') as stream:ids=tuple(line.rstrip('\n') for line in stream)
    else:
        ids=tuple(record['train_ids'])
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Training IDs must be nonempty and unique')
    return tuple(sorted(ids))


def target_matrix(rows,label_count: int,task: str) -> np.ndarray:
    if task=='single_label':return np.asarray([row.labels[0] for row in rows],dtype=np.int64)
    result=np.zeros((len(rows),label_count),dtype=np.float32)
    for i,row in enumerate(rows):result[i,list(row.labels)]=1
    return result


def _seed(seed: int):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def _joblib_atomic(value,path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    joblib.dump(value,temporary,compress=0)
    temporary.replace(path)


def _features(run,train_inputs,root):
    model=run.model_id
    if model not in TRADITIONAL|{'bilstm'}:return None
    hp=run.hparams
    if model in {'tfidf_lr','tfidf_svm','fixed_tfidf_svm'}:
        family='tfidf';settings={name:hp.get(name,default) for name,default in
            [('max_features',30000),('min_df',1),('sublinear_tf',True),('ngram_range',[1,2])]}
    else:
        family='word2vec'
        settings={**hp.get('embedding_initialization',{})}
        for name,default in [('vector_size',100),('window',5),('min_count',1),('epochs',10),
                             ('seed',2026),('vocabulary_limit',30000)]:
            settings[name]=hp.get(name,settings.get(name,default))
        # Neural training epochs never change the local SGNS training budget.
        if model=='bilstm':settings['epochs']=hp.get('embedding_initialization',{}).get('epochs',10)
    identity=canonical_hash({'family':family,'settings':settings,
        'inputs':[(row.sample_id,row.text_hash) for row in train_inputs],
        'feature_code_sha256':hash_file(Path(__file__).parent/'features.py')})
    directory=root/'features'/identity
    path=directory/'features.joblib';metadata=directory/'manifest.json'
    if metadata.exists():
        manifest=read_json(metadata)
        if not path.is_file() or hash_file(path)!=manifest['sha256']:raise ValueError('Feature cache hash changed')
        features=joblib.load(path)
        if features.fitted_sample_ids!=tuple(row.sample_id for row in train_inputs):
            raise ValueError('Feature cache training IDs changed')
        return features
    features=fit_features(model,train_inputs,settings)
    _joblib_atomic(features,path)
    atomic_write_json(metadata,{'feature_id':identity,'sha256':hash_file(path),
        'train_ids':list(features.fitted_sample_ids),'settings':settings},root)
    return features


def _batch(model_id,inputs,features,tokenizer,device):
    import torch
    if model_id=='bilstm':
        sequences=[torch.as_tensor(row,dtype=torch.long) for row in features.sequences(inputs)]
        lengths=torch.tensor([len(row) for row in sequences],dtype=torch.long)
        return {'ids':torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True,padding_value=0).to(device),
                'lengths':lengths}
    encoded=tokenizer(visible_texts(inputs),padding=True,truncation=False,return_tensors='pt',verbose=False)
    limit=8192 if model_id=='modernbert' else 256
    if encoded['input_ids'].shape[1]>limit:
        raise ValueError(f'Common visible input exceeds {model_id} capacity; additional truncation is forbidden')
    return {name:value.to(device) for name,value in encoded.items() if name in {'input_ids','attention_mask'}}


def _logits(model,model_id,batch):
    return model(batch['ids'],batch['lengths']) if model_id=='bilstm' else model(**batch).logits


def _neural_scores(model,model_id,inputs,features,tokenizer,hp,constants):
    import torch
    device=hp.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    micro=int(hp.get('micro_batch',4));outputs=[]
    use_amp=device.startswith('cuda') and model_id!='bilstm' and torch.cuda.is_bf16_supported()
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(inputs),micro):
            batch=_batch(model_id,inputs[start:start+micro],features,tokenizer,device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=use_amp):
                outputs.append(_logits(model,model_id,batch).float().cpu().numpy())
    result=np.concatenate(outputs,axis=0) if outputs else np.zeros((0,hp['label_count']),dtype=np.float32)
    for col,positive in constants.items():result[:,int(col)]=30 if positive else -30
    if not np.isfinite(result).all():raise ValueError('Nonfinite neural scores')
    return result


def _checkpoint(model,model_id,output):
    directory=output/'checkpoint';directory.mkdir(parents=True,exist_ok=True)
    if model_id=='bilstm':
        from safetensors.torch import save_file
        temporary=directory/'model.safetensors.part'
        save_file({name:value.detach().cpu().contiguous() for name,value in model.state_dict().items()},str(temporary))
        temporary.replace(directory/'model.safetensors')
    else:
        model.save_pretrained(directory,safe_serialization=True)


def _fit_neural(run,data,train_inputs,y,validation_inputs,validation_y,features,assets,output):
    import torch
    from transformers import AutoTokenizer,get_linear_schedule_with_warmup
    hp={**run.hparams,'label_count':len(data.label_names)}
    device=hp.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    hp['device']=device
    if device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    torch.set_num_threads(int(hp.get('cpu_threads',4)))
    epochs=int(hp.get('max_epochs',hp.get('epochs',10 if run.model_id=='bilstm' else 5)))
    if run.strict_no_val and epochs!=(10 if run.model_id=='bilstm' else 5):
        raise ValueError('Strict fewshot must use its fixed neural epoch budget')
    effective=int(hp.get('effective_batch',64 if run.model_id=='bilstm' else 16))
    initial_micro=min(int(hp.get('micro_batch',4)),effective)
    family='modernbert' if run.model_id=='modernbert' else 'distilbert'
    tokenizer=None if run.model_id=='bilstm' else AutoTokenizer.from_pretrained(assets['models'][family]['directory'],local_files_only=True)
    multilabel=data.task in {'multilabel','multi_label'}
    constants={int(col):bool(y[0,col]) for col in range(y.shape[1]) if np.all(y[:,col]==y[0,col])} if multilabel else {}
    active=[col for col in range(len(data.label_names)) if col not in constants]
    attempts=[];micro=initial_micro
    while True:
        try:
            _seed(int(run.training_seed or 42))
            hp['micro_batch']=micro
            model=build_model(run.model_id,data.task,len(data.label_names),assets,hp,features).to(device)
            optimizer_cls=torch.optim.Adam if run.model_id=='bilstm' else torch.optim.AdamW
            optimizer=optimizer_cls(model.parameters(),lr=float(hp.get('lr',.001 if run.model_id=='bilstm' else 2e-5)),
                weight_decay=float(hp.get('weight_decay',0 if run.model_id=='bilstm' else .01)))
            total_updates=math.ceil(len(y)/effective)*epochs
            scheduler=get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=int(total_updates*float(hp.get('warmup_ratio',0 if run.model_id=='bilstm' else .1))),
                num_training_steps=total_updates)
            history=[];best=-math.inf;selected_epoch=0;updates=0;last_progress=time.monotonic();probe_scores=None
            for epoch in range(1,epochs+1):
                model.train();order=np.random.default_rng(1000003*int(run.training_seed or 42)+epoch).permutation(len(y))
                sum_loss=0.;seen=0
                for chunks in accumulation_groups(order,effective,micro):
                    optimizer.zero_grad(set_to_none=True);group_n=sum(len(chunk) for chunk in chunks)
                    for indexes in chunks:
                        subset=tuple(train_inputs[int(i)] for i in indexes)
                        batch=_batch(run.model_id,subset,features,tokenizer,device)
                        target=torch.as_tensor(y[indexes],device=device)
                        use_amp=device.startswith('cuda') and run.model_id!='bilstm' and torch.cuda.is_bf16_supported()
                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=use_amp):
                            logits=_logits(model,run.model_id,batch)
                            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits[:,active].float(),target[:,active].float())
                                  if multilabel and active else logits.sum()*0 if multilabel
                                  else torch.nn.functional.cross_entropy(logits.float(),target.long()))
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite neural training loss')
                        (loss*(len(indexes)/group_n)).backward()
                        sum_loss+=float(loss.detach())*len(indexes);seen+=len(indexes)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();scheduler.step();updates+=1
                    if time.monotonic()-last_progress>=5:
                        atomic_write_json(output/'progress.json',{'status':'running','pid':os.getpid(),'epoch':epoch,
                            'epochs':epochs,'training_rows_seen':seen,'training_rows':len(y),'optimizer_updates':updates,
                            'micro_batch':micro,'effective_batch':effective},output)
                        last_progress=time.monotonic()
                if run.strict_no_val:
                    metric=None;improved=epoch==epochs
                else:
                    scores=_neural_scores(model,run.model_id,validation_inputs,features,tokenizer,hp,constants)
                    metric=metric_bundle(validation_y,scores,data.task,'logit',
                        label_mask=hp.get('select_ap_mask') if multilabel else None)
                    value=metric['macro_ap'] if multilabel else metric['macro_f1']
                    if value is None:raise ValueError('Validation metric has no supported labels')
                    improved=value>best
                    if improved:best=value
                history.append({'epoch':epoch,'train_loss':sum_loss/len(y),'validation':metric,'optimizer_updates':updates})
                if improved:
                    selected_epoch=epoch
                    probe_scores=_neural_scores(model,run.model_id,train_inputs[:16],features,tokenizer,hp,constants)
                    _checkpoint(model,run.model_id,output)
                atomic_write_json(output/'history.json',{'epochs':history,'attempts':attempts},output)
            del optimizer,scheduler
            model.to('cpu');del model;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
            return {'hparams':hp,'constants':constants,'epochs_completed':epochs,'selected_epoch':selected_epoch,
                    'optimizer_updates':updates,'effective_batch':effective,'micro_batch':micro,'attempts':attempts,
                    'validation_metric':None if run.strict_no_val else best,'history':history,'probe_scores':probe_scores}
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({'micro_batch':micro,'error':str(exc),'restart_from_initial_seed':True})
            if micro==1:raise
            micro=max(1,micro//2)
        # Exit the exception scope before collecting its model/traceback refs.
        for name in ['model','optimizer','scheduler']:
            if name in locals():
                if name=='model':model=None
                elif name=='optimizer':optimizer=None
                else:scheduler=None
        gc.collect();torch.cuda.empty_cache()


def _verified_bundle(output: Path,allow_verifying: bool=False):
    status=read_json(output/'status.json')
    if not (status['status']=='verifying' and allow_verifying) and not (
            status['status']=='complete' and status.get('reload_verified')):
        raise ValueError('Run is not complete with a verified checkpoint reload')
    for relative,expected in status['files'].items():
        path=bounded_path(output/relative,output)
        if not path.is_file() or hash_file(path)!=expected:raise ValueError(f'Saved artifact hash changed: {relative}')
    return status,joblib.load(output/'bundle.joblib')


def _score_saved(output: Path,inputs: tuple[VisibleInput,...],allow_verifying: bool=False) -> np.ndarray:
    output=Path(output);status,bundle=_verified_bundle(output,allow_verifying)
    if bundle['model_id'] in TRADITIONAL:
        return bundle['model'].scores(bundle['features'].transform(inputs))
    import torch
    hp=bundle['hparams'];model_id=bundle['model_id'];device=hp['device']
    if model_id=='bilstm':
        from safetensors.torch import load_file
        model=build_model(model_id,bundle['task'],bundle['label_count'],bundle['assets'],hp,bundle['features'])
        model.load_state_dict(load_file(str(output/'checkpoint/model.safetensors')))
        tokenizer=None
    else:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        model=AutoModelForSequenceClassification.from_pretrained(output/'checkpoint',local_files_only=True,attn_implementation='sdpa')
        family='modernbert' if model_id=='modernbert' else 'distilbert'
        tokenizer=AutoTokenizer.from_pretrained(bundle['assets']['models'][family]['directory'],local_files_only=True)
    model.to(device)
    result=_neural_scores(model,model_id,inputs,bundle['features'],tokenizer,hp,bundle['constants'])
    del model;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def score_saved(output: Path,inputs: tuple[VisibleInput,...]) -> np.ndarray:
    return _score_saved(output,inputs)


def _fit_unleased(run: RunSpec,data: PreparedDataset,inputs: dict[str,VisibleInput],assets: dict[str,Any],output: Path) -> dict[str,Any]:
    output=Path(output).resolve();root=output.parent.parent
    if run.strict_no_val and any(key in json.dumps(run.hparams) for key in ['selected_by','selection_node']):
        raise ValueError('Strict fewshot cannot use anchor selection references')
    identity=canonical_hash(asdict(run));status_path=output/'status.json'
    if status_path.exists():
        previous=read_json(status_path)
        if previous.get('request_hash')!=identity:raise ValueError('Run request identity changed; reuse is forbidden')
        if previous['status']=='complete':
            _verified_bundle(output);return {**previous,'reused_complete':True}
    ids=_condition(run,data,root);by_id={row.sample_id:row for row in data.partitions['train']}
    if not set(ids)<=set(by_id):raise ValueError('Condition requests labels outside the T partition')
    training_rows=tuple(by_id[sid] for sid in ids);train_inputs=tuple(inputs[sid] for sid in ids)
    visible_texts(train_inputs)
    if run.strict_no_val:
        validation_rows=();validation_inputs=()
    else:
        validation_rows=data.partitions['select']
        validation_inputs=tuple(inputs[row.sample_id] for row in validation_rows)
        if not validation_rows:raise ValueError('Non-strict fitting requires its registered validation partition')
    output.mkdir(parents=True,exist_ok=True)
    y=target_matrix(training_rows,len(data.label_names),data.task)
    vy=target_matrix(validation_rows,len(data.label_names),data.task) if validation_rows else None
    started=time.monotonic();features_started=time.monotonic()
    atomic_write_json(status_path,{'status':'running','run_id':run.run_id,'pid':os.getpid(),'request_hash':identity},output)
    try:
        features=_features(run,train_inputs,root);feature_seconds=time.monotonic()-features_started
        neural={};fit_started=time.monotonic()
        bundle={'model_id':run.model_id,'task':data.task,'label_count':len(data.label_names),
                'features':features,'hparams':dict(run.hparams),'assets':assets,'constants':{}}
        if run.model_id in TRADITIONAL:
            model=build_model(run.model_id,data.task,len(data.label_names),assets,run.hparams,features)
            model.fit(features.transform(train_inputs),y)
            bundle['model']=model;score_kind=model.score_kind
            before=model.scores(features.transform(train_inputs[:16]))
        else:
            neural=_fit_neural(run,data,train_inputs,y,validation_inputs,vy,features,assets,output)
            bundle['hparams']=neural['hparams'];bundle['constants']=neural['constants'];score_kind='logit'
            before=neural['probe_scores']
        fit_seconds=time.monotonic()-fit_started
        _joblib_atomic(bundle,output/'bundle.joblib')
        files={path.relative_to(output).as_posix():hash_file(path) for path in output.rglob('*')
               if path.is_file() and path.name not in {'status.json','progress.json'} and not path.name.endswith('.part')
               and not path.name.startswith('.lease')}
        result={'status':'verifying','run_id':run.run_id,'request_hash':identity,'run':asdict(run),
            'training_unique_rows':len(ids),'training_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'holdout_labels_accessed':len(validation_rows),'score_kind':score_kind,'files':files,
            'feature_degenerate':bool(features and features.degenerate),
            'feature_seconds':feature_seconds,'fit_seconds':fit_seconds,'total_seconds':time.monotonic()-started,
            'epochs_completed':neural.get('epochs_completed',0),'selected_epoch':neural.get('selected_epoch'),
            'optimizer_updates':neural.get('optimizer_updates',0),'micro_batch':neural.get('micro_batch'),
            'effective_batch':neural.get('effective_batch'),'oom_attempts':neural.get('attempts',[]),
            'reload_verified':False,'actual_model_training_run':True,'official_test_predictions_generated':False}
        atomic_write_json(status_path,result,output)
        reloaded=_score_saved(output,train_inputs[:16],allow_verifying=True)
        if not np.allclose(before,reloaded,rtol=1e-4,atol=1e-5):raise ValueError('Saved/reloaded score values differ')
        result['reload_verified']=True;result['reload_max_absolute_difference']=float(np.max(np.abs(before-reloaded)))
        if validation_rows:
            vs=_score_saved(output,validation_inputs,allow_verifying=True)
            result['validation']=metric_bundle(vy,vs,data.task,score_kind,
                label_mask=run.hparams.get('select_ap_mask') if data.task!='single_label' else None)
        result['total_seconds']=time.monotonic()-started;result['status']='complete'
        atomic_write_json(status_path,result,output)
        return result
    except Exception as exc:
        atomic_write_json(status_path,{'status':'failed','run_id':run.run_id,'request_hash':identity,
            'error_type':type(exc).__name__,'error':str(exc),'pid':os.getpid(),
            'official_test_predictions_generated':False},output)
        raise


def fit_run(run: RunSpec,data: PreparedDataset,inputs: dict[str,VisibleInput],assets: dict[str,Any],output: Path) -> dict[str,Any]:
    from .execution import run_lease
    from .resources import ResourceSampler
    with run_lease(Path(output),canonical_hash(asdict(run))):
        with ResourceSampler(run.hparams.get('device','cpu')) as resources:
            result=_fit_unleased(run,data,inputs,assets,output)
        if not result.get('reused_complete'):
            result['resources']=resources.report
            atomic_write_json(Path(output)/'status.json',result,output)
        return result

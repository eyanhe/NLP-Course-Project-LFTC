"""Isolated M2/M3/M5 fitting, sharing the core's checkpoint/score contract.

This module does not authorize a phase, read discovery/test labels, construct
new condition IDs, or generate official-test predictions. The stage controller
must authorize real runs. Optional config/state/tokenizer injection permits
small, offline CPU verification; production defaults use locked local assets.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np

from .config import canonical_hash
from .features import WordFeatures, visible_texts
from .metrics import metric_bundle
from .storage import atomic_write_json, hash_file, read_json
from .train import (_batch, _checkpoint, _condition, _features, _joblib_atomic,
                    _logits, _neural_scores, _score_saved, _verified_bundle,
                    accumulation_groups, fit_run, target_matrix)


def diagnostics_code_hash() -> str:
    """Fingerprint diagnostics separately, preserving all existing main IDs."""
    return hash_file(Path(__file__))


def epoch_order(size: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(1000003 * seed + epoch).permutation(size)


def current_tail_labels(head_order) -> tuple[int, ...]:
    order=tuple(int(x) for x in head_order)
    if len(order)<2 or set(order)!=set(range(len(order))):
        raise ValueError('Head order must be a full label permutation')
    return order[-max(1,len(order)//3):]


def natural_tail_labels(counts) -> tuple[int, ...] | None:
    counts=np.asarray(counts)
    if counts.ndim!=1 or len(counts)<2 or np.any(counts<0):
        raise ValueError('Invalid natural label counts')
    if np.all(counts==counts[0]):return None
    boundary=np.sort(counts)[max(1,len(counts)//3)-1]
    return tuple(int(x) for x in np.flatnonzero(counts<=boundary))


def tail_reduced_indices(sample_ids, labels, head_order, rho: int, seed: int) -> np.ndarray:
    """Select only a subset of U0; the registered condition remains authoritative."""
    labels=np.asarray(labels);ids=tuple(sample_ids);order=tuple(head_order)
    tail=set(current_tail_labels(order))
    if rho not in {10,50} or len(ids)!=len(labels) or len(ids)!=len(set(ids)):
        raise ValueError('Invalid unique-support reduction request')
    if labels.ndim!=1 or not np.isin(labels,np.arange(len(order))).all():
        raise ValueError('Labels outside declared space')
    result=[]
    for label in range(len(order)):
        group=list(np.flatnonzero(labels==label))
        if not group:raise ValueError('Reference support must cover every label')
        group.sort(key=lambda i:hashlib.sha256(f'{seed}:{ids[i]}'.encode()).hexdigest())
        result.extend(group[:max(1,len(group)//rho)] if label in tail else group)
    return np.asarray(sorted(result),dtype=np.int64)


def m3_sample_weights(labels, target_q, reference_n: int) -> np.ndarray:
    """Weights sum to N0, preserving SVM regularization across support sizes."""
    y=np.asarray(labels);q=np.asarray(target_q,dtype=float)
    if (y.ndim!=1 or not len(y) or q.ndim!=1 or len(q)<2 or
            not np.isfinite(q).all() or np.any(q<=0) or not np.isclose(q.sum(),1.) or
            not np.isin(y,np.arange(len(q))).all() or reference_n<len(y)):
        raise ValueError('Invalid M3 labels, target risk or reference support')
    counts=np.bincount(y.astype(int),minlength=len(q))
    if np.any(counts==0):raise ValueError('M3 support must retain every label')
    return reference_n*q[y.astype(int)]/counts[y.astype(int)]


def m3_head_loss(logits, labels, per_row_weight):
    """Mean of weighted per-row CE, never divide by the batch weight sum."""
    import torch
    loss=torch.nn.functional.cross_entropy(logits,labels.long(),reduction='none')
    return (loss*per_row_weight.to(device=loss.device,dtype=loss.dtype)).mean()


def _torch_seed(seed: int):
    # Seed the CPU generator directly: torch.manual_seed also touches CUDA.
    import torch
    torch.random.default_generator.manual_seed(int(seed))


def build_m2_model(config, arm: str, seed: int, *, pretrained_encoder_state=None,
                   pretrained_directory=None):
    """One HF CLS/head architecture; encoder/head RNG streams are independent."""
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertModel
    if arm not in {'pretrained_frozen','pretrained_unfrozen','random_frozen','random_unfrozen'}:
        raise ValueError('Unknown M2 arm')
    with torch.random.fork_rng(devices=[]):
        _torch_seed(seed)
        model=DistilBertForSequenceClassification(deepcopy(config))
        if arm.startswith('pretrained'):
            if pretrained_encoder_state is None:
                if pretrained_directory is None:
                    raise ValueError('Pretrained arm requires a generic locked encoder')
                pretrained_encoder_state=DistilBertModel.from_pretrained(
                    pretrained_directory,local_files_only=True).state_dict()
            model.distilbert.load_state_dict(pretrained_encoder_state,strict=True)
        # Explicitly initialize and copy the same two head layers in all arms.
        _torch_seed(2000003+seed)
        model._init_weights(model.pre_classifier)
        model._init_weights(model.classifier)
    for parameter in model.distilbert.parameters():
        parameter.requires_grad_(arm.endswith('_unfrozen'))
    return model


def m2_optimizer_groups(model, encoder_lr: float=2e-5):
    head=[p for n,p in model.named_parameters() if not n.startswith('distilbert.') and p.requires_grad]
    groups=[{'params':head,'lr':.001}]
    encoder=[p for p in model.distilbert.parameters() if p.requires_grad]
    if encoder:groups.append({'params':encoder,'lr':float(encoder_lr)})
    return groups


def paired_word_features(baseline: WordFeatures, expanded: WordFeatures, seed: int) -> WordFeatures:
    """Copy shared rows by token; initialize new rows with an independent stream."""
    if baseline.embedding_matrix.shape[1]!=expanded.embedding_matrix.shape[1]:
        raise ValueError('Paired embeddings must have the same dimension')
    if not set(baseline.vocabulary)<=set(expanded.vocabulary):
        raise ValueError('Expanded vocabulary must retain all baseline tokens')
    matrix=expanded.embedding_matrix.copy()
    matrix[:2]=baseline.embedding_matrix[:2]
    for word,index in expanded.vocabulary.items():
        if word in baseline.vocabulary:
            matrix[index]=baseline.embedding_matrix[baseline.vocabulary[word]]
        else:
            key=int.from_bytes(hashlib.sha256(f'new-row:{seed}:{word}'.encode()).digest()[:8],'little')
            matrix[index]=np.random.default_rng(key).normal(0,.02,size=matrix.shape[1])
    return WordFeatures(dict(expanded.vocabulary),matrix,expanded.degenerate,
                        tuple(expanded.fitted_sample_ids),deepcopy(expanded.settings))


def build_paired_bilstm(baseline: WordFeatures, features: WordFeatures, seed: int,
                        label_count: int, hparams: dict):
    """Baseline starts exactly like core; larger vocab copies its encoder/head."""
    import torch
    from .models import build_model
    with torch.random.fork_rng(devices=[]):
        _torch_seed(seed)
        base=build_model('bilstm','single_label',label_count,{},hparams,baseline)
        # Core seeds once, constructs the baseline, then consumes this stream
        # for dropout. Expanded arms resume that very same post-baseline state.
        training_rng_state=torch.random.get_rng_state().clone()
        if features is baseline:
            base._mechanism_training_rng_state=training_rng_state
            return base
        _torch_seed(seed)
        model=build_model('bilstm','single_label',label_count,{},hparams,features)
        common={name:value for name,value in base.state_dict().items() if name!='embedding.weight'}
        model.load_state_dict(common,strict=False)
        model._mechanism_training_rng_state=training_rng_state
    return model


def m5_baseline_alias_eligible(run, main_run) -> bool:
    """Root may alias an actual main checkpoint only for identical fit requests."""
    def normalized(r):
        hp=dict(r.hparams)
        hp.setdefault('vocabulary_limit',30000)
        hp.setdefault('max_features',30000)
        hp['max_epochs']=hp.get('max_epochs',hp.get('epochs',10 if r.model_id=='bilstm' else 5))
        hp.pop('epochs',None)
        return (r.condition_id,r.model_id,r.training_seed,r.input_hash,r.asset_hash,r.code_hash,
                r.strict_no_val,hp)
    return normalized(run)==normalized(main_run)


def _state_hash(state) -> str | None:
    if state is None:return None
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


class GenericCLSCache:
    """Cache only requested common-visible rows, keyed by encoder and text bytes.

    An optional directory allows reuse between the four risk/support cells.
    The generic encoder is always evaluated without target adaptation.
    """
    def __init__(self, directory=None):
        self.directory=Path(directory) if directory is not None else None
        self.rows={}
        self.rows_encoded=0

    def encode(self, model, inputs, tokenizer, device, micro_batch=32):
        import torch
        encoder_hash=_state_hash(model.distilbert.state_dict())
        tokenizer_hash=canonical_hash(tokenizer.get_vocab())
        keys=[canonical_hash({'encoder':encoder_hash,'tokenizer':tokenizer_hash,
                              'id':row.sample_id,'text':hashlib.sha256(row.text.encode()).hexdigest()})
              for row in inputs]
        pending={}
        for key,row in zip(keys,inputs):
            if key in self.rows:continue
            path=self.directory/f'{key}.npz' if self.directory is not None else None
            if path is not None and path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    value=saved['cls'];expected=str(saved['sha256'])
                if hashlib.sha256(value.tobytes()).hexdigest()!=expected:
                    raise ValueError('CLS cache row hash changed')
                self.rows[key]=value
            else:pending[key]=row
        items=list(pending.items());model.distilbert.eval()
        with torch.inference_mode():
            for start in range(0,len(items),micro_batch):
                chunk=items[start:start+micro_batch]
                batch=_batch('distilbert',tuple(row for _,row in chunk),None,tokenizer,device)
                values=model.distilbert(**batch).last_hidden_state[:,0,:].float().cpu().numpy()
                for (key,_),value in zip(chunk,values):
                    self.rows[key]=value.copy();self.rows_encoded+=1
                    if self.directory is not None:
                        self.directory.mkdir(parents=True,exist_ok=True)
                        path=self.directory/f'{key}.npz';temporary=path.with_suffix('.npz.part')
                        with temporary.open('wb') as stream:
                            np.savez(stream,cls=value,sha256=hashlib.sha256(value.tobytes()).hexdigest())
                        temporary.replace(path)
        return np.stack([self.rows[key] for key in keys]) if keys else np.zeros((0,model.config.dim),np.float32)


def _head_logits(model, cls):
    import torch
    return model.classifier(model.dropout(torch.relu(model.pre_classifier(cls))))


def _metric(data, y, scores, hp):
    return metric_bundle(y,scores,data.task,'logit',
                         label_mask=hp.get('select_ap_mask') if data.task!='single_label' else None)


def _fit_epochs(run,data,train_inputs,y,validation_inputs,vy,features,tokenizer,model,output,hp,mechanism):
    import torch
    device=hp['device'];model.to(device)
    epochs=10 if mechanism=='M2' else int(hp.get('max_epochs',hp.get('epochs',10)))
    effective=16 if mechanism=='M2' else int(hp.get('effective_batch',64))
    micro=min(int(hp.get('micro_batch',4)),effective)
    if mechanism=='M2':
        if int(hp.get('epochs',10))!=10 or int(hp.get('effective_batch',16))!=16:
            raise ValueError('M2 requires complete 10 epochs and effective batch16')
        lr=float(hp.get('encoder_lr',2e-5))
        if lr!=2e-5 and (hp['arm']!='random_unfrozen' or lr not in {1e-5,5e-5,5e-4}):
            raise ValueError('Undeclared M2 encoder learning-rate sensitivity')
        optimizer=torch.optim.AdamW(m2_optimizer_groups(model,lr),weight_decay=.01)
    else:
        if epochs not in {10,20}:raise ValueError('M5 BiLSTM budget must be 10 or 20 epochs')
        optimizer=torch.optim.Adam(model.parameters(),lr=float(hp.get('lr',.001)),weight_decay=0.)
    # M5 retains the existing anchor's linear optimizer schedule. M2 uses the
    # separately prescribed constant encoder/head rates for all ten epochs.
    scheduler=None
    if mechanism=='M5':
        from transformers import get_linear_schedule_with_warmup
        total_updates=math.ceil(len(y)/effective)*epochs
        scheduler=get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=int(total_updates*float(hp.get('warmup_ratio',0))),
            num_training_steps=total_updates)
    multilabel=data.task!='single_label'
    constants={int(c):bool(y[0,c]) for c in range(y.shape[1]) if np.all(y[:,c]==y[0,c])} if multilabel else {}
    active=[c for c in range(len(data.label_names)) if c not in constants]
    history=[];best=-math.inf;selected=0;updates=0;probe=None
    exposure=np.zeros(len(y),dtype=np.int64);seed=int(run.training_seed or 42)
    with torch.random.fork_rng(devices=[] if device=='cpu' else [torch.device(device)]):
        if mechanism=='M5':torch.random.set_rng_state(model._mechanism_training_rng_state)
        else:_torch_seed(3000017+seed)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(seed if mechanism=='M5' else 3000017+seed)
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.;tokens=0
            for chunks in accumulation_groups(epoch_order(len(y),seed,epoch),effective,micro):
                optimizer.zero_grad(set_to_none=True);group_n=sum(len(ix) for ix in chunks)
                for ix in chunks:
                    batch_inputs=tuple(train_inputs[int(i)] for i in ix)
                    batch=_batch(run.model_id,batch_inputs,features,tokenizer,device)
                    logits=_logits(model,run.model_id,batch).float()
                    target=torch.as_tensor(y[ix],device=device)
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits[:,active],target[:,active].float())
                          if multilabel and active else logits.sum()*0 if multilabel
                          else torch.nn.functional.cross_entropy(logits,target.long()))
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite mechanism loss')
                    (loss*len(ix)/group_n).backward();loss_sum+=float(loss.detach())*len(ix)
                    exposure[ix]+=1;tokens+=sum(row.content_tokens+2 for row in batch_inputs)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if scheduler is not None:scheduler.step()
                updates+=1
            scores=_neural_scores(model,run.model_id,validation_inputs,features,tokenizer,hp,constants)
            metric=_metric(data,vy,scores,hp);value=metric['macro_f1' if not multilabel else 'macro_ap']
            if value is None:raise ValueError('No supported validation labels')
            history.append({'epoch':epoch,'train_loss':loss_sum/len(y),'validation':metric,
                            'optimizer_updates':updates,'cumulative_tokens':tokens*epoch,
                            'cumulative_training_exposures':int(exposure.sum())})
            if value>best:
                best=value;selected=epoch
                probe=_neural_scores(model,run.model_id,train_inputs[:16],features,tokenizer,hp,constants)
                _checkpoint(model,run.model_id,output)
            atomic_write_json(output/'history.json',{'epochs':history,'attempts':[]},output)
            atomic_write_json(output/'progress.json',{'status':'running','pid':os.getpid(),
                'epoch':epoch,'epochs':epochs,'optimizer_updates':updates},output)
    return {'hparams':{**hp,'micro_batch':micro,'effective_batch':effective},'constants':constants,
            'epochs_completed':epochs,'selected_epoch':selected,'optimizer_updates':updates,
            'history':history,'probe_scores':probe,'validation_metric':best,
            'training_exposures':int(exposure.sum()),
            'exposure_by_sample_id':{row.sample_id:int(n) for row,n in zip(train_inputs,exposure)}}


def _fit_m3_head(run,data,train_inputs,y,validation_inputs,vy,tokenizer,model,output,hp,cache):
    import torch
    device=hp['device'];model.to(device);model.distilbert.eval()
    encoded_before=cache.rows_encoded
    cls=cache.encode(model,train_inputs,tokenizer,device,int(hp.get('micro_batch',32)))
    vcls=cache.encode(model,validation_inputs,tokenizer,device,int(hp.get('micro_batch',32)))
    x=torch.tensor(cls,device=device);vx=torch.tensor(vcls,device=device)
    n0=int(hp['reference_n']);updates=math.ceil(n0/16)*5
    if int(hp.get('updates',updates))!=updates or int(hp.get('effective_batch',16))!=16:
        raise ValueError('M3 requires ceil(N0/16)*5 updates and batch16')
    weights=m3_sample_weights(y,hp['target_q'],n0)*len(y)/n0
    weight=torch.tensor(weights,device=device,dtype=torch.float32)
    target=torch.tensor(y,device=device,dtype=torch.long)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0.)
    seed=int(run.training_seed or 42);rng=np.random.default_rng(1000003*seed)
    exposure=np.zeros(len(y),dtype=np.int64);history=[];best=-math.inf;selected=0;probe=None
    with torch.random.fork_rng(devices=[] if device=='cpu' else [torch.device(device)]):
        _torch_seed(3000017+seed)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(3000017+seed)
        for update in range(1,updates+1):
            ix=rng.integers(0,len(y),size=16);np.add.at(exposure,ix,1)
            index=torch.as_tensor(ix,device=device)
            model.train();model.distilbert.eval();optimizer.zero_grad(set_to_none=True)
            loss=m3_head_loss(_head_logits(model,x[index]),target[index],weight[index])
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite M3 head loss')
            loss.backward();optimizer.step()
            # Complete five budget blocks, selection on V without early stop.
            if update%math.ceil(n0/16)==0:
                model.eval()
                with torch.inference_mode():scores=_head_logits(model,vx).float().cpu().numpy()
                metric=_metric(data,vy,scores,hp);value=metric['macro_f1'];epoch=update//math.ceil(n0/16)
                history.append({'epoch':epoch,'optimizer_updates':update,'train_loss':float(loss.detach()),
                                'validation':metric,'cumulative_training_exposures':int(exposure.sum())})
                if value>best:
                    best=value;selected=epoch
                    # Original model's actual forward, before serialization, is the probe.
                    probe=_neural_scores(model,run.model_id,train_inputs[:16],None,tokenizer,hp,{})
                    _checkpoint(model,run.model_id,output)
                atomic_write_json(output/'history.json',{'epochs':history,'attempts':[]},output)
    return {'hparams':hp,'constants':{},'epochs_completed':5,'selected_epoch':selected,
            'optimizer_updates':updates,'history':history,'probe_scores':probe,'validation_metric':best,
            'training_exposures':int(exposure.sum()),'training_unique_rows_seen':int(np.count_nonzero(exposure)),
            'repeated_exposures':int(exposure.sum()-np.count_nonzero(exposure)),
            'exposure_by_sample_id':{row.sample_id:int(n) for row,n in zip(train_inputs,exposure)},
            'exposure_by_label':np.bincount(y,weights=exposure,minlength=len(data.label_names)).astype(int).tolist(),
            'cls_cache_rows_encoded':cache.rows_encoded-encoded_before}


def score_mechanism_saved(output: Path, inputs):
    """Read the same verified bundle/checkpoint format as core score_saved."""
    return _score_saved(Path(output),tuple(inputs))


def mechanism_request_identity(run,data,inputs,root: Path,*,shared_features=None,
                                baseline_features=None,model_config=None,tokenizer=None,
                                pretrained_encoder_state=None) -> dict:
    """Separate the planned model request from runtime resume-resource checks.

    A manifest can bind model_request_hash before learned TFIDF/CLS artifacts
    exist. Runtime request_hash additionally binds actual supervised bytes,
    optional derived features and injected offline assets, without changing
    the supplied RunSpec/run_id or the reusable main learner code hash.
    """
    if run.strict_no_val:raise ValueError('Mechanism runs cannot be strict E2 tasks')
    ids=_condition(run,data,Path(root));lookup={row.sample_id:row for row in data.partitions['train']}
    if not set(ids)<=set(lookup):raise ValueError('Mechanism condition requests labels outside T')
    rows=tuple(lookup[sid] for sid in ids)+tuple(data.partitions['select'])
    material={'task':data.task,'label_names':data.label_names,'pool_manifest':data.manifest_hash,
        'supervised_inputs':[(row.sample_id,row.labels,hashlib.sha256(inputs[row.sample_id].text.encode()).hexdigest(),
                              asdict(inputs[row.sample_id])) for row in rows],
        'shared_features':joblib.hash(shared_features),'baseline_features':joblib.hash(baseline_features),
        'model_config':model_config.to_dict() if model_config is not None else None,
        'tokenizer_vocab':tokenizer.get_vocab() if tokenizer is not None else None,
        'injected_pretrained_encoder':_state_hash(pretrained_encoder_state)}
    identity={'model_request_hash':canonical_hash(asdict(run)),
              'diagnostics_code_hash':diagnostics_code_hash(),
              'runtime_resource_fingerprint':canonical_hash(material)}
    return {**identity,'request_hash':canonical_hash(identity)}


def fit_mechanism_run(run,data,inputs,assets,output: Path,*,shared_features=None,
                       baseline_features=None,cls_cache=None,model_config=None,
                       tokenizer=None,pretrained_encoder_state=None):
    """Fit a supplied exact Condition ID; real phase authorization is external.

    M3 TF-IDF callers must supply the declared full-unlabeled-T representation.
    M3 generic CLS caching encodes only requested support/validation rows. For
    M5 BiLSTM, provide baseline_features or the core's 30k cache is obtained on
    the exact condition. Other M5 families use unchanged fit_run.
    """
    if run.strict_no_val:raise ValueError('Mechanism runs cannot be strict E2 tasks')
    if run.model_id not in {'m2_distilbert','fixed_tfidf_svm','frozen_generic_distilbert_head','bilstm'}:
        return fit_run(run,data,inputs,assets,Path(output))
    from .execution import run_lease
    from .resources import ResourceSampler
    identity=mechanism_request_identity(run,data,inputs,Path(output).resolve().parent.parent,
        shared_features=shared_features,baseline_features=baseline_features,model_config=model_config,
        tokenizer=tokenizer,pretrained_encoder_state=pretrained_encoder_state)
    with run_lease(Path(output),identity['request_hash']):
        with ResourceSampler(run.hparams.get('device','cpu')) as resources:
            result=_fit_mechanism_unleased(run,data,inputs,assets,Path(output),identity,
                shared_features,baseline_features,cls_cache,model_config,tokenizer,pretrained_encoder_state)
        if not result.get('reused_complete'):
            result['resources']=resources.report
            atomic_write_json(Path(output)/'status.json',result,output)
    return result


def _fit_mechanism_unleased(run,data,inputs,assets,output,identity,shared_features,
                            baseline_features,cls_cache,model_config,tokenizer,pretrained_encoder_state):
    import torch
    output=output.resolve();root=output.parent.parent;started=time.monotonic()
    if (output/'status.json').exists():
        previous=read_json(output/'status.json')
        if previous.get('request_hash')!=identity['request_hash']:raise ValueError('Mechanism request identity changed')
        if previous['status']=='complete':
            _verified_bundle(output);return {**previous,'reused_complete':True}
    ids=_condition(run,data,root);by_id={row.sample_id:row for row in data.partitions['train']}
    if not set(ids)<=set(by_id):raise ValueError('Mechanism condition requests labels outside T')
    train_rows=tuple(by_id[sid] for sid in ids);train_inputs=tuple(inputs[sid] for sid in ids)
    visible_texts(train_inputs)
    validation_rows=data.partitions['select']
    if not validation_rows:raise ValueError('Mechanism checkpoint selection requires registered V')
    validation_inputs=tuple(inputs[row.sample_id] for row in validation_rows)
    y=target_matrix(train_rows,len(data.label_names),data.task)
    vy=target_matrix(validation_rows,len(data.label_names),data.task)
    hp={**run.hparams,'label_count':len(data.label_names),'device':run.hparams.get('device','cpu')}
    if hp['device'].startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    torch.set_num_threads(int(hp.get('cpu_threads',4)))
    output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(output/'status.json',{'status':'running','run_id':run.run_id,**identity,
                      'pid':os.getpid()},output)
    try:
        feature_started=time.monotonic();features=None;model=None;neural={};extra={}
        mechanism='M2' if run.model_id=='m2_distilbert' else 'M5' if run.model_id=='bilstm' else 'M3'
        if mechanism=='M3':
            if data.task!='single_label':raise ValueError('M3 weighted target risk is single-label only')
            if hp.get('unlabeled_pool_manifest')!=data.manifest_hash:
                raise ValueError('M3 full-unlabeled-T representation pool identity changed')
            weights=m3_sample_weights(y,hp['target_q'],int(hp['reference_n']))
            extra={'sample_weight_sum':float(weights.sum()),
                   'representation_unlabeled_pool_manifest':data.manifest_hash,
                   'reference_n':int(hp['reference_n']),'target_q':hp['target_q']}
        if run.model_id=='fixed_tfidf_svm':
            from .models import TraditionalClassifier
            if shared_features is None:raise ValueError('M3 requires supplied shared full-unlabeled-T TFIDF features')
            features=shared_features
            if not set(ids)<=set(features.fitted_sample_ids):raise ValueError('Shared representation omits condition rows')
            model=TraditionalClassifier('fixed_tfidf_svm',data.task,len(data.label_names),
                                       {'C':1,'tol':1e-4,'max_iter':10000})
        elif mechanism=='M5':
            from dataclasses import replace
            if baseline_features is None:
                baseline_features=_features(replace(run,hparams={**hp,'vocabulary_limit':30000}),train_inputs,root)
            expanded=shared_features if shared_features is not None else _features(run,train_inputs,root)
            features=paired_word_features(baseline_features,expanded,int(run.training_seed or 42))
            model=build_paired_bilstm(baseline_features,features,int(run.training_seed or 42),len(data.label_names),hp)
            extra={'paired_initialization':True,'shared_embedding_rows':len(baseline_features.vocabulary),
                   'actual_vocabulary_size':len(features.vocabulary)}
        else:
            from transformers import AutoTokenizer, DistilBertConfig
            source=assets['models']['distilbert']
            if model_config is None:
                if len(source.get('revision',''))!=40:raise ValueError('Immutable generic encoder revision is required')
                model_config=DistilBertConfig.from_pretrained(source['directory'],local_files_only=True)
            model_config=deepcopy(model_config);model_config.num_labels=len(data.label_names)
            model_config.problem_type='single_label_classification' if data.task=='single_label' else 'multi_label_classification'
            if tokenizer is None:tokenizer=AutoTokenizer.from_pretrained(source['directory'],local_files_only=True)
            arm=hp.get('arm','pretrained_frozen') if mechanism=='M2' else 'pretrained_frozen'
            model=build_m2_model(model_config,arm,int(run.training_seed or 42),
                pretrained_encoder_state=pretrained_encoder_state,pretrained_directory=source['directory'])
            extra={'arm':arm,'pooling':'DistilBertForSequenceClassification_CLS',
                   'initial_encoder_sha256':_state_hash(model.distilbert.state_dict()),
                   'initial_head_sha256':_state_hash({n:p for n,p in model.state_dict().items() if not n.startswith('distilbert.')})}
            if mechanism=='M3':extra.update({'representation_unlabeled_pool_manifest':data.manifest_hash,
                                            'reference_n':int(hp['reference_n']),'target_q':hp['target_q']})
        feature_seconds=time.monotonic()-feature_started;fit_started=time.monotonic()
        bundle={'model_id':run.model_id,'task':data.task,'label_count':len(data.label_names),
                'features':features,'hparams':hp,'assets':assets,'constants':{}}
        if run.model_id=='fixed_tfidf_svm':
            model.fit(features.transform(train_inputs),y,sample_weight=weights)
            before=model.scores(features.transform(train_inputs[:16]));bundle['model']=model;score_kind=model.score_kind
        else:
            if mechanism=='M3':
                cache=cls_cache if cls_cache is not None else GenericCLSCache()
                neural=_fit_m3_head(run,data,train_inputs,y,validation_inputs,vy,tokenizer,model,output,hp,cache)
            else:
                neural=_fit_epochs(run,data,train_inputs,y,validation_inputs,vy,features,tokenizer,model,output,hp,mechanism)
            before=neural['probe_scores'];bundle['hparams']=neural['hparams'];bundle['constants']=neural['constants'];score_kind='logit'
        fit_seconds=time.monotonic()-fit_started
        _joblib_atomic(bundle,output/'bundle.joblib')
        # Store the original probe itself as an auditable artifact.
        _joblib_atomic(np.asarray(before),output/'original-score-probe.joblib')
        files={p.relative_to(output).as_posix():hash_file(p) for p in output.rglob('*') if p.is_file()
               and p.name not in {'status.json','progress.json'} and not p.name.endswith('.part') and not p.name.startswith('.lease')}
        result={'status':'verifying','run_id':run.run_id,**identity,'run':asdict(run),
                'mechanism':mechanism,'score_kind':score_kind,
                'files':files,'training_unique_rows':len(ids),
                'training_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                'holdout_labels_accessed':len(validation_rows),'feature_seconds':feature_seconds,'fit_seconds':fit_seconds,
                'feature_degenerate':bool(getattr(features,'degenerate',False)),
                'reload_verified':False,'actual_model_training_run':True,'official_test_predictions_generated':False,
                **extra,**{k:v for k,v in neural.items() if k not in {'hparams','constants','probe_scores','history'}}}
        atomic_write_json(output/'status.json',result,output)
        reloaded=_score_saved(output,train_inputs[:16],allow_verifying=True)
        if not np.allclose(before,reloaded,rtol=1e-4,atol=1e-5):raise ValueError('Mechanism saved/reloaded scores differ')
        result.update({'reload_verified':True,'reload_max_absolute_difference':float(np.max(np.abs(before-reloaded))),
                       'validation':metric_bundle(vy,_score_saved(output,validation_inputs,allow_verifying=True),data.task,score_kind,
                           label_mask=hp.get('select_ap_mask') if data.task!='single_label' else None),
                       'status':'complete','total_seconds':time.monotonic()-started})
        atomic_write_json(output/'status.json',result,output)
        return result
    except Exception as exc:
        atomic_write_json(output/'status.json',{'status':'failed','run_id':run.run_id,**identity,
            'error_type':type(exc).__name__,'error':str(exc),
            'pid':os.getpid(),'official_test_predictions_generated':False},output)
        raise

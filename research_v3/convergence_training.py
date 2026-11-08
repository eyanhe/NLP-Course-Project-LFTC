"""Versioned convergence learner; legacy learners remain byte-identical."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json, bounded_path, hash_file, read_json
from .types import RunSpec


def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


@dataclass
class StopState:
    min_epochs: int
    patience: int
    min_delta: float
    best: float = -math.inf
    best_epoch: int = 0
    reference: float = -math.inf
    bad_epochs: int = 0
    epoch: int = 0
    improved: bool = False

    def update(self, epoch: int, value: float) -> bool:
        if (self.min_epochs < 1 or self.patience < 1 or self.min_delta < 0
                or not math.isfinite(value) or epoch != self.epoch + 1):
            raise ValueError('Invalid early stopping value, policy or epoch sequence')
        self.epoch=epoch; self.improved=value>self.best
        if self.improved:self.best=value;self.best_epoch=epoch
        if epoch<=self.min_epochs:
            self.reference=self.best;self.bad_epochs=0
        elif value>self.reference+self.min_delta:
            self.reference=value;self.bad_epochs=0
        else:
            self.bad_epochs+=1
        return self.bad_epochs>=self.patience


def lr_multiplier(update: int, warmup_updates: int) -> float:
    if update<0 or warmup_updates<0:raise ValueError('Negative update budget')
    return min(1.,update/warmup_updates) if warmup_updates else 1.


def validate_runtime(device,threads):
    import torch
    if device not in {'cpu','cuda'}:raise ValueError('Unknown execution device')
    available=int(os.environ.get('SLURM_CPUS_PER_TASK',os.environ.get('SLURM_CPUS_ON_NODE',threads)))
    if not 1<=threads<=available:raise ValueError('Model threads exceed allocated CPUs')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Allocated CUDA device unavailable')


def score_fp32(model,model_id,inputs,features,tokenizer,hp,constants):
    import torch
    from .train import _batch,_logits
    model.eval();outputs=[];micro=int(hp['micro_batch'])
    with torch.inference_mode():
        for start in range(0,len(inputs),micro):
            batch=_batch(model_id,inputs[start:start+micro],features,tokenizer,hp['device'])
            outputs.append(_logits(model,model_id,batch).float().cpu().numpy())
    scores=np.concatenate(outputs,axis=0) if outputs else np.zeros((0,hp['label_count']),dtype=np.float32)
    for col,positive in constants.items():scores[:,int(col)]=30 if positive else -30
    if not np.isfinite(scores).all():raise ValueError('Nonfinite neural scores')
    return scores


def score_checkpoint(output,inputs,allow_verifying=False):
    from .train import _verified_bundle,build_model
    import torch
    status,bundle=_verified_bundle(output,allow_verifying)
    hp=bundle['hparams'];model_id=bundle['model_id'];tokenizer=None
    if model_id=='bilstm':
        from safetensors.torch import load_file
        model=build_model(model_id,bundle['task'],bundle['label_count'],bundle['assets'],hp,bundle['features'])
        model.load_state_dict(load_file(str(output/'checkpoint/model.safetensors')))
    else:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        model=AutoModelForSequenceClassification.from_pretrained(output/'checkpoint',local_files_only=True,attn_implementation='sdpa')
        tokenizer=AutoTokenizer.from_pretrained(bundle['assets']['models']['distilbert']['directory'],local_files_only=True)
    model.to(hp['device'])
    result=score_fp32(model,model_id,inputs,bundle['features'],tokenizer,hp,bundle['constants'])
    del model;gc.collect()
    if hp['device']=='cuda':torch.cuda.empty_cache()
    return result


def convergence_sources():
    names=['convergence_training.py','train.py','features.py','models.py','metrics.py',
           'types.py','inputs.py','resources.py','execution.py','storage.py','config.py']
    return {**{name:hash_file(Path(__file__).parent/name) for name in names},
            '../models_extra.py':hash_file(Path(__file__).parent.parent/'models_extra.py')}


def convergence_code_hash() -> str:
    return canonical_hash(convergence_sources())


def save_resume(path: Path, payload: dict):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    snapshot=path.with_name(path.stem+'.'+str(time.time_ns())+'.pt')
    partial=snapshot.with_suffix('.part')
    with partial.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(partial,snapshot)
    atomic_write_json(path.with_suffix('.manifest.json'),
        {'request_hash':payload['request_hash'],'snapshot':snapshot.name,'sha256':hash_file(snapshot)},path.parent)
    # The pointer switches only after its complete immutable generation exists.
    for obsolete in sorted(path.parent.glob(path.stem+'.*.pt'),key=lambda p:p.name,reverse=True)[2:]:
        obsolete.unlink()


def load_resume(path: Path, request_hash: str):
    import torch
    info=read_json(Path(path).with_suffix('.manifest.json'))
    snapshot=bounded_path(Path(path).parent/info['snapshot'],Path(path).parent)
    if info['request_hash']!=request_hash or hash_file(snapshot)!=info['sha256']:
        raise ValueError('Resume request/hash identity changed')
    saved=torch.load(snapshot,map_location='cpu',weights_only=False)
    if saved['request_hash']!=request_hash:raise ValueError('Resume request identity changed')
    return saved


class DeadlineReached(RuntimeError):
    pass


class NonfiniteTrainingLoss(ValueError):
    pass


def _check_deadline(deadline):
    if datetime.now(timezone.utc)>=deadline:raise DeadlineReached('Training hard deadline reached')


def _neural(run,data,train_inputs,y,validation_inputs,vy,features,assets,output,deadline):
    import torch
    from transformers import AutoTokenizer
    from . import train as old
    hp={**run.hparams,'label_count':len(data.label_names)}
    device=hp['device'];threads=int(hp.get('cpu_threads',8));validate_runtime(device,threads)
    torch.set_num_threads(threads)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    maximum=int(hp['max_epochs']);effective=int(hp['effective_batch']);micro=int(hp['micro_batch'])
    stopper=StopState(int(hp['min_epochs']),int(hp['patience']),float(hp['min_delta']))
    if maximum<stopper.min_epochs+stopper.patience:raise ValueError('Insufficient early-stopping safety budget')
    multilabel=data.task in {'multilabel','multi_label'}
    constants={int(c):bool(y[0,c]) for c in range(y.shape[1]) if np.all(y[:,c]==y[0,c])} if multilabel else {}
    active=[c for c in range(len(data.label_names)) if c not in constants]
    tokenizer=None if run.model_id=='bilstm' else AutoTokenizer.from_pretrained(
        assets['models']['distilbert']['directory'],local_files_only=True)
    old._seed(int(run.training_seed));model=old.build_model(run.model_id,data.task,len(data.label_names),assets,hp,features).to(device)
    cls=torch.optim.Adam if run.model_id=='bilstm' else torch.optim.AdamW
    optimizer=cls(model.parameters(),lr=float(hp['lr']),weight_decay=float(hp.get('weight_decay',0 if run.model_id=='bilstm' else .01)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,int(hp['warmup_updates'])))
    history=[];updates=0;best_state=None;best_probe=None;request=canonical_hash(asdict(run))
    resume=output/'resume.pt'
    if resume.with_suffix('.manifest.json').exists():
        saved=load_resume(resume,request)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);stopper=StopState(**saved['stopper'])
        history=saved['history'];updates=saved['updates'];best_state=saved['best_model'];best_probe=saved['best_probe']
        # Repair a best-file write interrupted after the preceding atomic resume snapshot.
        current=saved['model'];model.load_state_dict(best_state);old._checkpoint(model,run.model_id,output);model.load_state_dict(current)
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    stop_reason='patience' if stopper.bad_epochs>=stopper.patience else 'max_epochs'
    last_progress=time.monotonic()
    for epoch in range(stopper.epoch+1,maximum+1):
        if stop_reason=='patience':break
        _check_deadline(deadline)
        model.train();order=np.random.default_rng(1000003*int(run.training_seed)+epoch).permutation(len(y))
        total_loss=0.;seen=0;start_lr=float(optimizer.param_groups[0]['lr'])
        for chunks in old.accumulation_groups(order,effective,micro):
            _check_deadline(deadline);optimizer.zero_grad(set_to_none=True);group_n=sum(map(len,chunks))
            for indexes in chunks:
                subset=tuple(train_inputs[int(i)] for i in indexes)
                batch=old._batch(run.model_id,subset,features,tokenizer,device)
                target=torch.as_tensor(y[indexes],device=device)
                logits=old._logits(model,run.model_id,batch)
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits[:,active].float(),target[:,active].float())
                      if multilabel and active else logits.sum()*0 if multilabel
                      else torch.nn.functional.cross_entropy(logits.float(),target.long()))
                if not bool(torch.isfinite(loss)):raise NonfiniteTrainingLoss('Nonfinite training loss')
                (loss*(len(indexes)/group_n)).backward()
                total_loss+=float(loss.detach())*len(indexes);seen+=len(indexes)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();updates+=1
            if time.monotonic()-last_progress>=5:
                atomic_write_json(output/'progress.json',{'status':'running','epoch':epoch,'epochs':maximum,
                    'training_rows_seen':seen,'training_rows':len(y),'optimizer_updates':updates,
                    'lr':float(optimizer.param_groups[0]['lr']),'best_epoch':stopper.best_epoch,
                    'bad_epochs':stopper.bad_epochs,'pid':os.getpid()},output)
                last_progress=time.monotonic()
        scores=score_fp32(model,run.model_id,validation_inputs,features,tokenizer,hp,constants)
        metric=old.metric_bundle(vy,scores,data.task,'logit',label_mask=hp.get('select_ap_mask') if multilabel else None)
        value=metric['macro_ap' if multilabel else 'macro_f1']
        stop=stopper.update(epoch,value)
        if stopper.improved:
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_probe=score_fp32(model,run.model_id,train_inputs[:16],features,tokenizer,hp,constants)
            old._checkpoint(model,run.model_id,output)
        history.append({'epoch':epoch,'train_loss':total_loss/len(y),'validation':metric,
            'optimizer_updates':updates,'lr_start':start_lr,'lr_end':float(optimizer.param_groups[0]['lr']),
            'early_stopping':asdict(stopper)})
        save_resume(resume,{'request_hash':request,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'stopper':asdict(stopper),'history':history,'updates':updates,
            'best_model':best_state,'best_probe':best_probe,'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None})
        atomic_write_json(output/'history.json',{'epochs':history,'attempts':[],'schedule':'fixed_warmup_then_constant'},output)
        if stop:stop_reason='patience';break
    result={'hparams':hp,'constants':constants,'history':history,'epochs_completed':stopper.epoch,
        'selected_epoch':stopper.best_epoch,'optimizer_updates':updates,'effective_batch':effective,
        'micro_batch':micro,'probe_scores':best_probe,'stop_reason':stop_reason,
        'convergence_verified':stop_reason=='patience','validation_metric':stopper.best}
    del optimizer,scheduler,model,best_state;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return result


def fit_convergence(entry,data,inputs,assets,output: Path,deadline: datetime,start_cutoff=None) -> dict:
    from . import train as old
    from .execution import run_lease
    from .resources import ResourceSampler
    run=RunSpec(**entry['run']);output=Path(output).resolve();root=output.parent.parent
    if run.strict_no_val or run.model_id not in {'bilstm','distilbert'}:
        raise ValueError('Convergence fitting requires a V-selected neural model')
    if run.code_hash!=convergence_code_hash() or run.run_id!=canonical_hash(asdict(replace(run,run_id=''))):
        raise ValueError('Convergence learner or request identity changed')
    request=canonical_hash(asdict(run));status_path=output/'status.json'
    with run_lease(output,request):
        if status_path.exists():
            previous=read_json(status_path)
            if previous.get('request_hash')!=request:raise ValueError('Saved run request identity changed')
            if previous['status']=='complete':old._verified_bundle(output);return previous
        _check_deadline(deadline)
        if start_cutoff is not None and datetime.now(timezone.utc)>=start_cutoff:
            raise PermissionError('New-fit cutoff reached before fitting')
        ids=old._condition(run,data,root);byid={row.sample_id:row for row in data.partitions['train']}
        validation=data.partitions['select'];roles=run.hparams['role_ids_hash']
        if (not set(ids)<=set(byid) or not validation or roles['train']!=ids_hash(ids)
                or roles['select']!=ids_hash([r.sample_id for r in validation]) or roles['discovery']!=ids_hash([])):
            raise ValueError('Exact T/V roles or label access identity changed')
        if set(ids)&{r.sample_id for r in validation}:raise ValueError('T/V overlap')
        training=tuple(byid[sid] for sid in ids);ti=tuple(inputs[sid] for sid in ids)
        vi=tuple(inputs[row.sample_id] for row in validation);old.visible_texts(ti)
        y=old.target_matrix(training,len(data.label_names),data.task);vy=old.target_matrix(validation,len(data.label_names),data.task)
        started=time.monotonic()
        atomic_write_json(status_path,{'status':'running','run_id':run.run_id,'request_hash':request,
            'run':asdict(run),'pid':os.getpid(),'official_test_predictions_generated':False},output)
        try:
            with ResourceSampler(run.hparams['device']) as resources:
                if start_cutoff is not None and datetime.now(timezone.utc)>=start_cutoff:
                    raise PermissionError('New-fit cutoff reached before feature/model fitting')
                features=old._features(run,ti,root);feature_seconds=time.monotonic()-started;fit_started=time.monotonic()
                neural=_neural(run,data,ti,y,vi,vy,features,assets,output,deadline)
                fit_seconds=time.monotonic()-fit_started
                old._joblib_atomic({'model_id':run.model_id,'task':data.task,'label_count':len(data.label_names),
                    'features':features,'hparams':neural['hparams'],'assets':assets,'constants':neural['constants']},output/'bundle.joblib')
                files={p.relative_to(output).as_posix():hash_file(p) for p in output.rglob('*')
                    if p.is_file() and p.name not in {'status.json','progress.json'} and not p.name.startswith('.lease')
                    and not p.name.endswith('.part')}
                result={k:v for k,v in neural.items() if k not in {'probe_scores','hparams','constants'}}
                result.update(status='verifying',run_id=run.run_id,request_hash=request,run=asdict(run),files=files,
                    training_unique_rows=len(ids),training_ids_sha256=ids_hash(ids),holdout_labels_accessed=len(validation),
                    score_kind='logit',feature_seconds=feature_seconds,fit_seconds=fit_seconds,
                    reload_verified=False,actual_model_training_run=True,official_test_predictions_generated=False)
                atomic_write_json(status_path,result,output)
                actual=score_checkpoint(output,ti[:16],allow_verifying=True)
                if not np.allclose(actual,neural['probe_scores'],rtol=1e-4,atol=1e-5):raise ValueError('Best checkpoint reload differs')
                result['reload_verified']=True;result['reload_max_absolute_difference']=float(np.max(np.abs(actual-neural['probe_scores'])))
                vs=score_checkpoint(output,vi,allow_verifying=True)
                result['validation']=old.metric_bundle(vy,vs,data.task,'logit',label_mask=run.hparams.get('select_ap_mask') if data.task!='single_label' else None)
                key='macro_f1' if data.task=='single_label' else 'macro_ap'
                if not math.isclose(result['validation'][key],neural['validation_metric'],abs_tol=1e-10):
                    raise ValueError('Reloaded V metric differs from selected best')
            result.update(status='complete',total_seconds=time.monotonic()-started,resources=resources.report)
            atomic_write_json(status_path,result,output)
            atomic_write_json(output/'progress.json',{'status':'complete','stop_reason':result['stop_reason'],
                'epochs_completed':result['epochs_completed'],'selected_epoch':result['selected_epoch']},output)
            return result
        except Exception as exc:
            atomic_write_json(status_path,{'status':'incomplete' if isinstance(exc,DeadlineReached) else 'failed',
                'run_id':run.run_id,'run':asdict(run),'request_hash':request,
                'stop_reason':'deadline' if isinstance(exc,DeadlineReached) else 'nonfinite' if isinstance(exc,NonfiniteTrainingLoss) else 'failed',
                'error_type':type(exc).__name__,'error':str(exc),'convergence_verified':False,
                'official_test_predictions_generated':False},output)
            raise

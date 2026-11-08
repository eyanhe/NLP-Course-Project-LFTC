"""V2: four-class news, Word2Vec/SVM, BiLSTM and comprehensive metrics.

Keeps V1 checkpoints and test records unchanged. V2 model selection uses validation
only; the binary benchmark is an extension after initial test inspection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import pipeline as core
from models_extra import BiLSTM,words,mean_vectors,word_rows,word_batch

ROOT=core.ROOT
CFG=core.read_json(ROOT/'config-v2.json')
SOURCES=core.NAMES+['ag_news']
MODELS=['logistic_regression','linear_svm','word2vec_svm','bilstm','distilbert']
LABELS={s:['Negative','Positive'] for s in core.NAMES}
LABELS['ag_news']=['World','Sports','Business','Sci/Tech']
OUT=ROOT/'results-v2'

def metric_set(y,pred,source):
    labels=list(range(len(LABELS[source])))
    result={'accuracy':float(accuracy_score(y,pred)), 'mcc':float(matthews_corrcoef(y,pred)),
            'classification_report':classification_report(y,pred,labels=labels,target_names=LABELS[source],output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}
    for average in ['macro','micro','weighted']:
        p,r,f,_=precision_recall_fscore_support(y,pred,labels=labels,average=average,zero_division=0)
        result.update({f'{average}_precision':float(p),f'{average}_recall':float(r),f'{average}_f1':float(f)})
    return result

def load_split(source,split):
    if source!='ag_news': return core.load_split(source,split)
    path=ROOT/'data-v2'/source/f'{split}.jsonl'
    expected=core.read_json(ROOT/'data-v2'/'audit.json')['split_sha256'][split]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise RuntimeError('News sample file changed')
    return pd.read_json(path,lines=True)

def signature():
    return {'v2_config_sha256':core.canonical_hash(CFG), 'v1_manifests':core.split_hashes(),
            'news_manifests':core.read_json(ROOT/'data-v2'/'audit.json')['split_sha256'],
            'model_revision':core.read_json(ROOT/'assets.lock.json')['model']['revision']}

def check(meta):
    for key,value in signature().items():
        if meta.get(key)!=value: raise RuntimeError(f'V2 provenance changed: {key}')

def directory(source,name,seed=None):
    if source!='ag_news' and name in core.MODEL_NAMES: return core.model_dir(source,name,seed if name=='distilbert' else None)
    return ROOT/'models-v2'/source/(name if seed is None else f'{name}_seed{seed}')

def prepare_news():
    audit_path=ROOT/'data-v2'/'audit.json'
    if audit_path.exists():
        if core.read_json(audit_path)['config_sha256']!=core.canonical_hash(CFG): raise RuntimeError('News config changed')
        for split in ['train','validation','test']:load_split('ag_news',split)
        return
    info=core.read_json(ROOT/'assets.lock.json')['datasets']['ag_news']
    frames=[]
    for split in ['train','test']:
        files=[ROOT/'assets'/'ag_news'/f for f in info['files'] if Path(f).name.startswith(split)]
        f=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
        if set(f.label.unique())!={0,1,2,3}:raise RuntimeError('Unexpected AG News labels')
        f['sample_id']=[f'ag_news/{split}/{i}' for i in range(len(f))]
        f['official_split']=split;f['text']=f.text.map(core.clean_text);f['text_hash']=f.text.map(core.text_hash)
        frames.append(f)
    merged=pd.concat(frames,ignore_index=True)
    conflicts=merged.groupby('text_hash').label.nunique()
    eligible=merged[~merged.text_hash.isin(conflicts[conflicts>1].index)&(merged.text.str.len()>0)]
    eligible=eligible.copy()
    eligible['priority']=eligible.official_split.map({'test':0,'train':1})
    eligible=eligible.sort_values(['priority','sample_id'],kind='stable').drop_duplicates('text_hash')
    rng=np.random.default_rng(CFG['sample_seed'])
    def sample(pool,n):
        ids=np.concatenate([rng.choice(pool.index[pool.label==c],n//4,replace=False) for c in range(4)])
        return pool.loc[rng.permutation(ids)].copy()
    pool=eligible[eligible.official_split=='train']
    val=sample(pool,CFG['ag_news_validation_size']);train=sample(pool.drop(index=val.index),CFG['ag_news_train_size'])
    test=sample(eligible[eligible.official_split=='test'],CFG['ag_news_test_size'])
    folder=ROOT/'data-v2'/'ag_news';folder.mkdir(parents=True,exist_ok=True)
    audit={'created_utc':core.utcnow(),'config_sha256':core.canonical_hash(CFG),'revision':info['revision'],
           'raw_rows':len(merged),'eligible_unique_rows':len(eligible),'conflicting_hashes':int((conflicts>1).sum()),
           'label_names':LABELS['ag_news'],'split_sha256':{},'splits':{},
           'dedup_policy':'Normalized SHA256, official test precedence; conflicting labels removed; sampled train/validation/test globally unique within task.'}
    selected=[]
    for split,f in [('train',train),('validation',val),('test',test)]:
        f=f[['sample_id','official_split','text','label','text_hash']].copy();f['split']=split
        p=folder/f'{split}.jsonl';f.to_json(p,orient='records',lines=True,force_ascii=False)
        audit['split_sha256'][split]=hashlib.sha256(p.read_bytes()).hexdigest()
        audit['splits'][split]={'n':len(f),'class_counts':{LABELS['ag_news'][c]:int((f.label==c).sum()) for c in range(4)}}
        selected.append(f)
    manifest=pd.concat(selected,ignore_index=True)
    if manifest.text_hash.duplicated().any():raise AssertionError('News leakage')
    legacy=pd.read_csv(ROOT/'data'/'manifest.csv')
    audit['cross_task_exact_overlap']=len(set(legacy.text_hash)&set(manifest.text_hash))
    if audit['cross_task_exact_overlap']:raise AssertionError('Cross-task exact overlap must be reviewed')
    manifest.drop(columns='text').to_csv(ROOT/'data-v2'/'manifest.csv',index=False)
    core.write_json(audit_path,audit)
    core.log('AG News prepared: 6000 train, 1000 validation, 1000 test; four balanced classes.')

def word_assets(source):
    from gensim.models import Word2Vec
    dest=ROOT/'models-v2'/source/'word2vec';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metadata.json').exists():check(core.read_json(dest/'metadata.json'));return joblib.load(dest/'features.joblib')
    train=load_split(source,'train');sentences=[words(t) for t in train.text]
    tick=time.perf_counter()
    embedding=Word2Vec(sentences,vector_size=CFG['word2vec_dim'],window=CFG['word2vec_window'],
        min_count=CFG['word2vec_min_count'],sg=1,workers=1,epochs=CFG['word2vec_epochs'],seed=CFG['word2vec_seed'],hashfxn=lambda s:int(hashlib.md5(s.encode()).hexdigest(),16))
    counts=Counter(w for sent in sentences for w in sent)
    tokens=sorted(embedding.wv.key_to_index,key=lambda t:(-counts[t],t))[:CFG['word_vocabulary_limit']-2]
    vocabulary={token:i+2 for i,token in enumerate(tokens)}
    vectors={w:embedding.wv[w].copy() for w in tokens}
    rng=np.random.default_rng(2026)
    matrix=rng.normal(0,.02,(len(tokens)+2,CFG['word2vec_dim'])).astype(np.float32);matrix[0]=0
    for token,index in vocabulary.items():matrix[index]=vectors[token]
    assets={'vectors':vectors,'vocabulary':vocabulary,'matrix':matrix}
    joblib.dump(assets,dest/'features.joblib')
    core.write_json(dest/'metadata.json',{**signature(),'created_utc':core.utcnow(),'source':source,
        'fit_seconds':time.perf_counter()-tick,'vocabulary_size':len(vocabulary),'training_text_only':True,
        'scope':'Word2Vec SGNS; single worker, deterministic vocabulary; shared by mean-vector SVM and BiLSTM initialization.'})
    return assets

def train_classical(source,name):
    dest=directory(source,name)
    if (dest/'metadata.json').exists():check(core.read_json(dest/'metadata.json'));return
    train,val=load_split(source,'train'),load_split(source,'validation')
    tick=time.perf_counter();candidates=[];best=-1;winner=None
    assets=word_assets(source) if name=='word2vec_svm' else None
    if assets:
        x_train=mean_vectors(train.text.tolist(),assets['vectors']);x_val=mean_vectors(val.text.tolist(),assets['vectors'])
    for ngram in ([None] if assets else [(1,1),(1,2)]):
        for c in CFG['linear_C']:
            clf=LogisticRegression(C=c,max_iter=3000,solver='lbfgs') if name=='logistic_regression' else LinearSVC(C=c,max_iter=10000,random_state=42)
            model=make_pipeline(StandardScaler(),clf) if assets else make_pipeline(TfidfVectorizer(ngram_range=ngram,max_features=30000,min_df=2,sublinear_tf=True),clf)
            start=time.perf_counter()
            with threadpool_limits(limits=4):model.fit(x_train if assets else train.text,train.label)
            fit=time.perf_counter()-start
            pred=model.predict(x_val if assets else val.text);metric=metric_set(val.label,pred,source)
            candidate={'C':c,'ngram':ngram,'fit_seconds':fit,'validation_macro_f1':metric['macro_f1']};candidates.append(candidate)
            if metric['macro_f1']>best:best=metric['macro_f1'];winner=model;selected=candidate
    dest.mkdir(parents=True,exist_ok=True)
    bundle={'classifier':winner,'vectors':assets['vectors']} if assets else winner
    joblib.dump(bundle,dest/'model.joblib')
    core.write_json(dest/'metadata.json',{**signature(),'created_utc':core.utcnow(),'source':source,'model':name,
        'selected':selected,'candidates':candidates,'search_wall_seconds':time.perf_counter()-tick,'model_bytes':(dest/'model.joblib').stat().st_size})
    core.log(f'{source}/{name}: selected validation F1={best:.4f}')

def train_deep(source,name,lr,seed,dest,class_weights=None):
    if (dest/'metadata.json').exists():
        meta=core.read_json(dest/'metadata.json');check(meta);return meta
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    device=core.torch_setup(seed)
    train,val=load_split(source,'train'),load_split(source,'validation')
    start=time.perf_counter();is_bert=name=='distilbert';n_classes=len(LABELS[source]);batch_size=CFG['batch_size'] if is_bert else CFG['bilstm_batch_size']
    if is_bert:
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets'/'distilbert',local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(ROOT/'assets'/'distilbert',num_labels=n_classes,local_files_only=True,
            attn_implementation='eager',id2label=dict(enumerate(LABELS[source])),label2id={v:i for i,v in enumerate(LABELS[source])}).to(device)
        train_loader=core.loader(core.encode(train,tokenizer),tokenizer,batch_size,shuffle=True,seed=seed)
        epochs=CFG['transformer_epochs']
    else:
        assets=word_assets(source);tokenizer=assets['vocabulary']
        model=BiLSTM(assets['matrix'],CFG['bilstm_hidden'],n_classes,CFG['bilstm_dropout']).to(device)
        encoded=word_rows(train.text,tokenizer,CFG['max_length'])
        train_loader=torch.utils.data.DataLoader(list(zip(encoded,train.label.tolist())),batch_size=batch_size,shuffle=True,collate_fn=word_batch,generator=torch.Generator().manual_seed(seed))
        epochs=CFG['bilstm_epochs']
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01 if is_bert else 0)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(len(train_loader)*epochs*.1),len(train_loader)*epochs) if is_bert else None
    scaler=torch.cuda.amp.GradScaler(enabled=is_bert and device.type=='cuda')
    dest.mkdir(parents=True,exist_ok=True);best=-1;history=[]
    torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
    for epoch in range(1,epochs+1):
        model.train();losses=[];epoch_start=time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=is_bert and device.type=='cuda'):
                if is_bert:
                    moved={k:v.to(device) for k,v in batch.items()}
                    if class_weights is None:loss=model(**moved).loss
                    else:
                        labels=moved.pop('labels');logits=model(**moved).logits
                        loss=torch.nn.functional.cross_entropy(logits,labels,weight=torch.tensor(class_weights,device=device,dtype=logits.dtype))
                else:
                    ids,lengths,labels=batch;logits=model(ids.to(device),lengths)
                    loss=torch.nn.functional.cross_entropy(logits,labels.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scheduler and scaler.get_scale()>=previous:scheduler.step()
            losses.append(float(loss.detach()))
        pred,_=predict_deep(model,tokenizer,val.text.tolist(),name,str(device))
        metric=metric_set(val.label,pred,source)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_macro_f1':metric['macro_f1'],'epoch_seconds':time.perf_counter()-epoch_start})
        core.log(f'{source}/{name} lr={lr:g} seed={seed} epoch={epoch}: loss={np.mean(losses):.4f} val F1={metric["macro_f1"]:.4f}')
        if metric['macro_f1']>best:
            best=metric['macro_f1'];best_epoch=epoch
            if is_bert:model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest)
            else:
                torch.save(model.state_dict(),dest/'model.pt')
                joblib.dump({'vocabulary':tokenizer,'matrix_shape':assets['matrix'].shape,'n_classes':n_classes},dest/'tokens.joblib')
    meta={**signature(),'created_utc':core.utcnow(),'source':source,'model':name,'seed':seed,'learning_rate':lr,'best_epoch':best_epoch,
        'best_validation_macro_f1':best,'fit_wall_seconds':time.perf_counter()-start,'history':history,
        'class_weights':class_weights,'model_bytes':sum(f.stat().st_size for f in dest.iterdir() if f.is_file()),'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else None}
    core.write_json(dest/'metadata.json',meta)
    del model,optimizer,scaler;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return meta

def predict_deep(model,tokenizer,texts,name,device='cpu'):
    model.eval();out=[]
    if name=='distilbert':
        rows=core.encode(pd.DataFrame({'text':texts,'label':[0]*len(texts)}),tokenizer)
        batches=core.loader(rows,tokenizer,32)
    else:
        rows=word_rows(texts,tokenizer,CFG['max_length'])
        batches=torch.utils.data.DataLoader(list(zip(rows,[0]*len(rows))),batch_size=64,collate_fn=word_batch)
    with torch.inference_mode():
        for batch in batches:
            if name=='distilbert':
                batch.pop('labels');logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
            else:
                ids,lengths,_=batch;logits=model(ids.to(device),lengths)
            out.extend(logits.float().softmax(-1).cpu().tolist())
    probabilities=np.asarray(out);return probabilities.argmax(axis=1),probabilities

def train_all():
    for source in SOURCES:
        if source=='ag_news':
            for name in ['logistic_regression','linear_svm']:train_classical(source,name)
        train_classical(source,'word2vec_svm')
        for name in (['bilstm','distilbert'] if source=='ag_news' else ['bilstm']):
            candidates=[]
            for lr in (CFG['transformer_learning_rates'] if name=='distilbert' else CFG['bilstm_learning_rates']):
                candidate=ROOT/'.cache'/'v2-candidates'/source/f'{name}_{lr:g}_42'
                meta=train_deep(source,name,lr,42,candidate);candidates.append((meta,candidate))
            best,path=max(candidates,key=lambda p:p[0]['best_validation_macro_f1'])
            final=directory(source,name,42)
            if not final.exists():shutil.copytree(path,final)
            for seed in CFG['training_seeds'][1:]:train_deep(source,name,best['learning_rate'],seed,directory(source,name,seed))

def load_model(source,name,seed=42,device='cpu'):
    if source!='ag_news' and name in core.MODEL_NAMES:return core.load_model(source,name,seed,device)
    dest=directory(source,name,seed if name in ['bilstm','distilbert'] else None)
    if name=='distilbert':
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(dest,local_files_only=True,attn_implementation='eager').to(device).eval(),AutoTokenizer.from_pretrained(dest,local_files_only=True)
    if name=='bilstm':
        tokens=joblib.load(dest/'tokens.joblib')
        model=BiLSTM(np.zeros(tokens['matrix_shape'],dtype=np.float32),CFG['bilstm_hidden'],tokens['n_classes'],CFG['bilstm_dropout'])
        model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
        return model.to(device).eval(),tokens['vocabulary']
    return joblib.load(dest/'model.joblib'),None

def predict(model,tokenizer,texts,name,device='cpu'):
    texts=[core.clean_text(t) for t in texts]
    if name in ['distilbert','bilstm']:return (*predict_deep(model,tokenizer,texts,name,device),'probability')
    if name=='word2vec_svm':
        classifier=model['classifier'];x=mean_vectors(texts,model['vectors'])
    else:classifier=model;x=texts
    pred=classifier.predict(x)
    return pred,(classifier.predict_proba(x) if name=='logistic_regression' else classifier.decision_function(x)),('probability' if name=='logistic_regression' else 'margin')

def freeze():
    artifacts=[]
    for source in SOURCES:
        for name in MODELS:
            for seed in (CFG['training_seeds'] if name in ['bilstm','distilbert'] else [None]):
                dest=directory(source,name,seed)
                meta=core.read_json(dest/'metadata.json')
                if source=='ag_news' or name not in core.MODEL_NAMES:check(meta)
                else:core.assert_provenance(meta)
                weights=dest/('model.pt' if name=='bilstm' else 'model.safetensors' if name=='distilbert' else 'model.joblib')
                artifacts.append({'source':source,'model':name,'seed':seed,'path':str(dest.relative_to(ROOT)),
                    'weight_file':weights.name,'sha256':hashlib.sha256(weights.read_bytes()).hexdigest()})
    obj={**signature(),'created_utc':core.utcnow(),'artifacts':artifacts,
         'task_scope':'Sentiment and four-class news evaluated separately; no cross-task transfer.',
         'prior_test_exposure':'Binary V1 tests inspected before this extension was requested. Added binary methods are exploratory comparisons; AG News test is newly reserved.'}
    path=OUT/'protocol.freeze.json'
    if path.exists():
        old=core.read_json(path);check(old)
        if old['artifacts']!=artifacts:raise RuntimeError('Frozen V2 checkpoints changed')
    else:core.write_json(path,obj)
    core.log(f'V2 protocol frozen: {len(artifacts)} final artifacts.')

def evaluate():
    frozen=core.read_json(OUT/'protocol.freeze.json');check(frozen)
    completed=OUT/'evaluation.json'
    if completed.exists() and all((OUT/f).exists() for f in ['metrics.csv','predictions.csv','class_metrics.csv']):check(core.read_json(completed));return
    device='cuda' if torch.cuda.is_available() else 'cpu';rows=[];records=[]
    old=core.read_json(ROOT/'results'/'evaluation.json')['rows']
    old_predictions=pd.read_csv(ROOT/'results'/'predictions.csv')
    # Recompute all requested statistics from the V1 stored labels/predictions, without retraining.
    for row in old:
        subset=old_predictions[(old_predictions.source==row['source'])&(old_predictions.target==row['target'])&(old_predictions.model==row['model'])]
        if row['seed'] is not None:subset=subset[subset.seed==row['seed']]
        rows.append({k:row[k] for k in ['source','target','model','seed','test_n']}|metric_set(subset.label,subset.prediction,row['target']))
    for artifact in frozen['artifacts']:
        source,name,seed=artifact['source'],artifact['model'],artifact['seed']
        if source!='ag_news' and name in core.MODEL_NAMES:continue
        dest=ROOT/artifact['path']
        if hashlib.sha256((dest/artifact['weight_file']).read_bytes()).hexdigest()!=artifact['sha256']:raise RuntimeError('Checkpoint changed')
        test=load_split(source,'test');model,tok=load_model(source,name,seed,device)
        pred,score,kind=predict(model,tok,test.text.tolist(),name,device)
        result=metric_set(test.label,pred,source)
        rows.append({'source':source,'target':source,'model':name,'seed':seed,'test_n':len(test),**result})
        records.append(pd.DataFrame({'sample_id':test.sample_id,'source':source,'target':source,'model':name,'seed':seed,
            'label':test.label,'prediction':pred,'score':[json.dumps(np.asarray(s).tolist()) for s in score],'score_type':kind}))
        core.log(f'V2 TEST {source}/{name} seed={seed}: P={result["macro_precision"]:.4f} R={result["macro_recall"]:.4f} F1={result["macro_f1"]:.4f}')
        del model,tok;gc.collect();torch.cuda.empty_cache() if device=='cuda' else None
    core.write_json(completed,{**signature(),'created_utc':core.utcnow(),'rows':rows})
    pd.DataFrame([{k:v for k,v in row.items() if k not in ['classification_report','confusion_matrix']} for row in rows]).to_csv(OUT/'metrics.csv',index=False)
    pd.concat([old_predictions,*records],ignore_index=True).to_csv(OUT/'predictions.csv',index=False)
    per_class=[]
    for row in rows:
        for label in LABELS[row['target']]:
            per_class.append({k:row[k] for k in ['source','target','model','seed']}|{'class':label,**row['classification_report'][label]})
    pd.DataFrame(per_class).to_csv(OUT/'class_metrics.csv',index=False)

def benchmark():
    frozen=core.read_json(OUT/'protocol.freeze.json');check(frozen)
    path=OUT/'benchmark.json'
    if path.exists():check(core.read_json(path));return
    original=core.read_json(ROOT/'results'/'benchmark.json')['rows'];rows=original.copy()
    sample_ids=original[0]['sample_ids'];pool=pd.concat([core.load_split(n,'test') for n in core.NAMES]).set_index('sample_id')
    texts=pool.loc[sample_ids].text.tolist();torch.set_num_threads(4)
    for source in SOURCES:
        for name in MODELS:
            if source!='ag_news' and name in core.MODEL_NAMES:continue
            model,tok=load_model(source,name,42,'cpu');elapsed=[]
            with threadpool_limits(limits=4):
                for text in texts[:20]:predict(model,tok,[text],name)
                for _ in range(5):
                    for text in texts:
                        start=time.perf_counter();predict(model,tok,[text],name);elapsed.append((time.perf_counter()-start)*1000)
            rows.append({'source':source,'model':name,'device':'cpu','threads':4,'batch_size':1,'n_unique_texts':200,'repeats':5,
                'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),'sample_ids':sample_ids,
                'scope':'Common V1 review workload, including news-trained models. Cleaning + feature conversion + forward, no loading.'})
            core.log(f'V2 CPU {source}/{name}: {np.median(elapsed):.2f} ms')
            del model,tok;gc.collect()
    core.write_json(path,{**signature(),'created_utc':core.utcnow(),'rows':rows,'note':'Legacy measurements reused. Common review workload is not news-serving latency.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','freeze','evaluate','benchmark','all']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stages={'prepare':prepare_news,'train':train_all,'freeze':freeze,'evaluate':evaluate,'benchmark':benchmark}
    if args.stage=='all':
        for stage in stages.values():stage()
    else:stages[args.stage]()

if __name__=='__main__':main()

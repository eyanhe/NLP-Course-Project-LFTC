"""MultiNLI sentence-pair benchmark: fixed-budget matched/mismatched evaluation."""
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
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix,matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import pipeline as core
from models_extra import words
from pair_models import PairTfidf,PairWordMeans,PairBiLSTM,pair_rows,pair_collate

ROOT=core.ROOT
CFG=core.read_json(ROOT/'config-hard.json')
LABELS=CFG['labels']
MODELS=['logistic_regression','linear_svm','word2vec_svm','bilstm','distilbert']
OUT=ROOT/'results-hard'
DATA=ROOT/'data-hard'

def metrics(y,pred):
    result={'accuracy':float(accuracy_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),
        'classification_report':classification_report(y,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}
    for avg in ['macro','weighted','micro']:
        p,r,f,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],average=avg,zero_division=0)
        result.update({avg+'_precision':float(p),avg+'_recall':float(r),avg+'_f1':float(f)})
    return result

def pairs(frame):return frame[['premise','hypothesis']].to_dict('records')
def load_split(split):
    p=DATA/f'{split}.jsonl';audit=core.read_json(DATA/'audit.json')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=audit['split_sha256'][split]:raise RuntimeError('MultiNLI split changed')
    return pd.read_json(p,lines=True)
def signature():
    return {'config_sha256':core.canonical_hash(CFG),'split_sha256':core.read_json(DATA/'audit.json')['split_sha256'],
        'model_revision':core.read_json(ROOT/'assets.lock.json')['model']['revision']}
def check(meta):
    for k,v in signature().items():
        if meta.get(k)!=v:raise RuntimeError('Stale hard benchmark: '+k)
def directory(name,seed=None):return ROOT/'models-hard'/(name if seed is None else f'{name}_seed{seed}')

def prepare():
    if (DATA/'audit.json').exists():
        if core.read_json(DATA/'audit.json')['config_sha256']!=core.canonical_hash(CFG):raise RuntimeError('Hard protocol changed')
        for s in ['train','validation','matched','mismatched']:load_split(s)
        return
    info=core.read_json(ROOT/'assets.lock.json')['datasets']['mnli'];frames=[]
    for priority,split in enumerate(['validation_matched','validation_mismatched','train']):
        files=[ROOT/'assets'/'mnli'/f for f in info['files'] if Path(f).name.startswith(split)]
        frame=pd.concat([pd.read_parquet(f,columns=['premise','hypothesis','label','genre','pairID']) for f in files],ignore_index=True)
        frame['official_split']=split;frame['priority']=priority
        frame['sample_id']=[f'mnli/{split}/{i}' for i in range(len(frame))]
        frame.premise=frame.premise.map(core.clean_text);frame.hypothesis=frame.hypothesis.map(core.clean_text)
        frame['premise_hash']=frame.premise.map(core.text_hash)
        frame['pair_hash']=[hashlib.sha256((p.casefold()+'\x00'+h.casefold()).encode()).hexdigest() for p,h in zip(frame.premise,frame.hypothesis)]
        frames.append(frame)
    original=pd.concat(frames,ignore_index=True)
    valid=original[original.label.isin([0,1,2])&(original.premise.str.len()>0)&(original.hypothesis.str.len()>0)].copy()
    conflicts=valid.groupby('pair_hash').label.nunique();valid=valid[~valid.pair_hash.isin(conflicts[conflicts>1].index)]
    first=valid.groupby('premise_hash').priority.min()
    valid=valid[valid.priority==valid.premise_hash.map(first)].drop_duplicates('pair_hash')
    rng=np.random.default_rng(CFG['sample_seed'])
    def sample(frame,n):
        indices=np.concatenate([rng.choice(frame.index[frame.label==c],n//3,replace=False) for c in range(3)])
        return frame.loc[rng.permutation(indices)].copy()
    pool=valid[valid.official_split=='train']
    val=sample(pool,CFG['validation_size'])
    train=sample(pool[~pool.premise_hash.isin(val.premise_hash)],CFG['train_size'])
    matched=sample(valid[valid.official_split=='validation_matched'],CFG['matched_evaluation_size'])
    mismatched=sample(valid[valid.official_split=='validation_mismatched'],CFG['mismatched_evaluation_size'])
    DATA.mkdir(exist_ok=True)
    audit={'created_utc':core.utcnow(),'config_sha256':core.canonical_hash(CFG),'dataset_revision':info['revision'],
        'original_counts':original.groupby('official_split').size().to_dict(),'eligible_rows':len(valid),
        'conflicting_pair_hashes':int((conflicts>1).sum()),'split_sha256':{},'splits':{},
        'evaluation_caveat':'Official validation pools are reserved evaluation sets for this course experiment; official hidden test results are not available.',
        'grouping':'Premise hashes, not just exact pairs, never cross train/validation/matched/mismatched boundaries.'}
    selected=[]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'assets'/'distilbert',local_files_only=True)
    for split,f in [('train',train),('validation',val),('matched',matched),('mismatched',mismatched)]:
        f=f.drop(columns='priority').copy();f['split']=split
        path=DATA/f'{split}.jsonl';f.to_json(path,orient='records',lines=True,force_ascii=False)
        length=[]
        for i in range(0,len(f),256):
            length.extend(len(x) for x in tok(f.premise.iloc[i:i+256].tolist(),f.hypothesis.iloc[i:i+256].tolist(),truncation=False,verbose=False)['input_ids'])
        audit['split_sha256'][split]=hashlib.sha256(path.read_bytes()).hexdigest()
        audit['splits'][split]={'n':len(f),'classes':{LABELS[c]:int((f.label==c).sum()) for c in range(3)},
            'genres':{str(k):int(v) for k,v in f.genre.value_counts().items()},'unique_premises':int(f.premise_hash.nunique()),
            'median_pair_tokens':float(np.median(length)),'fraction_exceeding_256_subwords':float(np.mean(np.array(length)>256))}
        selected.append(f)
    joined=pd.concat(selected,ignore_index=True)
    if joined.pair_hash.duplicated().any():raise AssertionError('Pair leakage')
    if (joined.groupby('premise_hash').split.nunique()>1).any():raise AssertionError('Premise group leakage')
    joined.drop(columns=['premise','hypothesis']).to_csv(DATA/'manifest.csv',index=False)
    core.write_json(DATA/'audit.json',audit)
    core.log('MultiNLI prepared: premise-group separation; 6000 train, 999 validation, 999 matched and 999 mismatched.')

def word_assets():
    dest=directory('word2vec');dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metadata.json').exists():check(core.read_json(dest/'metadata.json'));return joblib.load(dest/'features.joblib')
    from gensim.models import Word2Vec
    train=load_split('train');sentences=[words(t) for t in train.premise]+[words(t) for t in train.hypothesis]
    tick=time.perf_counter()
    model=Word2Vec(sentences,vector_size=100,window=5,min_count=CFG['word2vec_min_count'],workers=1,sg=1,epochs=CFG['word2vec_epochs'],seed=2026,hashfxn=lambda s:int(hashlib.md5(s.encode()).hexdigest(),16))
    counts=Counter(w for row in sentences for w in row)
    tokens=sorted(model.wv.key_to_index,key=lambda w:(-counts[w],w))[:CFG['vocabulary_limit']-2]
    vocabulary={w:i+2 for i,w in enumerate(tokens)};vectors={w:model.wv[w].copy() for w in tokens}
    matrix=np.random.default_rng(2026).normal(0,.02,(len(tokens)+2,100)).astype(np.float32);matrix[0]=0
    for w,i in vocabulary.items():matrix[i]=vectors[w]
    result={'vocabulary':vocabulary,'vectors':vectors,'matrix':matrix};joblib.dump(result,dest/'features.joblib')
    core.write_json(dest/'metadata.json',{**signature(),'created_utc':core.utcnow(),'fit_seconds':time.perf_counter()-tick,'training_only':True,'vocabulary_size':len(tokens)})
    return result

def train_classical(name):
    dest=directory(name)
    if (dest/'metadata.json').exists():check(core.read_json(dest/'metadata.json'));return
    train,val=load_split('train'),load_split('validation');tick=time.perf_counter();candidates=[];best=-1
    vectors=word_assets()['vectors'] if name=='word2vec_svm' else None
    for ngram in ([None] if vectors is not None else [(1,1),(1,2)]):
        for c in CFG['linear_C']:
            clf=LogisticRegression(C=c,max_iter=3000) if name=='logistic_regression' else LinearSVC(C=c,max_iter=10000,random_state=42)
            model=make_pipeline(PairWordMeans(vectors),StandardScaler(),clf) if vectors is not None else make_pipeline(PairTfidf(ngram_range=ngram),clf)
            start=time.perf_counter()
            with threadpool_limits(limits=4):model.fit(pairs(train),train.label)
            result=metrics(val.label,model.predict(pairs(val)))
            candidates.append({'C':c,'ngram':ngram,'validation_macro_f1':result['macro_f1'],'fit_and_validation_seconds':time.perf_counter()-start})
            if result['macro_f1']>best:best=result['macro_f1'];winner=model;selected=candidates[-1]
    dest.mkdir(parents=True,exist_ok=True);joblib.dump(winner,dest/'model.joblib')
    core.write_json(dest/'metadata.json',{**signature(),'created_utc':core.utcnow(),'model':name,'selected':selected,'candidates':candidates,'search_seconds':time.perf_counter()-tick})
    core.log(f'MultiNLI {name}: validation F1={best:.4f}')

def encode_transformer(records,tok,labels=None):
    encoded=tok([r['premise'] for r in records],[r['hypothesis'] for r in records],truncation='longest_first',max_length=CFG['max_length'],padding=False)
    return [{**{k:v[i] for k,v in encoded.items()},'labels':int(labels[i]) if labels is not None else 0} for i in range(len(records))]

def predict_deep(model,tokenizer,records,name,device='cpu'):
    if name=='distilbert':loader=core.loader(encode_transformer(records,tokenizer),tokenizer,32)
    else:loader=torch.utils.data.DataLoader(pair_rows(records,tokenizer),batch_size=64,collate_fn=pair_collate)
    result=[];model.eval()
    with torch.inference_mode():
        for batch in loader:
            if name=='distilbert':
                batch.pop('labels');logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
            else:
                p,pl,h,hl,_=batch;logits=model(p.to(device),pl,h.to(device),hl)
            result.extend(logits.float().softmax(-1).cpu().tolist())
    probabilities=np.asarray(result);return probabilities.argmax(axis=1),probabilities

def train_deep(name,lr,seed,dest):
    if (dest/'metadata.json').exists():m=core.read_json(dest/'metadata.json');check(m);return m
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    device=core.torch_setup(seed);train,val=load_split('train'),load_split('validation');tick=time.perf_counter();bert=name=='distilbert'
    if bert:
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets'/'distilbert',local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(ROOT/'assets'/'distilbert',local_files_only=True,num_labels=3,
            id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)},attn_implementation='eager').to(device)
        loader=core.loader(encode_transformer(pairs(train),tokenizer,train.label.tolist()),tokenizer,16,shuffle=True,seed=seed);epochs=CFG['transformer_epochs']
    else:
        assets=word_assets();tokenizer=assets['vocabulary'];model=PairBiLSTM(assets['matrix'],CFG['bilstm_hidden'],CFG['bilstm_dropout']).to(device)
        loader=torch.utils.data.DataLoader(pair_rows(pairs(train),tokenizer,labels=train.label.tolist()),batch_size=CFG['bilstm_batch_size'],shuffle=True,collate_fn=pair_collate,generator=torch.Generator().manual_seed(seed));epochs=CFG['bilstm_epochs']
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01 if bert else 0)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(len(loader)*epochs*.1),len(loader)*epochs) if bert else None
    scaler=torch.cuda.amp.GradScaler(enabled=bert and device.type=='cuda');history=[];best=-1
    dest.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train();losses=[];start=time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bert and device.type=='cuda'):
                if bert:loss=model(**{k:v.to(device) for k,v in batch.items()}).loss
                else:
                    p,pl,h,hl,y=batch;loss=torch.nn.functional.cross_entropy(model(p.to(device),pl,h.to(device),hl),y.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scheduler and scaler.get_scale()>=old:scheduler.step()
            losses.append(float(loss.detach()))
        pred,_=predict_deep(model,tokenizer,pairs(val),name,str(device));score=metrics(val.label,pred)['macro_f1']
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_macro_f1':score,'epoch_seconds':time.perf_counter()-start})
        core.log(f'MNLI {name} lr={lr:g} seed={seed} epoch={epoch} loss={np.mean(losses):.4f} val F1={score:.4f}')
        if score>best:
            best=score;best_epoch=epoch
            if bert:model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest)
            else:
                torch.save(model.state_dict(),dest/'model.pt');joblib.dump({'vocabulary':tokenizer,'matrix_shape':assets['matrix'].shape},dest/'tokens.joblib')
    meta={**signature(),'created_utc':core.utcnow(),'model':name,'seed':seed,'learning_rate':lr,'best_epoch':best_epoch,
          'best_validation_macro_f1':best,'history':history,'fit_wall_seconds':time.perf_counter()-tick}
    core.write_json(dest/'metadata.json',meta);del model,optimizer,scaler;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return meta

def train():
    for name in MODELS[:3]:train_classical(name)
    for name in MODELS[3:]:
        candidates=[]
        for lr in CFG['transformer_learning_rates'] if name=='distilbert' else CFG['bilstm_learning_rates']:
            dest=ROOT/'.cache'/'mnli-candidates'/f'{name}_{lr:g}_seed42';candidates.append((train_deep(name,lr,42,dest),dest))
        best,dest=max(candidates,key=lambda p:p[0]['best_validation_macro_f1']);target=directory(name,42)
        if not target.exists():shutil.copytree(dest,target)
        for seed in CFG['seeds'][1:]:train_deep(name,best['learning_rate'],seed,directory(name,seed))
    dest=directory('hypothesis_only_svm')
    if not (dest/'metadata.json').exists():
        training=load_split('train');model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=2,sublinear_tf=True),LinearSVC(C=1,random_state=42,max_iter=10000))
        model.fit(training.hypothesis,training.label);dest.mkdir(parents=True,exist_ok=True);joblib.dump(model,dest/'model.joblib')
        core.write_json(dest/'metadata.json',{**signature(),'created_utc':core.utcnow(),'diagnostic_only':True,'C':1,'selection':'fixed before evaluation'})

def load_model(name,seed=42,device='cpu'):
    dest=directory(name,seed if name in MODELS[3:] else None)
    if name=='distilbert':
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(dest,local_files_only=True,attn_implementation='eager').to(device).eval(),AutoTokenizer.from_pretrained(dest,local_files_only=True)
    if name=='bilstm':
        tokens=joblib.load(dest/'tokens.joblib');model=PairBiLSTM(np.zeros(tokens['matrix_shape'],np.float32),CFG['bilstm_hidden'],CFG['bilstm_dropout'])
        model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True));return model.to(device).eval(),tokens['vocabulary']
    return joblib.load(dest/'model.joblib'),None
def predict(model,tokenizer,records,name,device='cpu'):
    if name in MODELS[3:]:return (*predict_deep(model,tokenizer,records,name,device),'probability')
    x=[r['hypothesis'] for r in records] if name=='hypothesis_only_svm' else records
    return model.predict(x),(model.predict_proba(x) if name=='logistic_regression' else model.decision_function(x)),('probability' if name=='logistic_regression' else 'margin')

def freeze():
    artifacts=[]
    for name in MODELS+['hypothesis_only_svm']:
        for seed in CFG['seeds'] if name in MODELS[3:] else [None]:
            dest=directory(name,seed);check(core.read_json(dest/'metadata.json'))
            file='model.pt' if name=='bilstm' else 'model.safetensors' if name=='distilbert' else 'model.joblib'
            artifacts.append({'model':name,'seed':seed,'path':str(dest.relative_to(ROOT)),'file':file,'sha256':hashlib.sha256((dest/file).read_bytes()).hexdigest()})
    obj={**signature(),'created_utc':core.utcnow(),'artifacts':artifacts,'evaluation_pools':['matched','mismatched'],'selection':'source-train validation only; no target evaluation tuning'}
    path=OUT/'protocol.freeze.json'
    if path.exists():
        old=core.read_json(path);check(old)
        if old['artifacts']!=artifacts:raise RuntimeError('Frozen MultiNLI model changed')
    else:core.write_json(path,obj)
    core.log('MultiNLI protocol frozen before final matched/mismatched evaluation.')

def cluster_bootstrap(frame,a,b):
    rng=np.random.default_rng(2026);groups=[x.index.to_numpy() for _,x in frame.reset_index(drop=True).groupby('premise_hash')]
    y=frame.label.to_numpy();deltas=[]
    for _ in range(CFG['bootstrap_repetitions']):
        ids=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        pa=precision_recall_fscore_support(y[ids],a[ids],labels=[0,1,2],average='macro',zero_division=0)[2]
        pb=precision_recall_fscore_support(y[ids],b[ids],labels=[0,1,2],average='macro',zero_division=0)[2];deltas.append(pb-pa)
    return {'comparison':'DistilBERT seed42 minus pair TF-IDF SVM','delta_macro_f1':metrics(y,b)['macro_f1']-metrics(y,a)['macro_f1'],
        'ci95_low':float(np.quantile(deltas,.025)),'ci95_high':float(np.quantile(deltas,.975)),
        'unit':'normalized premise group','repetitions':len(deltas),'scope':'Conditional on two fixed models; excludes training uncertainty and does not correct multiple comparisons.'}

def evaluate():
    frozen=core.read_json(OUT/'protocol.freeze.json');check(frozen)
    if all((OUT/f).exists() for f in ['evaluation.json','metrics.csv','predictions.csv','bootstrap.json']):check(core.read_json(OUT/'evaluation.json'));return
    device='cuda' if torch.cuda.is_available() else 'cpu';rows=[];predictions=[]
    tests={s:load_split(s) for s in ['matched','mismatched']}
    for artifact in frozen['artifacts']:
        name,seed=artifact['model'],artifact['seed'];dest=ROOT/artifact['path']
        if hashlib.sha256((dest/artifact['file']).read_bytes()).hexdigest()!=artifact['sha256']:raise RuntimeError('Changed checkpoint')
        model,tok=load_model(name,seed,device)
        for split,f in tests.items():
            pred,score,kind=predict(model,tok,pairs(f),name,device);result=metrics(f.label,pred)
            rows.append({'source':'mnli','target':'mnli_'+split,'model':name,'seed':seed,'test_n':len(f),**result})
            predictions.append(pd.DataFrame({'sample_id':f.sample_id,'premise_hash':f.premise_hash,'split':split,'model':name,'seed':seed,
                'label':f.label,'prediction':pred,'score':[json.dumps(np.asarray(v).tolist()) for v in score],'score_type':kind}))
            core.log(f'MNLI EVAL {split}/{name} seed={seed}: Accuracy={result["accuracy"]:.4f} Macro F1={result["macro_f1"]:.4f}')
        del model,tok;gc.collect();torch.cuda.empty_cache() if device=='cuda' else None
    prediction=pd.concat(predictions,ignore_index=True);prediction.to_csv(OUT/'predictions.csv',index=False)
    core.write_json(OUT/'evaluation.json',{**signature(),'created_utc':core.utcnow(),'rows':rows})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['classification_report','confusion_matrix']} for r in rows]).to_csv(OUT/'metrics.csv',index=False)
    bootstrap={}
    for split,f in tests.items():
        selected=prediction[prediction.split==split]
        def get(name,seed=None):
            p=selected[selected.model==name]
            if seed is not None:p=p[p.seed==seed]
            return p.set_index('sample_id').loc[f.sample_id].prediction.to_numpy()
        bootstrap[split]=cluster_bootstrap(f,get('linear_svm'),get('distilbert',42))
    core.write_json(OUT/'bootstrap.json',bootstrap)

def benchmark():
    path=OUT/'benchmark.json'
    if path.exists():check(core.read_json(path));return
    check(core.read_json(OUT/'protocol.freeze.json'));torch.set_num_threads(4)
    sample=load_split('matched').sample(200,random_state=2026);records=pairs(sample);rows=[]
    for name in MODELS:
        model,tok=load_model(name,42,'cpu');elapsed=[]
        with threadpool_limits(limits=4):
            for record in records[:20]:predict(model,tok,[record],name)
            for _ in range(5):
                for record in records:
                    start=time.perf_counter();predict(model,tok,[record],name);elapsed.append((time.perf_counter()-start)*1000)
        rows.append({'source':'mnli','model':name,'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
            'threads':4,'batch_size':1,'sample_ids':sample.sample_id.tolist(),'scope':'200 matched sentence pairs, five repeats, preprocessing plus forward, loading excluded'})
        core.log(f'MNLI CPU {name}: median={np.median(elapsed):.2f} ms');del model,tok;gc.collect()
    core.write_json(path,{**signature(),'created_utc':core.utcnow(),'rows':rows})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','freeze','evaluate','benchmark','all']);args=parser.parse_args();OUT.mkdir(exist_ok=True)
    stages={'prepare':prepare,'train':train,'freeze':freeze,'evaluate':evaluate,'benchmark':benchmark}
    if args.stage=='all':
        for stage in stages.values():stage()
    else:stages[args.stage]()
if __name__=='__main__':main()

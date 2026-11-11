"""Controlled, fixed-N AG News imbalance stress test. All stages are resumable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, contextlib, gc, hashlib, json, time
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import extension as ext
import pipeline as core
from models_extra import words, mean_vectors

ROOT=core.ROOT
CFG=core.read_json(ROOT/'config-imbalance.json')
DATA=ROOT/'data-imbalance'; OUT=ROOT/'results-imbalance'
LABELS=ext.LABELS['ag_news']; MODELS=ext.MODELS
BASE_LOAD=ext.load_split

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def directory(condition,name,seed=None,weighted=False):
    if condition=='balanced':return ext.directory('ag_news',name,seed)
    return ROOT/'models-imbalance'/condition/((name if seed is None else f'{name}_seed{seed}')+('_weighted' if weighted else ''))

def prepare():
    if (DATA/'audit.json').exists():
        audit=core.read_json(DATA/'audit.json')
        if audit['config_sha256']!=core.canonical_hash(CFG):raise RuntimeError('Imbalance config changed')
        for c in CFG['conditions']:load_train(c)
        return
    info=core.read_json(ROOT/'assets.lock.json')['datasets']['ag_news'];frames=[]
    for split in ['train','test']:
        f=pd.concat([pd.read_parquet(ROOT/'assets'/'ag_news'/p) for p in info['files'] if p.split('/')[-1].startswith(split)],ignore_index=True)
        f['sample_id']=[f'ag_news/{split}/{i}' for i in range(len(f))];f['official_split']=split
        f['text']=f.text.map(core.clean_text);f['text_hash']=f.text.map(core.text_hash);frames.append(f)
    merged=pd.concat(frames,ignore_index=True);conflicts=merged.groupby('text_hash').label.nunique()
    pool=merged[(merged.text.str.len()>0)&~merged.text_hash.isin(conflicts[conflicts>1].index)].copy()
    pool['priority']=pool.official_split.map({'test':0,'train':1})
    pool=pool.sort_values(['priority','sample_id'],kind='stable').drop_duplicates('text_hash')
    val=BASE_LOAD('ag_news','validation');test=BASE_LOAD('ag_news','test');base=BASE_LOAD('ag_news','train')
    pool=pool[(pool.official_split=='train')&~pool.text_hash.isin(val.text_hash)]
    assert set(base.sample_id)<=set(pool.sample_id)
    rng=np.random.default_rng(CFG['sample_seed']);orders={}
    for label in range(4):
        start=base[base.label==label].sample(frac=1,random_state=CFG['sample_seed']+label)
        extra=pool[(pool.label==label)&~pool.sample_id.isin(base.sample_id)]
        extra=extra.iloc[rng.permutation(len(extra))]
        orders[label]=pd.concat([start,extra],ignore_index=True)
    DATA.mkdir(exist_ok=True);audit={'config_sha256':core.canonical_hash(CFG),'created_utc':core.utcnow(),
        'base_split_sha256':core.read_json(ROOT/'data-v2'/'audit.json')['split_sha256'],
        'sampling':'Fixed total 6000; World majority; nested class orders; balanced reference reused exactly; no duplicate/oversampled text',
        'split_sha256':{},'counts':{},'eligible_train_n':len(pool)}
    manifests=[]
    for condition,counts in CFG['conditions'].items():
        f=base.copy() if condition=='balanced' else pd.concat([orders[c].iloc[:n] for c,n in enumerate(counts)],ignore_index=True).sample(frac=1,random_state=2026)
        assert len(f)==6000 and not f.text_hash.duplicated().any()
        assert not set(f.text_hash)&(set(val.text_hash)|set(test.text_hash))
        assert [int((f.label==c).sum()) for c in range(4)]==counts
        path=DATA/f'{condition}.jsonl';f.to_json(path,orient='records',lines=True,force_ascii=False)
        audit['split_sha256'][condition]=sha(path);audit['counts'][condition]=counts
        manifests.append(f.drop(columns='text').assign(condition=condition))
    pd.concat(manifests,ignore_index=True).to_csv(DATA/'manifest.csv',index=False)
    core.write_json(DATA/'audit.json',audit)
    core.write_json(OUT/'design.freeze.json',{'created_utc':core.utcnow(),'config':CFG,'audit':audit,
        'prior_test_exposure':'Balanced AG News test already inspected; all new ratios and mitigation fixed before new runs.',
        'base_models':[{ 'model':n,'seed':s,'metadata_sha256':sha(directory('balanced',n,s)/'metadata.json')} for n in MODELS for s in (CFG['training_seeds'] if n in ['bilstm','distilbert'] else [None])]})

def load_train(condition):
    path=DATA/f'{condition}.jsonl'
    if sha(path)!=core.read_json(DATA/'audit.json')['split_sha256'][condition]:raise RuntimeError('Imbalance data changed')
    return pd.read_json(path,lines=True)

def signature():
    return {'imbalance_config_sha256':core.canonical_hash(CFG),'data_sha256':core.read_json(DATA/'audit.json')['split_sha256'],
            'balanced_data_sha256':core.read_json(ROOT/'data-v2'/'audit.json')['split_sha256'],
            'base_model_revision':core.read_json(ROOT/'assets.lock.json')['model']['revision']}
def check(meta):
    for k,v in signature().items():
        if meta.get(k)!=v:raise RuntimeError('Imbalance provenance changed: '+k)

def word_assets(condition):
    from gensim.models import Word2Vec
    dest=ROOT/'models-imbalance'/condition/'word2vec';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metadata.json').exists():check(core.read_json(dest/'metadata.json'));return joblib.load(dest/'features.joblib')
    sentences=[words(t) for t in load_train(condition).text];tick=time.perf_counter()
    emb=Word2Vec(sentences,vector_size=100,window=5,min_count=2,sg=1,workers=1,epochs=10,seed=2026,hashfxn=lambda s:int(hashlib.md5(s.encode()).hexdigest(),16))
    counts=Counter(w for s in sentences for w in s);tokens=sorted(emb.wv.key_to_index,key=lambda w:(-counts[w],w))[:29998]
    vocab={w:i+2 for i,w in enumerate(tokens)};vectors={w:emb.wv[w].copy() for w in tokens}
    matrix=np.random.default_rng(2026).normal(0,.02,(len(tokens)+2,100)).astype(np.float32);matrix[0]=0
    for w,i in vocab.items():matrix[i]=vectors[w]
    assets={'vectors':vectors,'vocabulary':vocab,'matrix':matrix};joblib.dump(assets,dest/'features.joblib')
    core.write_json(dest/'metadata.json',{**signature(),'training_only':True,'fit_seconds':time.perf_counter()-tick,'vocabulary_size':len(vocab)})
    return assets

@contextlib.contextmanager
def adapted(condition):
    original=(ext.load_split,ext.word_assets,ext.signature,ext.check)
    ext.LABELS[condition]=LABELS
    ext.load_split=lambda source,split:load_train(condition) if split=='train' else BASE_LOAD('ag_news',split)
    ext.word_assets=lambda source:word_assets(condition);ext.signature=signature;ext.check=check
    try:yield
    finally:
        ext.load_split,ext.word_assets,ext.signature,ext.check=original
        if condition!='ag_news':ext.LABELS.pop(condition,None)

def train_classical(condition,name,weighted=False):
    dest=directory(condition,name,weighted=weighted)
    if (dest/'metadata.json').exists():check(core.read_json(dest/'metadata.json'));return
    selected=core.read_json(directory('balanced',name)/'metadata.json')['selected']
    train=load_train(condition);val=BASE_LOAD('ag_news','validation');start=time.perf_counter()
    clf=LogisticRegression(C=selected['C'],max_iter=3000,solver='lbfgs') if name=='logistic_regression' else LinearSVC(C=selected['C'],max_iter=10000,random_state=42,class_weight='balanced' if weighted else None)
    assets=word_assets(condition) if name=='word2vec_svm' else None
    model=make_pipeline(StandardScaler(),clf) if assets else make_pipeline(TfidfVectorizer(ngram_range=tuple(selected['ngram']),max_features=30000,min_df=2,sublinear_tf=True),clf)
    x=mean_vectors(train.text,assets['vectors']) if assets else train.text
    with threadpool_limits(limits=4):model.fit(x,train.label)
    pred=model.predict(mean_vectors(val.text,assets['vectors']) if assets else val.text)
    dest.mkdir(parents=True,exist_ok=True);joblib.dump({'classifier':model,'vectors':assets['vectors']} if assets else model,dest/'model.joblib')
    core.write_json(dest/'metadata.json',{**signature(),'condition':condition,'model':name,'weighted':weighted,'selected':selected,
        'validation_macro_f1':ext.metric_set(val.label,pred,'ag_news')['macro_f1'],'fit_seconds':time.perf_counter()-start})
    core.log(f'Imbalance {condition}/{name} weighted={weighted} fitted')

def train():
    for condition in list(CFG['conditions'])[1:]:
        for name in MODELS[:3]:train_classical(condition,name)
        for name in MODELS[3:]:
            lr=core.read_json(directory('balanced',name,42)/'metadata.json')['learning_rate']
            for seed in CFG['training_seeds']:
                with adapted(condition):ext.train_deep(condition,name,lr,seed,directory(condition,name,seed))
    train_classical('severe','linear_svm',True)
    weights=(6000/(4*np.asarray(CFG['conditions']['severe']))).tolist()
    lr=core.read_json(directory('balanced','distilbert',42)/'metadata.json')['learning_rate']
    for seed in CFG['training_seeds']:
        with adapted('severe'):ext.train_deep('severe','distilbert',lr,seed,directory('severe','distilbert',seed,True),class_weights=weights)

def runs():
    for condition in CFG['conditions']:
        for name in MODELS:
            for seed in (CFG['training_seeds'] if name in MODELS[3:] else [None]):yield condition,name,seed,False
    yield 'severe','linear_svm',None,True
    for seed in CFG['training_seeds']:yield 'severe','distilbert',seed,True

def load_model(condition,name,seed,weighted=False,device='cpu'):
    if condition=='balanced':return ext.load_model('ag_news',name,seed or 42,device)
    original=ext.directory
    ext.directory=lambda source,n,s=None:directory(condition,n,s,weighted)
    try:return ext.load_model('ag_news',name,seed or 42,device)
    finally:ext.directory=original

def measures(y,pred,condition):
    m=ext.metric_set(y,pred,'ag_news');report=m['classification_report']
    m['balanced_accuracy']=m['macro_recall']
    m['minority_recall']=float(np.mean([report[LABELS[c]]['recall'] for c in [1,2,3]]))
    m['minority_f1']=float(np.mean([report[LABELS[c]]['f1-score'] for c in [1,2,3]]))
    m['worst_class_recall']=float(min(report[x]['recall'] for x in LABELS))
    y=np.asarray(y);prior=np.asarray(CFG['conditions'][condition])/6000
    w=prior[y]/(np.bincount(y,minlength=4)[y]/len(y))
    m['simulated_prior_weighted_accuracy']=float(np.average(np.asarray(pred)==y,weights=w))
    return m

def evaluate():
    artifacts=[]
    for c,n,s,w in runs():
        d=directory(c,n,s,w);meta=core.read_json(d/'metadata.json')
        if c!='balanced':check(meta)
        p=d/('model.pt' if n=='bilstm' else 'model.safetensors' if n=='distilbert' else 'model.joblib')
        artifacts.append({'condition':c,'model':n,'seed':s,'weighted':w,'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    freeze=OUT/'protocol.freeze.json'
    if freeze.exists():
        if core.read_json(freeze)['artifacts']!=artifacts:raise RuntimeError('Frozen checkpoint changed')
    else:core.write_json(freeze,{**signature(),'artifacts':artifacts,'created_utc':core.utcnow()})
    complete=OUT/'evaluation.json'
    if complete.exists():check(core.read_json(complete));return
    test=BASE_LOAD('ag_news','test');legacy=pd.read_csv(ROOT/'results-v2'/'predictions.csv')
    rows=[];records=[];device=str(core.torch_setup(42))
    for c,n,s,w in runs():
        if c=='balanced':
            f=legacy[(legacy.source=='ag_news')&(legacy.target=='ag_news')&(legacy.model==n)]
            f=f[f.seed==s] if s is not None else f[f.seed.isna()]
            f=test[['sample_id','label']].merge(f[['sample_id','prediction','score','score_type']],on='sample_id',validate='one_to_one')
            pred=f.prediction.to_numpy();scores=f.score.tolist();kind=f.score_type.iloc[0]
        else:
            model,tok=load_model(c,n,s,w,device)
            pred,score,kind=ext.predict(model,tok,test.text.tolist(),n,device)
            scores=[json.dumps(x.tolist() if hasattr(x,'tolist') else x) for x in score]
            del model,tok;gc.collect();torch.cuda.empty_cache()
        row={'condition':c,'model':n,'seed':s,'weighted':w,'test_n':len(test),**measures(test.label,pred,c)};rows.append(row)
        for sid,label,p,score in zip(test.sample_id,test.label,pred,scores):records.append({'condition':c,'model':n,'seed':s,'weighted':w,'sample_id':sid,'label':int(label),'prediction':int(p),'score':score,'score_type':kind})
        core.log(f'Imbalance {c}/{n}/{s} weighted={w}: test macro F1={row["macro_f1"]:.4f}, minority recall={row["minority_recall"]:.4f}')
    pd.DataFrame(records).to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['classification_report','confusion_matrix']} for r in rows]).to_csv(OUT/'metrics.csv',index=False)
    core.write_json(complete,{**signature(),'rows':rows,'created_utc':core.utcnow()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate','all']);args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare()
    if args.stage in ['train','all']:train()
    if args.stage in ['evaluate','all']:evaluate()
if __name__=='__main__':main()

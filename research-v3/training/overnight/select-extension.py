"""Generate a reviewable subset of exact original requests; no fitting or Q access."""
import json
import hashlib
from collections import Counter
from pathlib import Path

r=Path('research-v3'); folder=r/'training/overnight'
base=json.loads((folder/'run-plan.json').read_text(encoding='utf-8'))
manifest=json.loads((r/'training/fixed-main-run-manifest.json').read_text(encoding='utf-8'))
design=json.loads((r/'matrix/design_manifest.json').read_text(encoding='utf-8'))
conditions={c['condition_id']:c for c in design['conditions']}
original={x['entry']['run_id'] for x in base['runs']}
chosen={}; groups=Counter(); reused=Counter()
datasets=('yelp','go_emotions','banking77','yahoo_answers_topics')


def choose(ds,model,experiment='E1',seed=2026,ts=42,**criteria):
    found=[]
    for entry in manifest['runs']:
        condition=conditions[entry['condition_id']]
        if entry['dataset_id']!=ds or entry['model_id']!=model or entry['data_seed']!=seed:continue
        if entry['training_seed'] not in (None,ts):continue
        if not any(alias['experiment_id']==experiment for alias in entry['semantic_aliases']):continue
        if any(condition['metadata'].get(k)!=v for k,v in criteria.items()):continue
        found.append(entry)
    assert len(found)==1,(ds,model,experiment,seed,ts,criteria,len(found))
    return found[0]


def add(entry,group,priority=0):
    key=entry['run_id']
    if key in original:return
    if key in chosen:chosen[key]['reasons'].append(group);return
    condition=conditions[entry['condition_id']]
    path=r/'runs'/key/'status.json'
    status=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    done=bool(status.get('status')=='complete' and status.get('reload_verified'))
    chosen[key]={'priority':priority,'dataset_id':entry['dataset_id'],'model_id':entry['model_id'],
        'training_rows':condition['metadata']['rows'],'data_seed':entry['data_seed'],'training_seed':entry['training_seed'],
        'completed_checkpoint_reuse':done,'status_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if done else None,
        'entry':entry,'reasons':[group]}
    groups[group]+=1;reused[group]+=int(done)


for ds in datasets:
    for seed in (2027,2028):
        for n in (1000,6000):
            for model in ('tfidf_svm','distilbert'):
                add(choose(ds,model,seed=seed,rows=n),'sampling_reproducibility')
    for ts in (43,44):
        for model in ('bilstm','distilbert'):
            add(choose(ds,model,ts=ts,rows=6000),'training_seed_reproducibility')
for ds in ('yelp','yahoo_answers_topics'):
    for model in ('tfidf_svm','bilstm','distilbert'):
        add(choose(ds,model,rows=100000),'larger_training_scale_100k',1)
for alpha in (-.5,.5):
    for model in ('tfidf_svm','bilstm','distilbert'):
        add(choose('go_emotions',model,'E4',alpha=alpha),'multilabel_distribution')
for ds,ns in [('banking77',(6000,)),('go_emotions',(1000,6000))]:
    for n in ns:
        for arm in ('pretrained_frozen','pretrained_unfrozen','random_frozen','random_unfrozen'):
            found=[e for e in manifest['runs'] if e['dataset_id']==ds and e['model_id']=='m2_distilbert'
                and e['data_seed']==2026 and e['training_seed']==42
                and conditions[e['condition_id']]['metadata']['rows']==n
                and e['run']['hparams'].get('arm')==arm and e['run']['hparams'].get('encoder_lr')==2e-5]
            assert len(found)==1
            add(found[0],'pretraining_and_adaptation_2x2')
first=next(c for c in design['conditions'] if c['dataset_id']=='banking77'
    and c['experiment_id']=='M3' and c['data_seed']==2026 and c['metadata'].get('rho')==10)
head=first['metadata']['head_order']
for support in ('full','tail_reduced'):
    for risk in ('uniform','skew'):
        for model in ('fixed_tfidf_svm','frozen_generic_distilbert_head'):
            found=[e for e in manifest['runs'] if e['dataset_id']=='banking77' and e['model_id']==model
                and e['data_seed']==2026 and e['training_seed'] in (None,42)
                and conditions[e['condition_id']]['experiment_id']=='M3'
                and all(conditions[e['condition_id']]['metadata'].get(k)==v for k,v in
                    {'rho':10,'head_order':head,'support':support,'risk':risk}.items())]
            assert len(found)==1,(support,risk,model,len(found))
            add(found[0],'support_vs_class_contribution_2x2')
for ds in datasets:
    add(choose(ds,'modernbert','E5',rows=6000),'modern_encoder_reference',1)

exclude={'runs','subset_hash','counts_by_priority','total_selected','new_fits_maximum','reused_complete','deferred_original_requests','sampling_rule'}
out={k:v for k,v in base.items() if k not in exclude}
out.update(scope='overnight_high_priority_extension',status='selected_not_submitted',
    user_instruction='2026-11-07: choose another high-priority batch from original2859 matching documented requirements; retain overnight deadline',
    runs=list(chosen.values()),total_selected=len(chosen),new_fits_maximum=sum(not x['completed_checkpoint_reuse'] for x in chosen.values()),
    reused_complete=sum(x['completed_checkpoint_reuse'] for x in chosen.values()),extension_of_subset_hash=base['subset_hash'],
    sampling_rule='Exact original requests and aliases; fixed matched arms/rate, first original head order; extension fills documented gaps after exploratory V results, no Q observed',
    counts_by_group={g:{'selected':n,'reused':reused[g],'new':n-reused[g]} for g,n in groups.items()})
out['subset_hash']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert not original.intersection(chosen)
assert all(row['entry'] in manifest['runs'] for row in out['runs'])
(folder/'extension-run-plan.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('total_selected','new_fits_maximum','reused_complete','counts_by_group','subset_hash')},indent=2))

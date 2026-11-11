"""Reconstruct saved Q metrics and assemble the study's empirical argument."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import collections
import json
import statistics

import numpy as np
import pyarrow.parquet as pq

from research_v3.config import canonical_hash
from research_v3.evaluate import _read_predictions, _metrics, validate_prediction_rows
from research_v3.storage import hash_file

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT/'research-v3'
REPORT=ROOT/'reports/overnight'
sources={}


def read(path):
    sources[str(path.relative_to(PROJECT))]=hash_file(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def close_tree(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:close_tree(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close_tree(x,y)
    elif isinstance(a,float):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def status(rid):
    for directory in [ROOT/'runs',REPORT/'analysis-handoff/cluster-evidence/runs',
        REPORT/'supplement-1-4/cluster-snapshot/runs',REPORT/'convergence-v1/cluster-snapshot/runs']:
        path=directory/rid/'status.json'
        if path.exists():
            value=read(path)
            if value.get('status')=='complete':return value
    raise ValueError('No complete status: '+rid)


locations=[('优化与历史参考',REPORT/'convergence-v1/cluster-snapshot',Path('reports/overnight/convergence-v1'),
    Path('training/convergence-v1/scoped-evaluation.freeze.json'))]
for stage in ['S1','S3']:
    folder=Path('reports/overnight/supplement-q-v1')/stage
    locations.append((stage,REPORT/'supplement-q-v1/cluster-snapshot',folder,folder/'evaluation.freeze.json'))
qrows=[];gold={}
for phase,snapshot,relative,freeze_path in locations:
    result=read(snapshot/relative/'test-results.json');lock=read(snapshot/freeze_path)
    assert result['status']=='complete'
    assert lock['lock_hash']==canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'})
    entries={entry['run_id']:entry for entry in lock['models']}
    for model in result['models']:
        rid=model['run_id'];entry=entries[rid];meta=lock['metadata'][model['dataset_id']]
        raw=ROOT/meta['partitions']['test']['file'];assert hash_file(raw)==meta['partitions']['test']['sha256']
        if model['dataset_id'] not in gold:
            records=pq.read_table(raw,columns=['sample_id','labels','group_id']).to_pylist()
            gold[model['dataset_id']]={r['sample_id']:(tuple(r['labels']),r['group_id']) for r in records}
        pred=snapshot/model['path'];assert hash_file(pred)==model['sha256']
        sources[str(pred.relative_to(PROJECT))]=model['sha256']
        assert canonical_hash(model['metrics'])==model['metrics_hash']
        assert canonical_hash(model['timing'])==model['timing_hash']
        saved=_read_predictions(pred);validate_prediction_rows(saved,len(meta['label_names']),meta['task'])
        assert len(saved)==meta['partitions']['test']['rows']==len(gold[model['dataset_id']])
        assert {r.sample_id for r in saved}==set(gold[model['dataset_id']])
        for row in saved:
            assert (row.labels,row.group_id)==gold[model['dataset_id']][row.sample_id]
            expected=tuple(int(i) for i in np.flatnonzero(np.asarray(row.scores)>=(.5 if row.score_kind=='probability' else 0.))) if meta['task']!='single_label' else (int(np.argmax(row.scores)),)
            assert row.prediction==expected
        metrics=_metrics(saved,meta);close_tree(metrics,model['metrics'])
        training=status(rid);assert training['run']==entry['run']
        hp=entry['run']['hparams'];metric='macro_ap' if model['dataset_id']=='go_emotions' else 'macro_f1'
        diagnostics=model.get('diagnostics',{})
        if phase=='S3':
            per={r['label_id']:r for r in metrics['per_label']}
            for label,group in [('tail',entry['tail_labels']),('head',sorted(set(per)-set(entry['tail_labels'])))]:
                for key in ['f1','recall']:assert abs(diagnostics[label+'_macro_'+key]-np.mean([per[i][key] for i in group]))<1e-12
        qrows.append({'run_id':rid,'stage':phase,'dataset':model['dataset_id'],'model':model['model_id'],
            'role':model['role'],'seed':entry['run']['training_seed'],'train_rows':training['training_unique_rows'],
            'lr':hp.get('lr'),'C':hp.get('C'),'budget':hp.get('max_epochs',hp.get('epochs')),
            'metric':metric,'score':metrics[metric],'macro_f1':metrics['macro_f1'],'macro_ap':metrics.get('macro_ap'),
            'test_rows':len(saved),'support':diagnostics.get('factors',{}).get('support'),
            'risk':diagnostics.get('factors',{}).get('risk'),'tail_f1':diagnostics.get('tail_macro_f1'),
            'tail_recall':diagnostics.get('tail_macro_recall'),'head_f1':diagnostics.get('head_macro_f1'),
            'predict_seconds':model['timing']['predict_seconds'],'load_seconds':model['timing']['model_load_seconds'],
            'milliseconds_per_row':model['timing'].get('milliseconds_per_row'),
            'source':str((snapshot/relative/'test-results.json').relative_to(PROJECT))})
assert len(qrows)==len({r['run_id'] for r in qrows})==42

primary=[]
for ds in ['banking77','go_emotions']:
    for model in ['tfidf_svm','bilstm','distilbert']:
        rows=[r for r in qrows if r['dataset']==ds and r['model']==model and r['role']=='primary']
        vals=[r['score'] for r in rows];f1s=[r['macro_f1'] for r in rows]
        primary.append({'dataset':ds,'model':model,'n':len(rows),'metric':rows[0]['metric'],
            'mean':statistics.mean(vals),'sd':statistics.stdev(vals) if len(vals)>1 else None,
            'min':min(vals),'max':max(vals),'mean_f1':statistics.mean(f1s),
            'sd_f1':statistics.stdev(f1s) if len(f1s)>1 else None,'run_ids':[r['run_id'] for r in rows]})
contrasts=[]
def contrast(section,context,metric,before,after,meaning,source_ids):
    contrasts.append({'section':section,'context':context,'metric':metric,'before':before,'after':after,
        'difference_pp':100*(after-before),'meaning':meaning,'run_ids':source_ids})
for ds in ['banking77','go_emotions']:
    base=next(r for r in primary if r['dataset']==ds and r['model']=='tfidf_svm')
    for model in ['distilbert','bilstm']:
        row=next(r for r in primary if r['dataset']==ds and r['model']==model)
        contrast('Q模型比较',ds+' / '+model+' − SVM',row['metric'],base['mean'],row['mean'],
            '神经模型3种子均值与确定性SVM比较',base['run_ids']+row['run_ids'])
for (ds,model,lr),rows in __import__('itertools').groupby(
        sorted([r for r in qrows if r['stage']=='S1'],key=lambda r:(r['dataset'],r['model'],r['lr'],r['budget'])),
        key=lambda r:(r['dataset'],r['model'],r['lr'])):
    rows=list(rows);a,b=rows
    contrast('S1预算',f'{ds} / {model} / lr={lr:g} / {a["budget"]}→{b["budget"]}轮',a['metric'],a['score'],b['score'],
        '总预算改变同时改变线性衰减计划，单seed42',[a['run_id'],b['run_id']])
    if ds=='go_emotions':contrast('S1阈值决策',f'{model} / lr={lr:g} / {a["budget"]}→{b["budget"]}轮','macro_f1',a['macro_f1'],b['macro_f1'],
        '相同固定0阈值下的F1变化',[a['run_id'],b['run_id']])
for model in ['fixed_tfidf_svm','frozen_generic_distilbert_head']:
    rows={(r['support'],r['risk']):r for r in qrows if r['stage']=='S3' and r['model']==model}
    for risk in ['uniform','skew']:
        a,b=rows['full',risk],rows['tail_reduced',risk]
        for key in ['macro_f1','tail_f1','tail_recall']:
            contrast('S3标注支持',model+' / '+risk+' / 完整→尾部删减',key,a[key],b[key],
                '固定表示、类别贡献、总权重/更新预算',[a['run_id'],b['run_id']])
    for support in ['full','tail_reduced']:
        a,b=rows[support,'uniform'],rows[support,'skew']
        for key in ['macro_f1','tail_f1','tail_recall']:
            contrast('S3类别贡献',model+' / '+support+' / 均匀→偏斜',key,a[key],b[key],
                '固定表示和独立标注支持',[a['run_id'],b['run_id']])

inventory=read(REPORT/'convergence-v1/experiment-inventory.json')
main=read(ROOT/'training/fixed-main-run-manifest.json');entries={e['run_id']:e for e in main['runs']}
m2=[]
for row in inventory['rows']:
    if row['model']!='m2_distilbert' or row['dataset']!='banking77' or row['train_rows']!=1000:continue
    record=status(row['run_id']);hp=record['run']['hparams']
    if abs(hp['encoder_lr']-2e-5)>1e-12:continue
    entry=entries[row['run_id']]
    m2.append({'run_id':row['run_id'],'arm':hp['arm'],'data_seed':entry['data_seed'],
        'training_seed':record['run']['training_seed'],'macro_f1':record['validation']['macro_f1']})
assert len(m2)==36
m2_summary=[]
for arm in ['random_frozen','pretrained_frozen','random_unfrozen','pretrained_unfrozen']:
    vals=[r['macro_f1'] for r in m2 if r['arm']==arm];assert len(vals)==9
    m2_summary.append({'arm':arm,'n':len(vals),'mean':statistics.mean(vals),'sd':statistics.stdev(vals)})
for a,b in [('random_frozen','pretrained_frozen'),('random_unfrozen','pretrained_unfrozen'),
            ('pretrained_frozen','pretrained_unfrozen'),('random_frozen','random_unfrozen')]:
    av={ (r['data_seed'],r['training_seed']):r for r in m2 if r['arm']==a}
    bv={ (r['data_seed'],r['training_seed']):r for r in m2 if r['arm']==b};assert set(av)==set(bv)
    diff=[bv[k]['macro_f1']-av[k]['macro_f1'] for k in av]
    contrast('M2预训练与适配',a+' → '+b,'V macro_f1',statistics.mean(r['macro_f1'] for r in av.values()),
        statistics.mean(r['macro_f1'] for r in bv.values()),f'3个数据抽样×3训练种子；{sum(d>0 for d in diff)}/9成对配置提高',
        [r['run_id'] for r in av.values()]+[r['run_id'] for r in bv.values()])

old=read(REPORT/'research-status.json');scale=[];imbalance=[]
for pair in old['core_scale_pairs']:
    for rid in pair['run_ids']:
        r=status(rid);metric='macro_ap' if pair['dataset']=='go_emotions' else 'macro_f1'
        scale.append({'dataset':pair['dataset'],'model':pair['model'],'n':r['training_unique_rows'],
            'metric':metric,'score':r['validation'][metric],'run_id':rid})
for pair in old['core_imbalance_pairs']:
    a,b=[status(rid) for rid in pair['run_ids']]
    av,bv=a['validation']['macro_f1'],b['validation']['macro_f1']
    assert abs(av-pair['balanced'])<1e-12 and abs(bv-pair['skewed'])<1e-12
    imbalance.append({'dataset':pair['dataset'],'model':pair['model'],'n':pair['n'],
        'balanced':av,'skewed':bv,'drop_pp':100*(av-bv),'run_ids':pair['run_ids']})
s2=read(REPORT/'supplement-1-4/cluster-snapshot/S2-summary.json')
trajectories=[]
for t in s2['matched_trajectories']:
    n=sum(t['counts'].values());assert n==len(t['rows'])
    trajectories.append({'dataset':t['dataset_id'],'model':t['model_id'],'n':n,**t['counts']})
edits=read(REPORT/'supplement-1-4/S4-ai-reviewed/results.json')
frozen=read(REPORT/'supplement-1-4/S4-ai-reviewed/inputs-frozen.json');s4=[]
for model in edits['models']:
    pairs=model['pairs'];summary=model['summary'];n=len(pairs);assert n==37
    for key in ['original','cue','control']:
        assert sum(p[key]['correct'] for p in pairs)==summary[key+'_correct']
    s4.append({'model':model['model_id'],'n':n,'original':summary['original_accuracy'],
        'cue':summary['cue_accuracy'],'control':summary['control_accuracy'],
        'drop_pp':100*(summary['control_accuracy']-summary['cue_accuracy']),
        'cue_flips':summary['cue_prediction_flips'],'control_flips':summary['control_prediction_flips']})
cases=[]
for case_id in ['M1-013','M1-019']:
    pair=next(r['pair'] for r in frozen['pairs'] if r['case_id']==case_id)
    cases.append({'case_id':case_id,'text_edits':pair,'predictions':{
        m['model_id']:next({k:v for k,v in p.items() if k not in ['original','cue','control']}|
            {k:{kk:vv for kk,vv in p[k].items() if kk!='scores'} for k in ['original','cue','control']}
            for p in m['pairs'] if p['case_id']==case_id) for m in edits['models']}})

output={'generated_at_sgt':datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat(),
    'verification':{'Q_models':42,'all_prediction_hashes_verified':True,'all_Q_gold_group_ID_coverage_verified':True,
        'all_saved_Q_metrics_reconstructed':True,'metrics_absolute_tolerance':1e-12,'new_training_or_inference':False},
    'q_records':qrows,'primary':primary,'contrasts':contrasts,'m2':m2,'m2_summary':m2_summary,
    'scale':scale,'imbalance':imbalance,'trajectories':trajectories,'s4':s4,'cases':cases,'source_sha256':sources}
(HERE/'analysis-data.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'Q_verified':42,'contrasts':len(contrasts),'primary':primary,'S3_effects':[
    r for r in contrasts if r['section'].startswith('S3') and (r['metric']=='macro_f1' or 'frozen_generic' in r['context'])],
    'M2':m2_summary},ensure_ascii=False))

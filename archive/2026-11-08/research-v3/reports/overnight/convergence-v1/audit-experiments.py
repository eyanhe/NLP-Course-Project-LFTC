"""Audit saved experiment metadata only; never fit or load held-out examples."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT / 'research-v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    fixed = read(ROOT/'training/fixed-main-run-manifest.json')
    main_entries = {e['run_id']: e for e in fixed['runs']}
    supp = read(ROOT/'reports/overnight/supplement-1-4/manifest.json')
    supp_entries = {e['entry']['run_id']: e for e in supp['fits']}
    conv = read(ROOT/'training/convergence-v1/manifest.json')
    conv_entries = {e['run_id']: e for e in conv['candidates']}
    for path in (HERE/'cluster-snapshot/training/convergence-v1/selections').glob('*.json'):
        conv_entries.update({e['run_id']: e for e in read(path)['confirmation_entries']})
    engineering = {e['run_id']: e for e in read(ROOT/'training/engineering-smoke.json')['cases']}
    anchors = {e['run_id']: e for e in read(ROOT/'training/anchor-search-progress.json')['candidates']}
    conditions = {e['condition_id']: e for e in read(ROOT/'matrix/design_manifest.json')['conditions']}
    scopes = [('优化与历史参考', HERE/'cluster-snapshot', Path('reports/overnight/convergence-v1/test-results.json'))]
    for stage in ['S1','S3']:
        scopes.append((stage, ROOT/'reports/overnight/supplement-q-v1/cluster-snapshot',
            Path('reports/overnight/supplement-q-v1')/stage/'test-results.json'))
    tested={};prediction_roots={};scope_records=[]
    for name,snapshot,relative in scopes:
        result=read(snapshot/relative)
        assert result['status']=='complete'
        for model in result['models']:
            assert model['run_id'] not in tested
            tested[model['run_id']]=model;prediction_roots[model['run_id']]=snapshot
        scope_records.append({'scope':name,'models':len(result['models']),'completed_at_utc':result['completed_at_utc'],
            'source':str((snapshot/relative).relative_to(PROJECT)),
            'source_sha256':hashlib.sha256((snapshot/relative).read_bytes()).hexdigest()})
    d_rows = read(ROOT/'reports/overnight/supplement-1-4/cluster-snapshot/S2-summary.json')['metrics']
    discovered = {e['run_id'] for e in d_rows}
    roots = [ROOT/'runs', ROOT/'reports/overnight/analysis-handoff/cluster-evidence/runs',
             ROOT/'reports/overnight/supplement-1-4/cluster-snapshot/runs', HERE/'cluster-snapshot/runs']
    records = {}
    for folder in roots:
        for path in sorted(folder.glob('*/status.json')):
            status = read(path)
            if status.get('status') == 'complete' or path.parent.name not in records:
                records[path.parent.name] = (status, path)
    rows = []
    for rid, (status, path) in sorted(records.items()):
        run = status.get('run', {})
        if rid in main_entries:
            category = 'v3首批固定预算'; entry = main_entries[rid]
        elif rid in supp_entries:
            item = supp_entries[rid]; entry = item['entry']
            category = 'S1预算与学习率' if item['supplement_id']=='S1' else 'S3支持与风险贡献'
        elif rid in conv_entries:
            category = '本轮动态早停与C值网格'; entry = conv_entries[rid]
        elif rid in anchors:
            category = '早期选参额外记录'; entry = {}
        elif rid in engineering:
            category = 'E0工程小样本'; entry = engineering[rid]
        else:
            category = '中断或未归类存量'; entry = {}
        dataset = entry.get('dataset_id') or conditions.get(run.get('condition_id'), {}).get('dataset_id')
        if not dataset and rid in anchors:
            dataset = anchors[rid]['node_id'].split(':')[1]
        validation = status.get('validation') or {}
        values = [validation.get(k) for k in ['macro_f1', 'macro_ap', 'accuracy', 'micro_f1']]
        has_v = any(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        done = status.get('status') == 'complete'
        aliases = sorted({a['experiment_id'] for a in entry.get('semantic_aliases', [])})
        test = tested.get(rid, {})
        hp = run.get('hparams', {})
        rows.append({'run_id':rid,'category':category,'dataset':dataset,'model':run.get('model_id'),
            'status':status.get('status'),'actual_training':status.get('actual_model_training_run',False),
            'train_rows':status.get('training_unique_rows'),'epochs':status.get('epochs_completed'),
            'seed':run.get('training_seed'),'lr':hp.get('lr'),'C':hp.get('C'),'arm':hp.get('arm'),
            'experiments':','.join(aliases),'validation':has_v,'strict_no_val':run.get('strict_no_val',False),
            'validation_rows':validation.get('rows'),'V_macro_f1':validation.get('macro_f1'),
            'V_macro_ap':validation.get('macro_ap'),'D_S2_prediction':rid in discovered,
            'formal_Q':rid in tested,'Q_rows':test.get('rows'),'Q_role':test.get('role'),
            'status_source':str(path.relative_to(PROJECT)), 'status_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'Q_prediction_source':str((prediction_roots[rid]/test['path']).relative_to(PROJECT)) if test else None,
            'notes':'工程用途，不纳入研究结论' if category=='E0工程小样本' else
                '早期选参额外记录；采用的36个候选已计入v3首批固定预算' if category=='早期选参额外记录' else
                '按设计禁止访问V，验证未做' if done and not has_v and run.get('strict_no_val') else ''})
    categories=['v3首批固定预算','S1预算与学习率','S3支持与风险贡献','本轮动态早停与C值网格','早期选参额外记录','E0工程小样本']
    totals=[]
    for category in categories:
        selected=[r for r in rows if r['category']==category and r['status']=='complete']
        totals.append({'category':category,'complete':len(selected),'V':sum(r['validation'] for r in selected),
                       'D_S2':sum(r['D_S2_prediction'] for r in selected),'Q':sum(r['formal_Q'] for r in selected)})
    complete=[r for r in rows if r['status']=='complete']
    assert all(r['actual_training'] for r in complete)
    assert len(complete)==428 and sum(t['complete'] for t in totals)==428
    assert sum(t['complete'] for t in totals[:4])==232
    assert sum(t['V'] for t in totals[:4])==226
    assert len(tested)==42 and sum(r['formal_Q'] for r in rows)==42
    assert all((PROJECT/r['Q_prediction_source']).is_file() for r in rows if r['formal_Q'])
    assert set(engineering).issubset(records) and set(anchors).issubset(records)
    assert len(set(anchors)&set(main_entries)&set(records))==36
    current=[]
    for (dataset, model), entries in __import__('itertools').groupby(
            sorted(conv_entries.values(),key=lambda e:(e['dataset_id'],e['model_id'])),
            key=lambda e:(e['dataset_id'],e['model_id'])):
        entries=list(entries)
        selection=read(HERE/'cluster-snapshot/training/convergence-v1/selections'/(dataset+'--'+model+'.json'))
        chosen=selection['chosen']; hp=chosen['run']['hparams']
        current.append({'dataset':dataset,'model':model,'records':len(entries),'V':len(entries),
            'Q':sum(e['run_id'] in tested for e in entries),'best_lr_or_C':hp.get('lr',hp.get('C')),
            'train_rows':chosen['training_rows']})
    payload={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'definitions':{
        'count':'Unique completed run_id; resume/reuse is counted once. Two interrupted records excluded.',
        'V':'Saved finite validation metrics; includes real model prediction on V. No-V arms intentionally did not access V.',
        'D':'S2 exploratory discovery-frame prediction, separately recorded from Q; S4 edits do not count as new training.',
        'Q':'42 saved official-test predictions in three frozen scopes; per-run legacy false flags are not used.'},
        'complete_total':428,'current_research_complete':232,'current_research_with_V':226,'formal_test_complete':42,
        'test_completed_at_utc':max(r['completed_at_utc'] for r in scope_records),'Q_scopes':scope_records,'early_search_completed':len(anchors),
        'early_search_reused_in_main':36,'totals':totals,'latest_round':current,'rows':rows}
    # Keep experiment-inventory.json immutable: it is the dated18-Q input snapshot pinned by analysis-data.json.
    target=HERE/'experiment-inventory-current.json'; target.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ['rows','definitions']},ensure_ascii=False))


if __name__=='__main__':main()

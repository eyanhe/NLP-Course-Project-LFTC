"""CPU-only discovery inference; exact linear evidence, no fitting or human labels."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from research_v3.batch import _run
from research_v3.orchestrator import load_case
from research_v3.partitions import clean_text
from research_v3.storage import read_json,hash_file
from research_v3.train import _verified_bundle

project=Path('/projects/yiyan002humanoid/textclass-overnight-20261107/repo')
assert os.environ.get('SLURM_JOB_ID') and Path.cwd().resolve()==project.resolve()
root=project/'research-v3';folder=root/'reports/overnight';packet=read_json(folder/'failure-frame-plan.json')
assert packet['official_test_predictions_allowed'] is False
prepared=read_json(root/'data/prepared_manifest.json')['datasets']
design={row['condition_id']:row for row in read_json(root/'matrix/design_manifest.json')['conditions']}
destination=folder/'discovery-cases';destination.mkdir(parents=True,exist_ok=True)


def parameters(model,label,width):
    if model.task!='single_label':
        est=model.estimators[label]
        if est is None:return np.zeros(width),float(model._constant(model.constants[label]))
        return est.coef_[0],float(est.intercept_[0])
    est=model.estimator
    if est is None:return np.zeros(width),float(2*model.priors[label]-1)
    classes=list(est.classes_.astype(int))
    if label not in classes:return np.zeros(width),float(-np.finfo(np.float32).max)
    index=classes.index(label)
    if est.coef_.shape[0]==1:
        sign=-1 if index==0 else 1
        return sign*est.coef_[0],float(sign*est.intercept_[0])
    return est.coef_[index],float(est.intercept_[index])


def evidence(bundle,x,score,gold,predicted):
    model=bundle['model'];features=bundle['features'];width=x.shape[1]
    if model.task=='single_label':
        wrong=predicted[0];correct=gold[0]
        a,ab=parameters(model,wrong,width);b,bb=parameters(model,correct,width)
        weights=a-b;bias=ab-bb;observed=float(score[wrong]-score[correct])
        kind='predicted_minus_dataset_label_margin';labels=[wrong,correct]
    else:
        missing=sorted(set(gold)-set(predicted));extra=sorted(set(predicted)-set(gold))
        label=missing[0] if missing else extra[0]
        weights,bias=parameters(model,label,width);observed=float(score[label])
        kind='false_negative_label_margin' if missing else 'false_positive_label_margin';labels=[label]
    values=x.data*weights[x.indices]
    reconstructed=float(values.sum()+bias)
    if not np.isclose(reconstructed,observed,atol=1e-7,rtol=1e-9):raise ValueError('Linear evidence does not reproduce actual score')
    names=features.vectorizer.get_feature_names_out() if not features.degenerate else np.asarray(['constant'])
    terms=[{'feature':str(names[index]),'contribution':float(value)} for index,value in zip(x.indices,values) if value]
    return {'kind':kind,'label_ids':labels,'observed_margin':observed,'intercept':float(bias),
        'sum_feature_contributions':float(values.sum()),'reconstruction_error':abs(observed-reconstructed),
        'active_feature_count':int(x.nnz),'top_positive':sorted((v for v in terms if v['contribution']>0),key=lambda v:-v['contribution'])[:6],
        'top_negative':sorted((v for v in terms if v['contribution']<0),key=lambda v:v['contribution'])[:6],
        'evidence_status':'exact_linear_decomposition; semantic_cause_unreviewed'}


summary={};all_cases=[]
prediction_stream=(destination/'D-predictions.jsonl').open('w',encoding='utf-8')
for ds,frame in packet['frames'].items():
    assert frame['source_partition']=='discovery'
    meta=prepared[ds];source=root/frame['source_file']
    assert hash_file(source)==frame['source_sha256']
    data,inputs=load_case(root,meta,(),selection_ids=(),discovery_ids=frame['sample_ids'])
    actual={row.sample_id:row for row in data.partitions['discovery']}
    assert {row.group_id for row in actual.values()}==set(frame['group_ids'])
    full={row['sample_id']:clean_text(row['text']) for row in pq.read_table(source,columns=['sample_id','text'],filters=[('sample_id','in',frame['sample_ids'])]).to_pylist()}
    ordered=tuple(inputs[sid] for sid in frame['sample_ids'])
    assert all(hashlib.sha256(row.text.encode()).hexdigest()==row.text_hash for row in ordered)
    scored=[]
    for record in packet['checkpoints']:
        if record['dataset_id']!=ds:continue
        run=_run(record['entry']);status,bundle=_verified_bundle(root/'runs'/run.run_id)
        assert status['run']==record['entry']['run'] and bundle['model_id']=='tfidf_svm'
        features=bundle['features'];x=features.transform(ordered);scores=bundle['model'].scores(x)
        condition=design[run.condition_id]
        key=('K'+str(condition['metadata']['shots_per_class'])) if run.strict_no_val else 'N'+str(record['training_rows'])
        rows=[]
        for i,visible in enumerate(ordered):
            gold=list(actual[visible.sample_id].labels)
            predicted=[int(scores[i].argmax())] if data.task=='single_label' else np.flatnonzero(scores[i]>=0).astype(int).tolist()
            correct=set(gold)==set(predicted)
            rows.append({'predicted':predicted,'correct':correct})
            prediction_stream.write(json.dumps({'sample_id':visible.sample_id,'group_id':actual[visible.sample_id].group_id,'dataset_id':ds,'partition':'discovery','run_id':run.run_id,'condition':key,'dataset_labels':gold,'prediction':predicted,'scores':scores[i].tolist(),'score_kind':'svm_margin','scoring_device':'cpu','human_review_status':'pending'},ensure_ascii=False)+'\n')
        scored.append({'key':key,'record':record,'bundle':bundle,'x':x,'scores':scores,'rows':rows})
        print(json.dumps({'dataset':ds,'condition':key,'scored_rows':len(ordered),'status':'D_scored_on_CPU'}),flush=True)
    cases=[]
    for i,visible in enumerate(ordered):
        errors=[item for item in scored if not item['rows'][i]['correct']]
        if not errors:continue
        focal=min(errors,key=lambda item:(0 if item['key']=='N6000' else 1 if item['key'].startswith('N') else 2,-item['record']['training_rows']))
        gold=list(actual[visible.sample_id].labels);predicted=focal['rows'][i]['predicted']
        case={'dataset_id':ds,'sample_id':visible.sample_id,'group_id':actual[visible.sample_id].group_id,
            'source_partition':'discovery','dataset_label_ids':gold,'dataset_label_names':[meta['label_names'][label] for label in gold],
            'visible_text':visible.text,'full_text_normalized':full[visible.sample_id],
            'visible_char_range':[visible.start_char,visible.end_char],'full_tokens':visible.full_tokens,
            'content_tokens':visible.content_tokens,'text_hash':visible.text_hash,
            'focal_run_id':focal['record']['run_id'],'focal_condition':focal['key'],
            'focal_prediction_ids':predicted,'focal_prediction_names':[meta['label_names'][label] for label in predicted],
            'trajectory':[{'condition':item['key'],'run_id':item['record']['run_id'],**item['rows'][i]} for item in scored],
            'linear_evidence':evidence(focal['bundle'],focal['x'][i],focal['scores'][i],gold,predicted),
            'human_review_status':'pending','semantic_failure_reason':'unknown',
            'scope_warning':'Error-enriched selection from an unreserved conditional D frame; not population frequency or confirmed cause'}
        cases.append(case)
        if len(cases)>=packet['max_cases_per_dataset']:break
    summary[ds]={'frame_rows':len(ordered),'reserved_groups_excluded':frame['excluded_reserved_review_groups'],'checkpoints_scored':len(scored),'failure_cases':len(cases),'human_review_status':'pending'}
    all_cases.extend(cases)
prediction_stream.close()
with (destination/'cases.jsonl').open('w',encoding='utf-8') as stream:
    for case in all_cases:stream.write(json.dumps(case,ensure_ascii=False)+'\n')
report={'summary':summary,'total_cases':len(all_cases),'all_linear_margins_verified':True,'official_test_predictions_generated':False,'actual_human_reviews_completed':0,'slurm_job_id':os.environ['SLURM_JOB_ID']}
(destination/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)

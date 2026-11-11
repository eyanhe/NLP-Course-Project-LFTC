"""User-authorized AI-reviewed D supplement, separate from the formal human pilot."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json,hash_file,read_json

REL=Path('reports/overnight/supplement-1-4/S4-ai-reviewed')


def validate_structure(pair):
    from .interventions import english_words,_edit_distance_and_originals
    if pair.get('mechanism')!='M1' or pair.get('source_partition')!='discovery':
        raise ValueError('S4 is an M1 discovery-only supplement')
    source=english_words(pair['original_text']);a=english_words(pair['intervention_text']);b=english_words(pair['control_text'])
    if not 5<=len(source)<=60 or len(source)!=len(a) or len(source)!=len(b):
        raise ValueError('M1 word length/substitution constraint failed')
    cues=set(pair['cue_words']);aa=[x for x,y in zip(source,a) if x!=y];bb=[x for x,y in zip(source,b) if x!=y]
    da,_=_edit_distance_and_originals(source,a);db,_=_edit_distance_and_originals(source,b)
    if not 1<=len(aa)<=3 or len(aa)!=len(bb) or da!=len(aa) or db!=len(bb):
        raise ValueError('Aligned matched substitutions required; word movements forbidden')
    if any(x not in cues for x in aa) or any(x in cues for x in bb):
        raise ValueError('Cue/noncue role mismatch')


def select_ai_subset(reports,draft_pairs,mapping):
    forms={}
    for report in reports:
        if report['reviewer'] in forms:raise ValueError('Duplicate form version')
        rows=report['records'];indexed={r['caseId']:r for r in rows}
        if len(indexed)!=len(rows):raise ValueError('Duplicate case identity in form')
        forms[report['reviewer']]=indexed
    if set(forms)!={'A','B'} or set(forms['A'])!=set(forms['B']):raise ValueError('Form coverage differs')
    draft_by_id={p['sample_id']:p for p in draft_pairs};map_by_case={m['case_id']:m for m in mapping}
    if len(draft_by_id)!=len(draft_pairs) or len(map_by_case)!=len(mapping):raise ValueError('Duplicate draft/map identity')
    if set(forms['A'])!=set(map_by_case):raise ValueError('Source draft coverage differs')
    selected=[];rejected=[];groups=set()
    for m in sorted(mapping,key=lambda r:r['candidate_order']):
        case=m['case_id'];a=forms['A'][case];b=forms['B'][case]
        if a['sampleId']!=m['sample_id'] or b['sampleId']!=m['sample_id']:
            raise ValueError('Form and source sample identity mismatch')
        pair=copy.deepcopy(draft_by_id[m['sample_id']])
        if any(r.get('source_changes') for r in [a,b]):raise ValueError('Submitted source text was changed')
        for r in [a,b]:
            if len(r['decisions'])!=7 or any(v not in {'是','否','不确定'} for v in r['decisions']):
                raise ValueError('Incomplete or unknown AI form value')
        row=dict(case_id=case,candidate_order=m['candidate_order'],review_source='Gemini_AI_assisted',
                 independent_human_reviews=False,form_decisions={'A':a['decisions'],'B':b['decisions']},pair=pair)
        if a['decisions']!=['是']*7 or b['decisions']!=['是']*7:
            rejected.append(dict(case_id=case,candidate_order=m['candidate_order'],reason='not_all_seven_yes_in_both_AI_form_versions'));continue
        validate_structure(pair)
        if pair['group_id'] in groups:raise ValueError('Selected pairs share a base group')
        groups.add(pair['group_id']);selected.append(row)
    return selected,rejected


def summarize_contrast(gold,original,cue,control):
    y=np.asarray(gold);o=np.asarray(original);a=np.asarray(cue);b=np.asarray(control)
    if y.ndim!=1 or not len(y) or o.ndim!=2 or a.shape!=o.shape or b.shape!=o.shape or len(o)!=len(y):
        raise ValueError('Aligned nonempty score matrices required')
    if not all(np.isfinite(x).all() for x in [o,a,b]) or not np.isin(y,np.arange(o.shape[1])).all():
        raise ValueError('Invalid labels or nonfinite scores')
    op=o.argmax(1);ap=a.argmax(1);bp=b.argmax(1);oc=op==y;ac=ap==y;bc=bp==y
    denom=int(oc.sum());rate=lambda n:float(n/denom) if denom else None
    ai=int((oc&~ac).sum());bi=int((oc&~bc).sum())
    return dict(n=len(y),original_correct=denom,cue_correct=int(ac.sum()),control_correct=int(bc.sum()),
        original_accuracy=float(oc.mean()),cue_accuracy=float(ac.mean()),control_accuracy=float(bc.mean()),
        cue_minus_control_accuracy=float(ac.mean()-bc.mean()),
        cue_prediction_flips=int((op!=ap).sum()),control_prediction_flips=int((op!=bp).sum()),
        introduced_error_denominator=denom,cue_introduced_errors=ai,control_introduced_errors=bi,
        cue_introduced_error_rate=rate(ai),control_introduced_error_rate=rate(bi),
        control_correct_cue_wrong=int((bc&~ac).sum()),cue_correct_control_wrong=int((ac&~bc).sum()),
        original_wrong_count=int((~oc).sum()),cue_repairs_original_error=int((~oc&ac).sum()),
        control_repairs_original_error=int((~oc&bc).sum()),
        interpretation='Descriptive paired results conditional on AI-accepted drafts; no population/causal significance claim')


def prepare(root):
    root=Path(root).resolve();project=root.parent;folder=root/REL
    from .pilots import _context,_selected_checkpoint
    from .train import _verified_bundle
    context=_context(root,'Bank_M1_SVM')
    audit_dir=project/'outputs/supplement14-human-review-20261108/build'
    audit=read_json(audit_dir/'submitted-review-audit.json')
    if audit['accepted_as_real_independent_reviews'] is not False or '由gemini补齐的' not in audit['source_human_provenance']:
        raise ValueError('AI review provenance must be explicit')
    submitted=read_json(audit_dir/'completed-review-readonly.json')
    for report in submitted['reports']:
        if hash_file(Path(report['file']))!=audit['submitted_file_sha256'][report['reviewer']]:
            raise ValueError('Submitted Excel bytes changed since audit')
    human=root/'reports/overnight/supplement-1-4/human-review'
    drafts=[json.loads(x) for x in (human/'AI-draft-pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    mapping=[json.loads(x) for x in (human/'arm-key-do-not-open-before-review.jsonl').read_text(encoding='utf-8').splitlines()]
    selected,rejected=select_ai_subset(submitted['reports'],drafts,mapping)
    if len(selected)!=37:raise ValueError('Expected the audited37 jointly affirmative draft pairs')
    source={r['sample_id']:r for r in context['candidates']}
    for row in selected:
        pair=row['pair'];template=source[pair['sample_id']]['pair_template']
        for field in ['mechanism','source_partition','sample_id','group_id','gold_labels','original_text','cue_words']:
            if pair[field]!=template[field]:raise ValueError('Registered source identity changed')
    _,svm_id,_,_= _selected_checkpoint(root,context)
    main=read_json(root/'reports/overnight/supplement-1-4/manifest.json');models=[]
    for model in ['tfidf_svm','distilbert']:
        matches=[r for r in main['trajectory_sources'] if r['dataset_id']=='banking77' and r['model_id']==model and r['training_rows']==6000]
        if len(matches)!=1:raise ValueError('Fixed6k source not unique')
        entry=matches[0]['entry'];rid=entry['run_id'];status,_=_verified_bundle(root/'runs'/rid)
        if status['run']!=entry['run'] or model=='tfidf_svm' and rid!=svm_id:
            raise ValueError('Fixed selected checkpoint changed')
        models.append(dict(model_id=model,run_id=rid,entry=entry,status_sha256=hash_file(root/'runs'/rid/'status.json'),
                           role='primary' if model=='tfidf_svm' else 'auxiliary_fixed_recipe'))
    meta=context['meta']
    body=dict(scope='S4_AI_reviewed_exploratory_D_only',user_instruction='无所谓的，我相信gemini的结果',
        user_approved_AI_review=True,review_source='Gemini_two_form_versions_not_independent_humans',
        formal_human_pilot=False,formal_human_pilot_updated=False,official_test_predictions_allowed=False,
        original_candidate_count=80,draft_pair_count=56,selected_pair_count=len(selected),no_safe_draft_count=24,
        selection_rule='Both existing AI form versions answer all7 criteria yes; fixed candidate order, before edited-text scoring',
        power_planning_or_confirmation_authorized=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),pairs=selected,rejected_drafted_pairs=rejected,models=models,
        submitted_excel_sha256=audit['submitted_file_sha256'],submitted_snapshot_sha256=hash_file(audit_dir/'completed-review-readonly.json'),
        candidate_context_binding=context['binding'],
        prepared_sha256=hash_file(root/'data/prepared_manifest.json'),design_sha256=hash_file(root/'matrix/design_manifest.json'),
        common_inputs_file=meta['inputs']['file'],common_inputs_sha256=hash_file(root/meta['inputs']['file']),
        discovery_file=meta['partitions']['discovery']['file'],discovery_sha256=hash_file(root/meta['partitions']['discovery']['file']),
        driver_sha256=hash_file(Path(__file__)),scorer_sha256=hash_file(Path(__file__).with_name('supplemental.py')),
        generalization='37 AI-accepted draftable examples only; not the completeD population or a human-confirmed mechanism')
    frozen={**body,'freeze_hash':canonical_hash(body)}
    path=folder/'inputs-frozen.json'
    if path.exists():
        old=verify_freeze(root)
        if old['pairs']!=selected or old['models']!=models:raise ValueError('Existing AI supplement inputs are immutable')
        return old
    atomic_write_json(path,frozen,root);return frozen


def verify_freeze(root):
    root=Path(root);frozen=read_json(root/REL/'inputs-frozen.json')
    if frozen['freeze_hash']!=canonical_hash({k:v for k,v in frozen.items() if k!='freeze_hash'}):
        raise ValueError('AI supplement freeze changed')
    if frozen['scope']!='S4_AI_reviewed_exploratory_D_only' or not frozen['user_approved_AI_review'] or frozen['formal_human_pilot']:
        raise ValueError('AI supplement authorization/scope changed')
    for path,key in [(root/'data/prepared_manifest.json','prepared_sha256'),(root/'matrix/design_manifest.json','design_sha256'),
                     (Path(__file__),'driver_sha256'),(Path(__file__).with_name('supplemental.py'),'scorer_sha256'),
                     (root/frozen['common_inputs_file'],'common_inputs_sha256'),(root/frozen['discovery_file'],'discovery_sha256')]:
        if hash_file(path)!=frozen[key]:raise ValueError(f'Frozen resource changed: {key}')
    for row in frozen['pairs']:validate_structure(row['pair'])
    return frozen


def evaluate(root):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Cluster CPU inference requires Slurm')
    from transformers import AutoTokenizer
    from .orchestrator import load_case
    from .overnight_training import runtime_assets
    from .pilots import _visible_pair_input
    from .supplemental import score_checkpoint_cpu
    root=Path(root).resolve();folder=root/REL;frozen=verify_freeze(root)
    target=folder/'result-manifest.json'
    if target.exists():
        previous=read_json(target)
        if previous['freeze_hash']!=frozen['freeze_hash'] or hash_file(folder/'results.json')!=previous['results_sha256']:
            raise ValueError('Existing AI-reviewed inference result changed')
        return previous
    pairs=[r['pair'] for r in frozen['pairs']];meta=read_json(root/'data/prepared_manifest.json')['datasets']['banking77']
    data,actual_inputs=load_case(root,meta,(),selection_ids=(),discovery_ids=[p['sample_id'] for p in pairs])
    actual={r.sample_id:r for r in data.partitions['discovery']}
    assets=runtime_assets(read_json(root/'assets/assets.lock.json'),root.parent)
    tokenizer=AutoTokenizer.from_pretrained(assets['models']['distilbert']['directory'],local_files_only=True)
    visible=[]
    for pair in pairs:
        sid=pair['sample_id'];row=actual[sid]
        if list(row.labels)!=pair['gold_labels'] or row.group_id!=pair['group_id'] or actual_inputs[sid].text!=pair['original_text']:
            raise ValueError('Actual D input/group/gold differs from frozen source')
        visible.extend(_visible_pair_input(pair,role,tokenizer) for role in ['original','intervention','control'])
    y=np.asarray([p['gold_labels'][0] for p in pairs]);models=[]
    for record in frozen['models']:
        output=root/'runs'/record['run_id']
        if hash_file(output/'status.json')!=record['status_sha256']:raise ValueError('Fixed checkpoint status changed')
        scores,status=score_checkpoint_cpu(output,tuple(visible))
        if status['run']!=record['entry']['run'] or scores.shape!=(3*len(pairs),len(meta['label_names'])):
            raise ValueError('Checkpoint identity or prediction shape changed')
        original=scores[0::3];cue=scores[1::3];control=scores[2::3]
        summary=summarize_contrast(y,original,cue,control);rows=[]
        for i,pair in enumerate(pairs):
            row=dict(case_id=frozen['pairs'][i]['case_id'],sample_id=pair['sample_id'],group_id=pair['group_id'],
                     gold_label=int(y[i]),gold_name=meta['label_names'][int(y[i])])
            for role,values in [('original',original),('cue',cue),('control',control)]:
                pred=int(values[i].argmax());row[role]=dict(prediction=pred,prediction_name=meta['label_names'][pred],
                    correct=pred==int(y[i]),scores=values[i].tolist())
            rows.append(row)
        models.append(dict(model_id=record['model_id'],role=record['role'],run_id=record['run_id'],score_kind=status['score_kind'],
                           checkpoint_status_sha256=record['status_sha256'],summary=summary,pairs=rows))
        print(json.dumps(dict(model=record['model_id'],summary=summary)),flush=True)
    if verify_freeze(root)!=frozen:raise ValueError('Frozen input changed during inference')
    for record in frozen['models']:
        if hash_file(root/'runs'/record['run_id']/'status.json')!=record['status_sha256']:raise ValueError('Checkpoint changed during inference')
    predicted_at=datetime.now(timezone.utc).isoformat()
    if datetime.fromisoformat(predicted_at)<=datetime.fromisoformat(frozen['frozen_at_utc']):raise ValueError('Inference must follow freezing')
    payload=dict(scope=frozen['scope'],freeze_hash=frozen['freeze_hash'],predicted_at_utc=predicted_at,models=models,
                 label_names=list(meta['label_names']),review_source=frozen['review_source'],official_test_predictions_generated=False,
                 formal_human_pilot=False,scoring_device='cpu',job_id=os.environ['SLURM_JOB_ID'])
    atomic_write_json(folder/'results.json',payload,root)
    result=dict(status='complete',scope=frozen['scope'],selected_pair_count=len(pairs),freeze_hash=frozen['freeze_hash'],
        results_sha256=hash_file(folder/'results.json'),job_id=os.environ['SLURM_JOB_ID'],review_source=frozen['review_source'],
        formal_human_pilot=False,official_test_predictions_generated=False,models=[{k:v for k,v in m.items() if k!='pairs'} for m in models])
    atomic_write_json(target,result,root);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','evaluate']);parser.add_argument('--root',type=Path,default=Path('research-v3'))
    args=parser.parse_args();result=prepare(args.root) if args.action=='prepare' else evaluate(args.root)
    print(json.dumps({k:v for k,v in result.items() if k not in {'pairs','rejected_drafted_pairs','models'}},ensure_ascii=False))


if __name__=='__main__':main()

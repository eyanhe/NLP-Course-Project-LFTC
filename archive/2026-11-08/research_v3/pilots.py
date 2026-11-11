"""Import human D edits, freeze before prediction, and score fixed anchors.

No edit or review is generated here. A caller supplies reviewed pairs and
explicitly invokes each action. This workflow never creates selection/evaluation
locks, enables a training phase, reads Q, or marks the research complete.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import canonical_hash
from .interventions import build_evidence_windows, plan_pair_count, validate_pair
from .partitions import clean_text
from .storage import atomic_write_json, bounded_path, hash_file, read_json
from .types import VisibleInput


def _now():
    return datetime.now(timezone.utc).isoformat()


def _directory(root, pilot_id):
    if not isinstance(pilot_id,str) or not pilot_id or Path(pilot_id).name!=pilot_id or pilot_id in {'.','..'}:
        raise ValueError('Invalid pilot identity')
    return bounded_path(Path(root)/'training/discovery-pilots'/pilot_id,root)


def _context(root, pilot_id):
    """Authenticate the declared packet and actual D bytes without reading Q."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    root=Path(root).resolve();_directory(root,pilot_id)
    from .matrix import verify_design_lock
    if not (root/'design.lock.json').is_file():raise PermissionError('A verified design lock is required before real D pilots')
    design_lock=verify_design_lock(root)
    design=read_json(root/'matrix/design_manifest.json')
    if design_lock['bindings']['design_hash']!=design['design_hash']:
        raise ValueError('Pilot design-lock identity changed')
    nodes=[node for node in design['discovery_pilot_nodes'] if node['pilot_id']==pilot_id]
    if len(nodes)!=1:raise ValueError('Pilot is not uniquely declared by the design')
    node=nodes[0];dataset=node['dataset_id'];mechanism=node['mechanism']
    expected={'Bank_M1_SVM':('banking77','M1'),
              'Yelp_M4_DistilBERT':('yelp','M4'),
              'Yahoo_M4_DistilBERT':('yahoo_answers_topics','M4')}
    if expected.get(pilot_id)!=(dataset,mechanism):raise ValueError('Pilot dataset/mechanism identity changed')
    directory=bounded_path(root/'training/discovery-review'/pilot_id,root)
    manifest=read_json(directory/'manifest.json')
    index=read_json(directory.parent/'manifest.json')
    registered=[row for row in index['pilots'] if row['pilot_id']==pilot_id]
    if len(registered)!=1 or registered[0]!=manifest:raise ValueError('Pilot packet manifest identity changed')
    model='tfidf_svm' if mechanism=='M1' else 'distilbert'
    selection_node=f'select:{dataset}:{model}'
    if (manifest.get('dataset_id')!=dataset or manifest.get('mechanism')!=mechanism or
            manifest.get('source_partition')!='discovery' or manifest.get('pilot_n_required')!=50 or
            manifest.get('candidate_order_fixed_before_model_reveal') is not True or
            manifest.get('model_predictions_in_packet') is not False or
            manifest.get('selection_node_reference')!=selection_node or
            manifest.get('design_hash')!=design['design_hash']):
        raise ValueError('Pilot packet declaration/source identity changed')
    prepared=read_json(root/'data/prepared_manifest.json')
    if not prepared.get('complete') or manifest['prepared_hash']!=prepared['prepared_hash']:
        raise ValueError('Pilot prepared source identity changed')
    if canonical_hash({k:v for k,v in prepared.items() if k!='prepared_hash'})!=prepared['prepared_hash']:
        raise ValueError('Prepared source manifest content/hash changed')
    meta=prepared['datasets'][dataset];source_record=meta['partitions']['discovery']
    source_path=bounded_path(root/source_record['file'],root)
    if hash_file(source_path)!=source_record['sha256']:raise ValueError('Pilot D source file hash changed')
    source={row['sample_id']:row for row in pq.read_table(source_path,
        columns=['sample_id','text','labels','group_id']).to_pylist()}
    candidates_path=bounded_path(directory/manifest['candidate_file'],directory)
    if hash_file(candidates_path)!=manifest['candidate_file_sha256']:
        raise ValueError('Pilot candidate packet file hash changed')
    with candidates_path.open(encoding='utf-8-sig') as stream:
        candidates=[json.loads(line) for line in stream if line.strip()]
    orders=[row['candidate_order'] for row in candidates]
    if sorted(orders)!=list(range(len(candidates))):raise ValueError('Pilot candidate order is not unique/complete')
    candidates.sort(key=lambda row:row['candidate_order'])
    if len({row['sample_id'] for row in candidates})!=len(candidates):
        raise ValueError('Duplicate candidate source identities')
    for candidate in candidates:
        original=source.get(candidate['sample_id'])
        if (original is None or candidate['dataset_id']!=dataset or candidate['group_id']!=original['group_id'] or
                candidate['gold_labels']!=original['labels'] or
                candidate['source_full_text_normalized']!=clean_text(original['text']) or
                candidate['common_visible_text']!=clean_text(original['text'])[:candidate['common_visible_end_char']]):
            raise ValueError('Candidate text/gold/group does not match actual D source provenance')
    assets=read_json(root/'assets/assets.lock.json')
    if assets.get('asset_hash')!=prepared['identity']['asset_hash']:
        raise ValueError('Pilot tokenizer asset identity changed')
    tokenizer=AutoTokenizer.from_pretrained(assets['models']['distilbert']['directory'],local_files_only=True)
    from .inputs import make_visible_input
    from .types import Example
    for candidate in candidates:
        original=source[candidate['sample_id']]
        visible=make_visible_input(Example(dataset,candidate['sample_id'],original['text'],
            tuple(original['labels']),'discovery',original['group_id']),tokenizer)
        expected_original=visible.text if mechanism=='M1' else clean_text(original['text'])
        template=candidate['pair_template']
        expected={'mechanism':mechanism,'source_partition':'discovery','sample_id':candidate['sample_id'],
            'group_id':original['group_id'],'gold_labels':original['labels'],'original_text':expected_original}
        if (any(template.get(k)!=v for k,v in expected.items()) or candidate['common_visible_text']!=visible.text
                or candidate['common_visible_end_char']!=visible.end_char
                or candidate['full_content_tokens']!=visible.full_tokens):
            raise ValueError('Candidate pair template source provenance changed')
    selection_nodes=[row for row in design['selection_nodes'] if row['node_id']==selection_node]
    if len(selection_nodes)!=1:raise ValueError('Pilot anchor selection identity is not uniquely declared')
    binding={'design_hash':design['design_hash'],'prepared_manifest_sha256':hash_file(root/'data/prepared_manifest.json'),
             'candidate_file_sha256':hash_file(candidates_path),'packet_manifest_sha256':hash_file(directory/'manifest.json'),
             'source_file_sha256':hash_file(source_path),'asset_hash':assets['asset_hash'],
             'tokenizer_vocab_sha256':canonical_hash(tokenizer.get_vocab())}
    return {'node':node,'manifest':manifest,'meta':meta,'candidates':candidates,'source':source,
            'tokenizer':tokenizer,'binding':binding,'selection_node':selection_node,'model_id':model,
            'selection_definition':selection_nodes[0]}


def _authenticate_pair(pair, candidate, context):
    template=candidate['pair_template'];mechanism=context['node']['mechanism']
    fields=['mechanism','source_partition','sample_id','group_id','gold_labels','original_text']
    if any(pair.get(field)!=template.get(field) for field in fields):
        raise ValueError('Reviewed pair source/gold/group provenance changed')
    if pair['mechanism']!=mechanism or pair['source_partition']!='discovery':
        raise ValueError('Reviewed pair must use the declared D source/mechanism')
    if mechanism=='M1':
        if pair.get('cue_words')!=template.get('cue_words'):
            raise ValueError('Registered training cue words changed')
        for role in ['intervention','control']:
            text=pair.get(f'{role}_text')
            if not isinstance(text,str):continue  # validate_pair records incomplete human work.
            tokens=context['tokenizer'](text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
            if len(tokens)>254:raise ValueError('Human edit exceeds the fixed common-visible input budget')
    else:
        windows=pair.get('windows')
        if not isinstance(windows,dict):return  # still-unreviewed candidates are not valid pairs.
        source=candidate['source_full_text_normalized']
        offsets=context['tokenizer'](source,add_special_tokens=False,truncation=False,
            return_offsets_mapping=True,verbose=False)['offset_mapping']
        span=windows.get('evidence_span')
        if not span:raise ValueError('M4 source evidence span is missing')
        expected=build_evidence_windows(source,offsets,span,pair['sample_id'])
        if windows!=expected or pair.get('evidence_span') not in (None,span):
            raise ValueError('M4 window text/offsets do not match the deterministic actual source windows')


def _select_reviewed(context, reviewed_pairs):
    candidates={row['sample_id']:row for row in context['candidates']}
    imported={};rejections=[]
    for raw in reviewed_pairs:
        # Canonical JSON copies detach the freeze from later caller mutations.
        pair=json.loads(json.dumps(raw,ensure_ascii=False,allow_nan=False))
        sid=pair.get('sample_id')
        if sid not in candidates:raise ValueError('Reviewed source ID is outside the declared D candidates')
        if sid in imported:raise ValueError('Duplicate reviewed source identity')
        _authenticate_pair(pair,candidates[sid],context)
        imported[sid]=pair
    selected=[];seen_groups=set()
    for candidate in context['candidates']:
        pair=imported.get(candidate['sample_id'])
        if pair is None:
            if len(selected)<50:raise PermissionError('Earlier candidates need explicit human review decisions in fixed order')
            continue
        try:validate_pair(pair)
        except (ValueError,KeyError,TypeError) as exc:
            reason=pair.get('rejection_reason')
            reviewers=pair.get('reviews',[])
            if (not isinstance(reason,str) or not reason.strip() or not any(
                    isinstance(row.get('reviewer_id'),str) and row['reviewer_id'].strip()
                    and row.get('blind_to_predictions') is True for row in reviewers)):
                raise PermissionError('Incomplete human review cannot be treated as a rejected earlier candidate') from exc
            rejections.append({'sample_id':pair['sample_id'],'candidate_order':candidate['candidate_order'],
                               'status':'invalid_human_pair','reason':str(exc),'human_rejection_reason':reason})
            continue
        if pair['group_id'] in seen_groups:continue
        seen_groups.add(pair['group_id'])
        if len(selected)<50:
            selected.append({'candidate_order':candidate['candidate_order'],'pair':pair,
                             'edits_sha256':canonical_hash(pair)})
    if len(selected)!=50:
        raise PermissionError(f'Insufficient valid independent human-reviewed D pairs: {len(selected)}/50; reviews are never generated')
    return selected,rejections


def freeze_pilot_edits(root: Path,pilot_id: str,reviewed_pairs) -> dict:
    """Freeze exactly the first50 valid reviewed base groups in declared order."""
    root=Path(root).resolve();directory=_directory(root,pilot_id);context=_context(root,pilot_id)
    order={row['sample_id']:row['candidate_order'] for row in context['candidates']}
    reviewed_pairs=json.loads(json.dumps(list(reviewed_pairs),ensure_ascii=False,allow_nan=False))
    frozen_path=directory/'edits.freeze.json';edits_path=directory/'edits.json'
    if frozen_path.exists():
        previous=_verified_edits(root,pilot_id,context)
        submitted=sorted(reviewed_pairs,key=lambda row:order.get(row.get('sample_id'),len(order)))
        if previous['payload']['reviewed_pairs']!=submitted:
            raise ValueError('Previously frozen pilot edits are immutable; submitted edits changed')
        return previous['freeze']
    selected,rejections=_select_reviewed(context,reviewed_pairs)
    reviewed_pairs.sort(key=lambda row:order[row['sample_id']])
    if edits_path.exists():raise ValueError('Uncommitted pilot edits already exist; explicit recovery is required')
    frozen_at=_now()
    payload={'schema_version':1,'pilot_id':pilot_id,'source_partition':'discovery',
             'dataset_id':context['node']['dataset_id'],'mechanism':context['node']['mechanism'],
             'edits_frozen_at_utc':frozen_at,'pairs':selected,'reviewed_pairs':reviewed_pairs,
             'rejected_reviewed_pairs':rejections,
             'candidate_order_fixed_before_model_reveal':True,'model_predictions_generated':False}
    atomic_write_json(edits_path,payload,root)
    frozen={'schema_version':1,'pilot_id':pilot_id,'status':'edits_frozen','pilot_n':50,
            'source_partition':'discovery','edits_file':edits_path.relative_to(root).as_posix(),
            'edits_file_sha256':hash_file(edits_path),'edits_frozen_at_utc':frozen_at,
            'binding':context['binding'],'official_test_predictions_generated':False}
    frozen['freeze_hash']=canonical_hash(frozen)
    atomic_write_json(frozen_path,frozen,root)
    return frozen


def _verified_edits(root,pilot_id,context=None):
    context=_context(root,pilot_id) if context is None else context
    directory=_directory(root,pilot_id);frozen=read_json(directory/'edits.freeze.json')
    if canonical_hash({k:v for k,v in frozen.items() if k!='freeze_hash'})!=frozen.get('freeze_hash'):
        raise ValueError('Pilot edit freeze content/hash changed')
    if frozen.get('pilot_id')!=pilot_id or frozen.get('binding')!=context['binding']:
        raise ValueError('Frozen pilot source/candidate/tokenizer binding changed')
    path=bounded_path(Path(root)/frozen['edits_file'],directory)
    if hash_file(path)!=frozen['edits_file_sha256']:raise ValueError('Frozen pilot edits file hash changed')
    payload=read_json(path)
    if payload.get('pilot_id')!=pilot_id or payload.get('source_partition')!='discovery' or len(payload.get('pairs',[]))!=50:
        raise ValueError('Frozen pilot identity/count changed')
    if payload['edits_frozen_at_utc']!=frozen['edits_frozen_at_utc']:
        raise ValueError('Frozen pilot edit timestamp changed')
    selected,rejections=_select_reviewed(context,payload['reviewed_pairs'])
    if selected!=payload['pairs']:raise ValueError('Frozen canonical pair hashes/order changed')
    if rejections!=payload['rejected_reviewed_pairs']:raise ValueError('Frozen human rejection decisions changed')
    return {'freeze':frozen,'payload':payload,'context':context}


def _selected_checkpoint(root,context):
    from .freeze import _checkpoint
    from .train import _verified_bundle
    selections=read_json(Path(root)/'selections.json')
    if selections.get('design_hash')!=context['binding']['design_hash']:
        raise ValueError('Selected anchor design identity changed')
    choice=selections.get('choices',{}).get(context['selection_node'],{})
    if choice.get('status')!='complete':raise PermissionError('The declared selected anchor checkpoint is pending')
    if choice.get('dataset_id')!=context['node']['dataset_id'] or choice.get('model_id')!=context['model_id']:
        raise ValueError('Pilot selected anchor dataset/model identity changed')
    run_id=choice['selected_run_id'];status=_checkpoint(root,run_id,{})
    output=Path(root)/'runs'/run_id
    checkpoint_evidence=choice.get('checkpoint_evidence_sha256') if choice.get('selection_policy')=='fixed_defaults' else choice.get('evidence_sha256')
    if hash_file(output/'status.json')!=checkpoint_evidence:
        raise ValueError('Selected checkpoint evidence hash changed')
    _,bundle=_verified_bundle(output)
    if bundle.get('model_id')!=context['model_id'] or status.get('run',{}).get('model_id')!=context['model_id']:
        raise ValueError('Pilot selected checkpoint model family changed')
    run=status.get('run',{})
    if run.get('condition_id')!=context['selection_definition']['anchor_condition_id'] or run.get('strict_no_val') is not False:
        raise ValueError('Pilot must use the declared E1 anchor condition identity')
    if run.get('training_seed')!=(None if context['model_id']=='tfidf_svm' else 42):
        raise ValueError('Pilot must use the predeclared seed42 checkpoint')
    return selections,run_id,output,status


def _visible_pair_input(pair,role,tokenizer):
    if pair['mechanism']=='M1':text=pair[f'{role}_text'];start=0;end=len(text)
    else:
        window=pair['windows'][role];text=window['text'];start=window['char_start'];end=window['char_end']
    count=len(tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids'])
    if count>254:raise ValueError('Edited/window input exceeds fixed common-visible budget')
    return VisibleInput(f'{pair["sample_id"]}/{role}',text,start,end,count,count,
                        hashlib.sha256(text.encode()).hexdigest())


def evaluate_pilot(root: Path,pilot_id: str) -> dict:
    """Verify frozen human edits first, then score the actual fixed anchor."""
    from .freeze import _verify_pilot
    from .train import score_saved
    root=Path(root).resolve();verified=_verified_edits(root,pilot_id)
    context=verified['context'];selections,run_id,output,status=_selected_checkpoint(root,context)
    directory=_directory(root,pilot_id);metadata_path=directory/'result-manifest.json'
    if metadata_path.exists():
        previous=read_json(metadata_path)
        _verify_pilot(root,context['node'],previous,selections['choices'],{})
        _verify_result_edits(root,previous,verified)
        return previous
    pairs=verified['payload']['pairs'];tokenizer=context['tokenizer'];visible=[]
    for row in pairs:
        visible.extend(_visible_pair_input(row['pair'],role,tokenizer) for role in ['intervention','control'])
    scores=score_saved(output,tuple(visible))
    expected=(100,len(context['meta']['label_names']))
    if scores.shape!=expected or not np.isfinite(scores).all():raise ValueError('Pilot scores have invalid label shape/values')
    original_scores=None
    if context['node']['mechanism']=='M1':
        original_scores=score_saved(output,tuple(_visible_pair_input(row['pair'],'original',tokenizer) for row in pairs))
        if original_scores.shape!=(50,expected[1]) or not np.isfinite(original_scores).all():
            raise ValueError('Pilot original-input scores have invalid shape/values')
    # Detect concurrent edits/checkpoint changes before recording predictions.
    after=_verified_edits(root,pilot_id)
    _,after_run,_,after_status=_selected_checkpoint(root,context)
    if after['freeze']!=verified['freeze'] or after_run!=run_id or canonical_hash(after_status)!=canonical_hash(status):
        raise ValueError('Pilot frozen edits or selected checkpoint changed during prediction')
    predicted_at=_now();frozen_at=verified['freeze']['edits_frozen_at_utc']
    if datetime.fromisoformat(predicted_at)<=datetime.fromisoformat(frozen_at):
        raise ValueError('Predictions must follow edit/review freezing')
    records=[];discordant=0
    for index,row in enumerate(pairs):
        a=scores[2*index];b=scores[2*index+1];gold=int(row['pair']['gold_labels'][0])
        discordant+=int((int(a.argmax())==gold)!=(int(b.argmax())==gold))
        records.append({**row,'checkpoint_run_id':run_id,'edits_frozen_at_utc':frozen_at,
                        'predicted_at_utc':predicted_at,'intervention_scores':a.tolist(),'control_scores':b.tolist()})
        if original_scores is not None:records[-1]['original_scores']=original_scores[index].tolist()
    payload={'schema_version':1,'pilot_id':pilot_id,'source_partition':'discovery','pairs':records,
             'checkpoint_status_sha256':hash_file(output/'status.json'),
             'edits_file_sha256':verified['freeze']['edits_file_sha256'],
             'score_kind':status['score_kind'],'official_test_predictions_generated':False}
    results_path=directory/'results.json';atomic_write_json(results_path,payload,root)
    result={'status':'complete','pilot_id':pilot_id,'source_partition':'discovery','pilot_n':50,
            'discordant':discordant,'n_planned':plan_pair_count(discordant,50),
            'results_file':results_path.relative_to(root).as_posix(),'results_sha256':hash_file(results_path),
            'checkpoint_run_id':run_id,'edits_file_sha256':verified['freeze']['edits_file_sha256'],
            'edits_frozen_at_utc':frozen_at,'predicted_at_utc':predicted_at,
            'official_test_predictions_generated':False}
    _verify_pilot(root,context['node'],result,selections['choices'],{})
    atomic_write_json(metadata_path,result,root)
    return result


def _verify_result_edits(root,result,verified):
    payload=read_json(bounded_path(Path(root)/result['results_file'],root))
    actual=[{k:row[k] for k in ['candidate_order','pair','edits_sha256']} for row in payload['pairs']]
    if (actual!=verified['payload']['pairs'] or
            result.get('edits_file_sha256')!=verified['freeze']['edits_file_sha256'] or
            payload.get('edits_file_sha256')!=verified['freeze']['edits_file_sha256'] or
            any(row['edits_frozen_at_utc']!=verified['freeze']['edits_frozen_at_utc'] for row in payload['pairs'])):
        raise ValueError('Pilot result pairs do not match the frozen human edits')


def merge_pilot_result(root: Path,pilot_id: str) -> dict:
    """Explicit caller action: merge an authenticated completed pilot only.

    This does not create any lock or enable training/Q. evaluate_pilot itself
    leaves selections untouched.
    """
    from .freeze import _verify_pilot
    root=Path(root).resolve()
    if (root/'selection.lock.json').exists() or (root/'evaluation.freeze.json').exists():
        raise PermissionError('Pilot selections are frozen by the global lock')
    verified=_verified_edits(root,pilot_id)
    selections,run_id,_,_=_selected_checkpoint(root,verified['context'])
    result=read_json(_directory(root,pilot_id)/'result-manifest.json')
    _verify_pilot(root,verified['context']['node'],result,selections['choices'],{})
    _verify_result_edits(root,result,verified)
    if result.get('checkpoint_run_id')!=run_id or result.get('edits_file_sha256')!=verified['freeze']['edits_file_sha256']:
        raise ValueError('Pilot result metadata does not match frozen edits/selected checkpoint')
    selections.setdefault('discovery_pilots',{})[pilot_id]=result
    atomic_write_json(root/'selections.json',selections,root)
    return selections

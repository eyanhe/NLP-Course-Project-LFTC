"""Post-freeze human confirmation sampling and fixed-checkpoint F6 views.

No human edit/review, D-pilot receipt, or research freeze is synthesized. Each
public API first verifies the real evaluation freeze. Audit frame references
are required for all four declared audit datasets; no outcome-based selection
or user-supplied exclusion list substitutes for authenticated actual frames.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .config import canonical_hash
from .audit import draw_audit
from .freeze import _checkpoint, verify_freeze
from .inputs import make_visible_input
from .interventions import (build_evidence_windows, english_words, extract_cues,
                            pilot_candidate_packets, validate_pair)
from .partitions import clean_text
from .pilots import _visible_pair_input
from .storage import atomic_write_json, bounded_path, hash_file, read_json
from .types import Example, PredictionRow, PreparedDataset, VisibleInput

AUDIT_DATASETS = {'yelp','go_emotions','banking77','yahoo_answers_topics'}
PILOTS = {'Bank_M1_SVM':('banking77','M1'), 'Yelp_M4_DistilBERT':('yelp','M4'),
          'Yahoo_M4_DistilBERT':('yahoo_answers_topics','M4')}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _directory(root,pilot_id):
    if pilot_id not in PILOTS:
        raise ValueError('Unknown declared confirmation pilot identity')
    return bounded_path(root/'confirmation'/pilot_id,root)


def _checked_file(root,record,key='sha256'):
    path=bounded_path(root/record['file'],root)
    if not path.is_file() or hash_file(path)!=record[key]:
        raise ValueError(f'Frozen source file bytes/hash changed: {record["file"]}')
    return path


def _partition(root,meta,name,columns):
    import pyarrow.parquet as pq
    return pq.read_table(_checked_file(root,meta['partitions'][name]),columns=columns).to_pylist()


def _input_rows(root,meta,partition):
    import pyarrow.parquet as pq
    path=_checked_file(root,meta['inputs'],'file_sha256')
    records=pq.read_table(path,columns=['sample_id','text','start_char','end_char','content_tokens','full_tokens','text_hash'],
                          filters=[('partition','=',partition)]).to_pylist()
    inputs={r['sample_id']:VisibleInput(**r) for r in records}
    if len(inputs)!=len(records):raise ValueError('Duplicate frozen visible input IDs')
    for value in inputs.values():
        if value.text_hash!=hashlib.sha256(value.text.encode()).hexdigest():
            raise ValueError('Frozen visible input text hash changed')
    return inputs


def _train_ids(root,condition):
    if 'train_ids' in condition:ids=condition['train_ids']
    else:
        path=bounded_path(root/condition['train_ids_file'],root)
        if hash_file(path)!=condition['train_ids_file_sha256']:raise ValueError('Anchor training ID file changed')
        with gzip.open(path,'rt',encoding='utf-8') as stream:ids=[line.rstrip('\n') for line in stream]
    if len(ids)!=6000 or len(set(ids))!=6000:raise ValueError('M1 cues require the unique fixed 6000-row anchor')
    digest=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    if digest!=condition['training_set_hash']:raise ValueError('Anchor training IDs hash changed')
    return sorted(ids)


def _audit_exclusions(root,prepared,audit_frames):
    if not isinstance(audit_frames,dict) or set(audit_frames)!=AUDIT_DATASETS:
        raise PermissionError('Actual audit frame references for all four datasets are pending')
    excluded=set();sources={};bindings={}
    for ds in sorted(AUDIT_DATASETS):
        path=_checked_file(root,audit_frames[ds]);packet=read_json(path)
        if (packet.get('dataset_id')!=ds or packet.get('purpose')!='audit'
                or packet.get('selection_policy')!='uniform_groups_then_uniform_row'
                or packet.get('model_outcomes_used_for_sampling') is not False
                or type(packet.get('sampling_seed')) is not int or packet['sampling_seed']!=7301):
            raise ValueError('Audit frame source/sampling declaration changed')
        meta=prepared['datasets'][ds]
        source=_partition(root,meta,'test',['sample_id','text','labels','group_id'])
        lookup={r['sample_id']:r for r in source};groups={r['group_id'] for r in source}
        if len(lookup)!=len(source) or not all(groups):raise ValueError('Actual Q source IDs/groups are invalid')
        records=packet.get('records',[])
        if len(records)!=min(200,len(groups)) or len({r['group_id'] for r in records})!=len(records):
            raise ValueError('Audit frame must contain the fixed 200 or all available independent Q groups')
        for row in records:
            actual=lookup.get(row['sample_id'])
            if actual is None or row['group_id']!=actual['group_id'] or row.get('labels')!=actual['labels']:
                raise ValueError('Audit sample/gold/group does not match actual Q source')
            excluded.add(row['group_id'])
        expected=_fixed_audit_draw(meta,source)
        if _audit_mapping(records,ds)!=_audit_mapping(expected,ds):
            raise ValueError('Audit frame is not the predeclared seed7301 uniform group/row draw')
        sources[ds]=source;bindings[ds]={'file':audit_frames[ds]['file'],'sha256':hash_file(path)}
    # Exclude all D base groups across datasets, a superset of all pilot groups.
    discovery_files={}
    for ds,meta in prepared['datasets'].items():
        rows=_partition(root,meta,'discovery',['sample_id','group_id'])
        if len({r['sample_id'] for r in rows})!=len(rows) or any(not r['group_id'] for r in rows):
            raise ValueError('Actual D source IDs/groups are invalid')
        excluded.update(r['group_id'] for r in rows)
        discovery_files[ds]=meta['partitions']['discovery']['sha256']
    return excluded,sources,bindings,discovery_files


def _fixed_audit_draw(meta,source):
    rows=tuple(Example(meta['dataset_id'],r['sample_id'],r['text'],tuple(r['labels']),'test',r['group_id'])
               for r in source)
    data=PreparedDataset(meta['dataset_id'],meta['task'],tuple(meta['label_names']),{'test':rows},meta['manifest_hash'])
    return draw_audit(data,'audit',200,7301,set())


def _audit_mapping(records,dataset_id=None):
    """Identity comparison permits serialization order, never alternate A rows."""
    result={}
    for row in records:
        ds=row.get('dataset_id',dataset_id)
        if ds not in AUDIT_DATASETS or (dataset_id is not None and ds!=dataset_id):
            raise ValueError('Audit frame includes undeclared/different dataset identities')
        if any(k in row for k in ('scores','prediction','model_id','model_outputs')):
            raise ValueError('Blind audit frame contains revealed model responses')
        sid=row.get('sample_id');group=row.get('group_id');labels=row.get('labels')
        if (not isinstance(sid,str) or not sid or not isinstance(group,str) or not group
                or not isinstance(labels,(list,tuple)) or not labels or (ds,sid) in result):
            raise ValueError('Audit frame identities/targets are missing or duplicated')
        result[(ds,sid)]=(group,tuple(labels))
    return result


def confirmation_candidates(data,inputs,mechanism,excluded_groups,cue_map=None):
    """Pure fixed-seed group permutation, one uniform eligible row per group.

    All algorithmically eligible remaining groups are offered in fixed order.
    Humans later decide eligibility without responses; first planned valid
    groups are selected only after preceding decisions have been recorded.
    """
    eligible=[];rejections=[]
    for row in data.partitions['test']:
        reason=None;source=clean_text(row.text);visible=inputs[row.sample_id]
        if row.group_id in excluded_groups:reason='excluded_A_or_discovery_base_group'
        elif len(row.labels)!=1:reason='requires_one_gold_label'
        elif mechanism=='M1':
            cues={r['word'] if isinstance(r,dict) else str(r) for r in (cue_map or {}).get(row.labels[0],[])}
            if not 5<=len(english_words(source))<=60:reason='M1_word_count_outside_5_60'
            elif not set(english_words(visible.text))&cues:reason='M1_no_registered_training_cue'
        elif mechanism=='M4':
            if not 320<=visible.full_tokens<=2048:reason='M4_content_tokens_outside_320_2048'
        else:raise ValueError('Unknown confirmation mechanism')
        if reason:rejections.append({'sample_id':row.sample_id,'group_id':row.group_id,'reason':reason})
        else:eligible.append(row)
    temporary=PreparedDataset(data.dataset_id,data.task,data.label_names,{'discovery':tuple(eligible)},data.manifest_hash)
    packets=list(pilot_candidate_packets(temporary,inputs,mechanism,seed=7301,cue_map=cue_map))
    for packet in packets:
        packet['pair_template']['source_partition']='confirmation'
        if mechanism=='M1':
            for template in packet['review_templates']:
                template.update(intervention_content_words_replaced=None,control_content_words_replaced=None)
    return packets,rejections


def _context(root,pilot_id,audit_frames=None):
    # This must precede every Q source, asset/model, or saved-edit read.
    evaluation=verify_freeze('evaluation',root)
    directory=_directory(root,pilot_id)
    selection=verify_freeze('selection',root)
    if audit_frames is None:
        if not (directory/'manifest.json').is_file():raise PermissionError('Confirmation audit preparation is pending')
        audit_frames=read_json(directory/'manifest.json')['audit_frames']
    design=read_json(root/'matrix/design_manifest.json')
    nodes=[n for n in design['discovery_pilot_nodes'] if n['pilot_id']==pilot_id]
    if len(nodes)!=1 or (nodes[0]['dataset_id'],nodes[0]['mechanism'])!=PILOTS[pilot_id]:
        raise ValueError('Confirmation pilot design identity changed')
    node=nodes[0];dataset,mechanism=PILOTS[pilot_id]
    plan=selection.get('pilots',{}).get(pilot_id,{})
    if (plan.get('pilot_n')!=50 or plan.get('source_partition')!='discovery'
            or plan.get('human_reviews_present') is not True):
        raise PermissionError('Authenticated actual human D pilot plan is pending')
    n=plan.get('n_planned')
    if not isinstance(n,int) or isinstance(n,bool) or not 400<=n<=1200:
        raise ValueError('Frozen D pilot sample-size plan is invalid')
    prepared=read_json(root/'data/prepared_manifest.json')
    if (not prepared.get('complete') or canonical_hash({k:v for k,v in prepared.items() if k!='prepared_hash'})
            !=prepared.get('prepared_hash')):
        raise ValueError('Prepared source manifest hash changed')
    meta=prepared['datasets'][dataset]
    if meta['task']!='single_label':raise ValueError('F6 behavioral confirmations are single-label tasks')
    contrasts=read_json(root/'contrasts.json');manifest=read_json(root/'run_manifest.json')
    if contrasts['design_hash']!=design['design_hash'] or manifest['design_hash']!=design['design_hash']:
        raise ValueError('Confirmation registry design identity changed')
    model='tfidf_svm' if mechanism=='M1' else 'distilbert';seed=None if mechanism=='M1' else 42
    checkpoint=plan['checkpoint_run_id']
    views={}
    for arm in ['intervention','control']:
        view_id=f'eval:{pilot_id}:{arm}';view=contrasts.get('runs',{}).get(view_id,{})
        expected={'dataset_id':dataset,'model_id':model,'data_seed':2026,'training_seed':seed,
                  'checkpoint_run_id':checkpoint,'evaluation_id':view_id,'n_planned':n,
                  'base_group_namespace':f'paired:{pilot_id}'}
        if any(view.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed F6 registry checkpoint/view identity changed')
        views[view_id]=view
    entries=[r for r in manifest['runs'] if r.get('artifact_id',r['run_id'])==checkpoint]
    if not entries or any(r['dataset_id']!=dataset for r in entries):raise ValueError('Fixed confirmation checkpoint is undeclared')
    entry=entries[0];files={}
    status=_checkpoint(root,checkpoint,files,expected_model_request=entry['expected_model_request_hash'])
    run=status['run']
    if run!=entry['run'] or run['model_id']!=model or run['training_seed']!=seed or run['strict_no_val']:
        raise ValueError('Fixed confirmation model/seed request changed')
    conditions=[c for c in design['conditions'] if c['condition_id']==run['condition_id']]
    selections=read_json(root/'selections.json');choice=selections.get('choices',{}).get(f'select:{dataset}:{model}',{})
    if (len(conditions)!=1 or conditions[0]['dataset_id']!=dataset or conditions[0]['data_seed']!=2026
            or conditions[0]['experiment_id']!='E1' or conditions[0]['metadata']['size']!=6000
            or choice.get('status')!='complete' or choice.get('selected_run_id')!=checkpoint
            or choice.get('evidence_sha256')!=hash_file(root/'runs'/checkpoint/'status.json')):
        raise ValueError('Confirmation requires the actual selected E1 6k/data2026 checkpoint')
    if run['input_hash']!=meta['inputs']['input_hash']:raise ValueError('Confirmation common-input identity changed')
    assets=read_json(root/'assets/assets.lock.json')
    if assets['asset_hash']!=prepared['identity']['asset_hash'] or run['asset_hash']!=assets['asset_hash']:
        raise ValueError('Confirmation tokenizer asset identity changed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(assets['models']['distilbert']['directory'],local_files_only=True,use_fast=True)
    excluded,sources,audit_refs,discovery_files=_audit_exclusions(root,prepared,audit_frames)
    source=sources[dataset];inputs=_input_rows(root,meta,'test')
    rows=tuple(Example(dataset,r['sample_id'],r['text'],tuple(r['labels']),'test',r['group_id']) for r in source)
    if set(inputs)!={r.sample_id for r in rows}:raise ValueError('Actual Q/common-visible input ID sets differ')
    for row in rows:
        if inputs[row.sample_id]!=make_visible_input(row,tokenizer):raise ValueError('Actual source/tokenizer/common-visible input changed')
    cues=None
    if mechanism=='M1':
        ids=_train_ids(root,conditions[0]);train=_partition(root,meta,'train',['sample_id','labels'])
        lookup={r['sample_id']:r for r in train};train_inputs=_input_rows(root,meta,'train')
        if not set(ids)<=set(lookup) or not set(ids)<=set(train_inputs):raise ValueError('Anchor training source/inputs missing')
        if run['hparams'].get('role_ids_hash',{}).get('train')!=conditions[0]['training_set_hash']:
            raise ValueError('Anchor model training role identity changed')
        if any(len(lookup[sid]['labels'])!=1 for sid in ids):raise ValueError('M1 training cue labels must be single-label')
        cues=extract_cues(tuple(train_inputs[sid] for sid in ids),[lookup[sid]['labels'][0] for sid in ids],len(meta['label_names']))
    data=PreparedDataset(dataset,'single_label',tuple(meta['label_names']),{'test':rows},meta['manifest_hash'])
    candidates,rejections=confirmation_candidates(data,inputs,mechanism,excluded,cues)
    binding={'evaluation_freeze_hash':evaluation['lock_hash'],'selection_lock_hash':selection['lock_hash'],
             'design_hash':design['design_hash'],'prepared_manifest_sha256':hash_file(root/'data/prepared_manifest.json'),
             'checkpoint_run_id':checkpoint,'checkpoint_status_sha256':hash_file(root/'runs'/checkpoint/'status.json'),
             'model_request_hash':entry['expected_model_request_hash'],'n_planned':n,
             'audit_frames':audit_refs,'discovery_source_files':discovery_files,
             'tokenizer_vocab_hash':canonical_hash(tokenizer.get_vocab()),'asset_hash':assets['asset_hash'],
             'Q_source_sha256':meta['partitions']['test']['sha256'],'input_file_sha256':meta['inputs']['file_sha256'],
             'cue_map_hash':canonical_hash({str(k):v for k,v in cues.items()}) if cues is not None else None}
    return {'node':node,'meta':meta,'tokenizer':tokenizer,'candidates':candidates,'rejections':rejections,
            'binding':binding,'views':views,'status':status,'run':run,'audit_frames':audit_refs}


def prepare_confirmation(root: Path,pilot_id: str,*,audit_frames: dict) -> dict:
    """Export all remaining eligible groups and blank human review templates.

    audit_frames maps each of the four audit dataset IDs to {file,sha256}.
    Each JSON file declares dataset_id,purpose='audit',selection_policy=
    'uniform_groups_then_uniform_row',model_outcomes_used_for_sampling=False,
    sampling_seed=7301,records=[{sample_id,group_id,labels}]. Its identities must
    exactly reproduce draw_audit(...,'audit',200,7301,set()), not another valid
    200-row subset. This seed is predeclared and is never caller-configurable.
    """
    root=Path(root).resolve();context=_context(root,pilot_id,audit_frames)
    directory=_directory(root,pilot_id);path=directory/'manifest.json'
    payload={'pilot_id':pilot_id,'source_partition':'confirmation','candidates':context['candidates'],
             'rejections':context['rejections'],'candidate_order_fixed_before_model_reveal':True,
             'model_predictions_generated':False}
    if path.exists():
        previous=read_json(path);_verify_packet(root,pilot_id,context)
        return previous
    candidates_path=directory/'candidates.json'
    if candidates_path.exists():raise ValueError('Uncommitted confirmation packet exists; explicit recovery is required')
    atomic_write_json(candidates_path,payload,root)
    manifest={'schema_version':1,'status':'awaiting_actual_human_review','pilot_id':pilot_id,
              'dataset_id':context['node']['dataset_id'],'mechanism':context['node']['mechanism'],
              'binding':context['binding'],'audit_frames':context['audit_frames'],'seed':7301,
              'n_planned':context['binding']['n_planned'],'eligible_groups':len(context['candidates']),
              'underpowered':len(context['candidates'])<context['binding']['n_planned'],
              'candidates_file':candidates_path.relative_to(root).as_posix(),
              'candidates_sha256':hash_file(candidates_path),'model_predictions_in_packet':False,
              'created_at_utc':_now()}
    atomic_write_json(path,manifest,root)
    return manifest


def _verify_packet(root,pilot_id,context):
    directory=_directory(root,pilot_id);manifest=read_json(directory/'manifest.json')
    if manifest.get('pilot_id')!=pilot_id or manifest.get('binding')!=context['binding']:
        raise ValueError('Confirmation source/audit/checkpoint binding changed')
    path=bounded_path(root/manifest['candidates_file'],directory)
    if hash_file(path)!=manifest['candidates_sha256']:raise ValueError('Confirmation candidate packet bytes changed')
    payload=read_json(path)
    if (payload.get('candidates')!=context['candidates'] or payload.get('rejections')!=context['rejections']
            or payload.get('pilot_id')!=pilot_id or payload.get('source_partition')!='confirmation'
            or payload.get('model_predictions_generated') is not False):
        raise ValueError('Confirmation candidate packet does not match actual source/cues/windows')
    return manifest


def _reviews(pair):
    reviews=pair.get('reviews',[]);people=[r.get('reviewer_id') for r in reviews]
    if (len(people)<2 or len(set(people))!=len(people)
            or any(not isinstance(p,str) or not p.strip() for p in people)
            or any(r.get('blind_to_predictions') is not True for r in reviews)):
        raise PermissionError('Two actual independent blind human reviews are pending')


def _authenticate_pair(pair,candidate,context):
    template=candidate['pair_template']
    for field in ['mechanism','source_partition','sample_id','group_id','gold_labels','original_text']:
        if pair.get(field)!=template.get(field):raise ValueError('Confirmation reviewed pair source/gold/group provenance changed')
    if pair['mechanism']=='M1':
        if pair.get('cue_words')!=template.get('cue_words'):raise ValueError('Registered anchor training cue words changed')
        for role in ['intervention','control']:
            text=pair.get(f'{role}_text')
            if not isinstance(text,str):raise PermissionError('Actual human edit text is pending')
            if len(context['tokenizer'](text,add_special_tokens=False,truncation=False,verbose=False)['input_ids'])>254:
                raise ValueError('Human edit exceeds the fixed common-visible input budget')
    else:
        windows=pair.get('windows')
        if not isinstance(windows,dict):raise PermissionError('Actual human evidence windows are pending')
        source=candidate['source_full_text_normalized']
        offsets=context['tokenizer'](source,add_special_tokens=False,truncation=False,
                                      return_offsets_mapping=True,verbose=False)['offset_mapping']
        span=windows.get('evidence_span')
        if not span:raise ValueError('Human evidence span is missing')
        expected=build_evidence_windows(source,offsets,span,pair['sample_id'])
        if windows!=expected or pair.get('evidence_span') not in (None,span):
            raise ValueError('M4 windows do not match deterministic actual source/tokenizer windows')


def _select(context,reviewed_pairs):
    candidates={r['sample_id']:r for r in context['candidates']};imports={}
    for raw in reviewed_pairs:
        row=json.loads(json.dumps(raw,ensure_ascii=False,allow_nan=False));sid=row.get('sample_id')
        if sid not in candidates or sid in imports:raise ValueError('Reviewed source ID is undeclared or duplicated')
        imports[sid]=row
    selected=[];rejections=[];planned=context['binding']['n_planned']
    for candidate in context['candidates']:
        if len(selected)>=planned:break
        row=imports.get(candidate['sample_id'])
        if row is None:raise PermissionError('A preceding confirmation human decision is unreviewed/pending')
        _reviews(row)
        if row.get('decision')=='reject':
            if not isinstance(row.get('reason'),str) or not row['reason'].strip():raise ValueError('Human rejection reason is required')
            rejections.append(dict(row,candidate_order=candidate['candidate_order']));continue
        _authenticate_pair(row,candidate,context)
        try:validate_pair(row)
        except (ValueError,KeyError,TypeError) as error:
            rejections.append({'sample_id':row['sample_id'],'candidate_order':candidate['candidate_order'],
                               'decision':'invalid_pair','reason':str(error),'submitted_pair':row});continue
        selected.append({'candidate_order':candidate['candidate_order'],'pair':row,'edits_sha256':canonical_hash(row)})
    return selected,rejections


def freeze_confirmation_edits(root: Path,pilot_id: str,reviewed_pairs) -> dict:
    """Freeze all selected valid human edits before any model response is read."""
    root=Path(root).resolve();context=_context(root,pilot_id);_verify_packet(root,pilot_id,context)
    selected,rejections=_select(context,reviewed_pairs)
    directory=_directory(root,pilot_id);path=directory/'edits.freeze.json'
    if path.exists():
        previous=_verified_edits(root,pilot_id,context)
        if previous['payload']['pairs']!=selected or previous['payload']['rejections']!=rejections:
            raise ValueError('Frozen confirmation human edits/rejections are immutable and changed')
        return previous['freeze']
    if (directory/'prediction-started.json').exists():raise ValueError('Confirmation predictions already started; edits cannot change')
    edits_path=directory/'edits.json'
    if edits_path.exists():raise ValueError('Uncommitted confirmation edits exist; explicit recovery is required')
    at=_now();payload={'pilot_id':pilot_id,'source_partition':'confirmation','pairs':selected,'rejections':rejections,
                      'edits_frozen_at_utc':at,'all_selected_edits_frozen_before_prediction':True}
    atomic_write_json(edits_path,payload,root)
    frozen={'schema_version':1,'pilot_id':pilot_id,'status':'edits_frozen','binding':context['binding'],
            'edits_file':edits_path.relative_to(root).as_posix(),'edits_file_sha256':hash_file(edits_path),
            'edits_frozen_at_utc':at,'n_actual':len(selected),'n_planned':context['binding']['n_planned'],
            'underpowered':len(selected)<context['binding']['n_planned'],
            'real_human_reviews_verified':bool(selected),
            'human_pair_status':'verified' if selected else 'no_valid_pairs_pending'}
    frozen['freeze_hash']=canonical_hash(frozen);atomic_write_json(path,frozen,root)
    return frozen


def _verified_edits(root,pilot_id,context=None):
    context=_context(root,pilot_id) if context is None else context
    _verify_packet(root,pilot_id,context);directory=_directory(root,pilot_id)
    path=directory/'edits.freeze.json'
    if not path.is_file():raise PermissionError('Actual confirmation human edits are not frozen')
    frozen=read_json(path)
    if canonical_hash({k:v for k,v in frozen.items() if k!='freeze_hash'})!=frozen.get('freeze_hash'):
        raise ValueError('Confirmation edit freeze content/hash changed')
    if frozen.get('pilot_id')!=pilot_id or frozen.get('binding')!=context['binding']:
        raise ValueError('Confirmation frozen edit source/checkpoint binding changed')
    edits=bounded_path(root/frozen['edits_file'],directory)
    if hash_file(edits)!=frozen['edits_file_sha256']:raise ValueError('Frozen confirmation edits bytes changed')
    payload=read_json(edits)
    submissions=[r['pair'] for r in payload['pairs']]
    submissions.extend(r.get('submitted_pair',r) for r in payload['rejections'])
    selected,rejections=_select(context,submissions)
    if (payload.get('pilot_id')!=pilot_id or payload.get('source_partition')!='confirmation'
            or selected!=payload['pairs'] or rejections!=payload['rejections']
            or frozen['n_actual']!=len(selected) or frozen['edits_frozen_at_utc']!=payload.get('edits_frozen_at_utc')):
        raise ValueError('Frozen confirmation pair/rejection hashes/order changed')
    return {'freeze':frozen,'payload':payload,'context':context}


def _rows(pilot_id,verified,records,role):
    context=verified['context'];view=f'eval:{pilot_id}:{role}';rows=[]
    for selected,result in zip(verified['payload']['pairs'],records):
        pair=selected['pair'];value=_visible_pair_input(pair,role,context['tokenizer'])
        scores=np.asarray(result[f'{role}_scores'],dtype=float)
        if scores.shape!=(len(context['meta']['label_names']),) or not np.isfinite(scores).all():
            raise ValueError('Confirmation score columns are invalid')
        if context['status']['score_kind']=='probability' and ((scores<0).any() or (scores>1).any()):
            raise ValueError('Confirmation probability scores are invalid')
        rows.append(PredictionRow(view,pair['sample_id'],pair['group_id'],tuple(pair['gold_labels']),
                    (int(scores.argmax()),),tuple(float(x) for x in scores),context['status']['score_kind'],
                    canonical_hash(asdict(value))))
    return rows


def _save_rows(root,directory,name,rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=directory/f'{name}.parquet';temporary=path.with_suffix('.parquet.part')
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]),temporary,compression='zstd');temporary.replace(path)
    return {'file':path.relative_to(root).as_posix(),'sha256':hash_file(path),'rows':len(rows)}


def _verify_result(root,pilot_id,verified,request,result):
    if result.get('request_hash')!=request:raise ValueError('Confirmation result/edit/checkpoint request identity changed')
    directory=_directory(root,pilot_id);path=bounded_path(root/result['results_file'],directory)
    n=verified['freeze']['n_actual'];planned=verified['context']['binding']['n_planned'];underpowered=n<planned
    expected_summary={'status':'complete','pilot_id':pilot_id,'n_actual':n,'n_planned':planned,
                      'underpowered':underpowered,'mechanism_evidence_sufficient':not underpowered,
                      'checkpoint_run_id':verified['context']['binding']['checkpoint_run_id'],
                      'evaluation_freeze_hash':verified['context']['binding']['evaluation_freeze_hash']}
    if any(result.get(k)!=v for k,v in expected_summary.items()):
        raise ValueError('Confirmation result count/power/checkpoint summary changed')
    started=read_json(directory/'prediction-started.json')
    if (started.get('request_hash')!=request or started.get('edits_freeze_hash')!=verified['freeze']['freeze_hash']
            or started.get('evaluation_freeze_hash')!=verified['context']['binding']['evaluation_freeze_hash']):
        raise ValueError('Confirmation started-response/frozen-edit identity changed')
    if hash_file(path)!=result['results_sha256']:raise ValueError('Confirmation result bytes changed')
    payload=read_json(path);records=payload.get('pairs',[])
    actual=[{k:r[k] for k in ['candidate_order','pair','edits_sha256']} for r in records]
    if (actual!=verified['payload']['pairs'] or payload.get('edits_file_sha256')!=verified['freeze']['edits_file_sha256']
            or payload.get('request_hash')!=request or len(records)!=verified['freeze']['n_actual']):
        raise ValueError('Confirmation result pairs do not match frozen human edits')
    if payload['edits_frozen_at_utc']!=verified['freeze']['edits_frozen_at_utc']:
        raise ValueError('Confirmation frozen edit timestamp changed')
    if datetime.fromisoformat(payload['predicted_at_utc'])<=datetime.fromisoformat(payload['edits_frozen_at_utc']):
        raise ValueError('Confirmation responses preceded human edit freezing')
    if not (datetime.fromisoformat(payload['edits_frozen_at_utc'])<datetime.fromisoformat(started['started_at_utc'])
            <=datetime.fromisoformat(payload['predicted_at_utc'])):
        raise ValueError('Confirmation response journal timing differs from immutable human edits')
    import pyarrow.parquet as pq
    expected_views={f'eval:{pilot_id}:{r}' for r in ['intervention','control']}
    if set(result.get('views',{}))!=expected_views:raise ValueError('F6 evaluation view IDs changed')
    if ('original' in result)!=(verified['context']['node']['mechanism']=='M1'):
        raise ValueError('M1 original-response coverage changed')
    timing=payload.get('scoring_seconds_by_role',{})
    timing_roles={'intervention','control'}|({'original'} if 'original' in result else set())
    if (set(timing)!=timing_roles or any(not np.isfinite(v) or v<0 for v in timing.values())
            or result.get('checkpoint_to_scores_seconds')!=sum(timing.values())):
        raise ValueError('Confirmation scoring cost receipt changed')
    for role in ['intervention','control']+(['original'] if verified['context']['node']['mechanism']=='M1' else []):
        info=result['original'] if role=='original' else result['views'][f'eval:{pilot_id}:{role}']
        path=bounded_path(root/info['file'],directory)
        if hash_file(path)!=info['sha256']:raise ValueError('Confirmation prediction file bytes changed')
        expected=[asdict(row) for row in _rows(pilot_id,verified,records,role)]
        if canonical_hash(pq.read_table(path).to_pylist())!=canonical_hash(expected):
            raise ValueError('Confirmation prediction rows do not align with frozen pairs/responses')
        if info.get('rows')!=n:raise ValueError('Confirmation prediction view row count changed')
        if info.get('checkpoint_run_id')!=verified['context']['binding']['checkpoint_run_id']:
            raise ValueError('Confirmation checkpoint/view manifest identity changed')
        if role!='original':
            declaration=verified['context']['views'][f'eval:{pilot_id}:{role}']
            if any(info.get(k)!=declaration[k] for k in ['evaluation_id','base_group_namespace','dataset_id']):
                raise ValueError('Confirmation evaluation view/base-group namespace changed')
    if result.get('mcnemar_sensitivity')!=_mcnemar(verified,records):
        raise ValueError('Exact McNemar sensitivity does not reconstruct from paired responses')
    return result


def _mcnemar(verified,records):
    from .statistics import exact_mcnemar
    gold=[r['pair']['gold_labels'][0] for r in verified['payload']['pairs']]
    intervention=[int(np.argmax(r['intervention_scores']))==y for r,y in zip(records,gold)]
    control=[int(np.argmax(r['control_scores']))==y for r,y in zip(records,gold)]
    groups=[r['pair']['group_id'] for r in verified['payload']['pairs']]
    return exact_mcnemar(intervention,control,groups)


def _request(pilot_id,verified):
    context=verified['context']
    roles=['intervention','control']+(['original'] if context['node']['mechanism']=='M1' else [])
    inputs={role:tuple(_visible_pair_input(r['pair'],role,context['tokenizer']) for r in verified['payload']['pairs'])
            for role in roles}
    request=canonical_hash({'binding':context['binding'],'edits_freeze_hash':verified['freeze']['freeze_hash'],
                            'edited_inputs':{role:[asdict(v) for v in values] for role,values in inputs.items()}})
    return inputs,request


def verify_confirmation(root: Path,pilot_id: str) -> dict:
    """Read-only authentication of real confirmation evidence for reporting.

    Re-verifies the evaluation lock, prepared/audit/D sources, actual training
    cues/tokenizer windows, exact checkpoint request, two human reviews, frozen
    edit order, response timing, full score columns and paired parquet rows.
    Recomputes actual count and underpowering instead of trusting receipt flags.
    Does not score a model, write artifacts, or generate missing human work.
    """
    root=Path(root).resolve();verified=_verified_edits(root,pilot_id)
    path=_directory(root,pilot_id)/'result-manifest.json'
    if not path.is_file():raise PermissionError('Actual confirmation model responses are pending')
    _,request=_request(pilot_id,verified)
    return _verify_result(root,pilot_id,verified,request,read_json(path))


def verify_confirmation_audit_frame(root: Path,pilot_id: str,actual_global_frame: dict) -> dict:
    """Bind report's actual global A envelope to confirmation's exact exclusions.

    Takes the envelope {records: draw_audit-shaped records, seeds: {dataset:
    7301}, evaluation_freeze_hash, ...}. The reporting caller authenticates its
    byte reference, guide freeze and chronology. This hook independently checks
    the evaluation lock, prepared Q/D sources and confirmation-bound per-dataset
    frame bytes, reconstructs the predeclared group/row draw and compares every
    dataset/sample/group/gold identity. It reads no model responses and writes
    nothing; alternate valid A frames or seeds cannot pass.
    """
    root=Path(root).resolve();evaluation=verify_freeze('evaluation',root)
    directory=_directory(root,pilot_id);path=directory/'manifest.json'
    if not path.is_file():raise PermissionError('Confirmation audit preparation is pending')
    manifest=read_json(path);binding=manifest.get('binding',{})
    if binding.get('evaluation_freeze_hash')!=evaluation['lock_hash']:
        raise ValueError('Confirmation/global audit evaluation freeze identity changed')
    prepared_path=root/'data/prepared_manifest.json';prepared=read_json(prepared_path)
    if (not prepared.get('complete') or hash_file(prepared_path)!=binding.get('prepared_manifest_sha256')
            or canonical_hash({k:v for k,v in prepared.items() if k!='prepared_hash'})!=prepared.get('prepared_hash')):
        raise ValueError('Confirmation audit prepared source hash changed')
    _,sources,refs,discovery_files=_audit_exclusions(root,prepared,manifest.get('audit_frames'))
    if refs!=binding.get('audit_frames') or discovery_files!=binding.get('discovery_source_files'):
        raise ValueError('Confirmation-bound audit/D source identities changed')
    seeds=actual_global_frame.get('seeds',{})
    if (set(seeds)!=AUDIT_DATASETS or any(type(seed) is not int or seed!=7301 for seed in seeds.values())):
        raise ValueError('Actual global audit frame requires predeclared seed7301 for all four datasets')
    if actual_global_frame.get('evaluation_freeze_hash')!=evaluation['lock_hash']:
        raise ValueError('Actual global audit frame evaluation freeze identity changed')
    records=actual_global_frame.get('records')
    if not isinstance(records,(list,tuple)):raise ValueError('Actual global audit frame records are missing')
    actual=_audit_mapping(records)
    expected={};datasets={}
    for ds in sorted(AUDIT_DATASETS):
        fixed=_fixed_audit_draw(prepared['datasets'][ds],sources[ds]);mapping=_audit_mapping(fixed,ds)
        expected.update(mapping)
        bound=_audit_mapping(read_json(bounded_path(root/refs[ds]['file'],root))['records'],ds)
        if bound!=mapping:raise ValueError('Confirmation audit exclusions do not match the fixed draw')
        datasets[ds]={'sampling_seed':7301,'samples':len(mapping),'frame_sha256':refs[ds]['sha256'],
                      'sample_group_gold_hash':canonical_hash([
                          (dataset,sid,group,list(labels)) for (dataset,sid),(group,labels) in sorted(mapping.items())])}
    if actual!=expected:
        raise ValueError('Actual global audit A frame differs from confirmation-bound fixed sample/group/gold identities')
    return {'pilot_id':pilot_id,'binding_verified':True,'datasets':datasets,
            'evaluation_freeze_hash':evaluation['lock_hash'],
            'actual_global_frame_hash':canonical_hash(actual_global_frame),'confirmation_audit_frames':refs}


def predict_confirmation(root: Path,pilot_id: str) -> dict:
    """Score immutable reviewed inputs with the exact registry anchor; emit F6.

    This bypasses full-original-Q prediction intentionally: edited views retain
    original sample IDs, gold labels and group IDs while using their exact F6
    evaluation IDs. Saved M1 original responses form a separate auxiliary file.
    """
    root=Path(root).resolve();verified=_verified_edits(root,pilot_id)
    context=verified['context'];directory=_directory(root,pilot_id)
    if not verified['payload']['pairs']:raise PermissionError('No valid human confirmation pairs; mechanism evidence remains pending')
    inputs,request=_request(pilot_id,verified);roles=list(inputs)
    result_path=directory/'result-manifest.json'
    if result_path.exists():return _verify_result(root,pilot_id,verified,request,read_json(result_path))
    started_path=directory/'prediction-started.json'
    if started_path.exists() and read_json(started_path)['request_hash']!=request:
        raise ValueError('Confirmation inputs/checkpoint changed after responses started')
    atomic_write_json(started_path,{'request_hash':request,'edits_freeze_hash':verified['freeze']['freeze_hash'],
                                  'started_at_utc':_now(),'evaluation_freeze_hash':context['binding']['evaluation_freeze_hash']},root)
    from .train import score_saved
    output=root/'runs'/context['binding']['checkpoint_run_id'];scores={};scoring_seconds={}
    for role,visible in inputs.items():
        score_started=time.perf_counter()
        scores[role]=np.asarray(score_saved(output,visible),dtype=float)
        scoring_seconds[role]=time.perf_counter()-score_started
        if scores[role].shape!=(len(visible),len(context['meta']['label_names'])) or not np.isfinite(scores[role]).all():
            raise ValueError('Confirmation scores have invalid full-target shape/values')
    after=_verified_edits(root,pilot_id)
    if after['freeze']!=verified['freeze']:raise ValueError('Frozen confirmation edits changed during prediction')
    predicted=_now();frozen_at=verified['freeze']['edits_frozen_at_utc']
    if datetime.fromisoformat(predicted)<=datetime.fromisoformat(frozen_at):raise ValueError('Prediction must follow human edit freezing')
    records=[dict(selected,**{f'{role}_scores':values[i].tolist() for role,values in scores.items()})
             for i,selected in enumerate(verified['payload']['pairs'])]
    payload={'pilot_id':pilot_id,'request_hash':request,'pairs':records,'edits_frozen_at_utc':frozen_at,
             'predicted_at_utc':predicted,'edits_file_sha256':verified['freeze']['edits_file_sha256'],
             'checkpoint_run_id':context['binding']['checkpoint_run_id'],'scoring_seconds_by_role':scoring_seconds}
    views={};original=None
    for role in roles:
        info=_save_rows(root,directory,f'predictions-{role}',_rows(pilot_id,verified,records,role))
        info['checkpoint_run_id']=context['binding']['checkpoint_run_id']
        if role=='original':original=info
        else:
            view_id=f'eval:{pilot_id}:{role}';views[view_id]=dict(info,evaluation_id=view_id,
                base_group_namespace=f'paired:{pilot_id}',dataset_id=context['node']['dataset_id'])
    path=directory/'results.json';atomic_write_json(path,payload,root)
    result={'schema_version':1,'status':'complete','pilot_id':pilot_id,'request_hash':request,
            'evaluation_freeze_hash':context['binding']['evaluation_freeze_hash'],
            'checkpoint_run_id':context['binding']['checkpoint_run_id'],'n_actual':len(records),
            'n_planned':verified['freeze']['n_planned'],'underpowered':verified['freeze']['underpowered'],
            'mechanism_evidence_sufficient':not verified['freeze']['underpowered'],
            'mcnemar_sensitivity':_mcnemar(verified,records),
            'checkpoint_to_scores_seconds':sum(scoring_seconds.values()),
            'scoring_cost_scope':'saved_artifact_verification_load_tokenization_feature_transform_forward; excludes_global_freeze_source_io_output_write',
            'views':views,'results_file':path.relative_to(root).as_posix(),'results_sha256':hash_file(path),
            'official_test_predictions_generated':True,'conditioned_on_fixed_checkpoint':True}
    if original is not None:result['original']=original
    _verify_result(root,pilot_id,verified,request,result);atomic_write_json(result_path,result,root)
    return result

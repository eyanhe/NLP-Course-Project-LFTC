"""Predeclared paired edit/window eligibility; human judgments are never inferred."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any
import numpy as np
from scipy.stats import norm

from .features import visible_texts
from .partitions import clean_text


def english_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.casefold())


def extract_cues(train_inputs,labels,label_count: int,top_k: int=100,min_count: int=5):
    texts=visible_texts(train_inputs);y=np.asarray(labels,dtype=int)
    if y.shape!=(len(texts),) or not np.isin(y,np.arange(label_count)).all():
        raise ValueError('Cue extraction requires aligned single-label training documents')
    total=Counter();by_class=[Counter() for _ in range(label_count)]
    for text,label in zip(texts,y):
        words=set(english_words(text));total.update(words);by_class[int(label)].update(words)
    sizes=np.bincount(y,minlength=label_count);result={}
    for label in range(label_count):
        rows=[];inside=int(sizes[label]);outside=len(y)-inside
        if inside==0 or outside==0:
            result[label]=[];continue
        for word,count in total.items():
            if count<min_count:continue
            a=by_class[label][word];b=count-a
            score=math.log((a+.5)/(inside-a+.5))-math.log((b+.5)/(outside-b+.5))
            rows.append({'word':word,'log_odds':score,'training_document_frequency':count,
                         'class_document_frequency':a,'smoothing':.5})
        result[label]=sorted(rows,key=lambda row:(-row['log_odds'],row['word']))[:top_k]
    return result


def plan_pair_count(discordant: int,pilot_n: int,family_size: int=3) -> int:
    if pilot_n<1 or family_size<1 or not 0<=discordant<=pilot_n or int(discordant)!=discordant:
        raise ValueError('Invalid observed paired discordance count')
    z=float(norm.ppf(.975));p=discordant/pilot_n
    upper=(p+z*z/(2*pilot_n)+z*math.sqrt(p*(1-p)/pilot_n+z*z/(4*pilot_n**2)))/(1+z*z/pilot_n)
    planned=math.ceil((float(norm.ppf(1-.05/(2*family_size)))+float(norm.ppf(.8)))**2*upper/.1**2)
    return max(400,min(1200,planned))


def _edit_distance_and_originals(a,b):
    dp=np.zeros((len(a)+1,len(b)+1),dtype=np.int32)
    dp[:,0]=np.arange(len(a)+1);dp[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i,j]=min(dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+(a[i-1]!=b[j-1]))
    i,j=len(a),len(b);changed=[]
    while i or j:
        if i and j and dp[i,j]==dp[i-1,j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:changed.append(a[i-1])
            i-=1;j-=1
        elif i and dp[i,j]==dp[i-1,j]+1:
            changed.append(a[i-1]);i-=1
        else:j-=1
    return int(dp[-1,-1]),changed


def build_evidence_windows(text: str,offsets,evidence_span,sample_id: str,
                           budget: int=254,stride: int=32,seed: int=7301):
    if len(offsets)<320 or len(offsets)>2048:
        raise ValueError('M4 full content length must be between 320 and 2048 tokens')
    start,end=map(int,evidence_span)
    if not 0<=start<end<=len(text):raise ValueError('Evidence character span is outside the source')
    covered=[i for i,(a,b) in enumerate(offsets) if b>start and a<end]
    if not covered:raise ValueError('Evidence has no content-token span')
    lo,hi=covered[0],covered[-1]+1
    if lo<254:raise ValueError('M4 evidence must be outside the first 254-token main prefix')
    if hi-lo>budget:raise ValueError('Evidence exceeds one complete 254-token window')
    window_start=max(0,min(len(offsets)-budget,(lo+hi-budget)//2))
    def window(index):
        a=int(offsets[index][0]);b=int(offsets[index+budget-1][1])
        return {'token_start':index,'token_end':index+budget,'char_start':a,'char_end':b,'text':text[a:b]}
    candidates=[s for s in range(0,len(offsets)-budget+1,stride) if s+budget<=lo or s>=hi]
    if not candidates:raise ValueError('No valid evidence-free matched control window')
    def key(index):
        return abs(index-window_start),hashlib.sha256(f'{seed}:{sample_id}:{index}'.encode()).digest()
    control=min(candidates,key=key)
    intervention=window(window_start);matched=window(control)
    if not intervention['char_start']<=start or not intervention['char_end']>=end:
        raise ValueError('Evidence window does not contain the complete human span')
    if not (matched['char_end']<=start or matched['char_start']>=end):
        raise ValueError('Control window overlaps marked evidence')
    return {'intervention':intervention,'control':matched,'evidence_span':[start,end],
            'full_content_tokens':len(offsets),'window_budget':budget,'control_stride':stride,
            'sample_id':sample_id,'control_selected_without_model_predictions':True}


def validate_pair(pair: dict[str,Any]) -> None:
    if pair.get('mechanism') not in {'M1','M4'}:
        raise ValueError('Unknown paired intervention mechanism')
    if pair.get('source_partition') not in {'discovery','confirmation'}:
        raise ValueError('Pilot/confirmation source must be explicitly declared')
    reviews=pair.get('reviews',[])
    people=[row.get('reviewer_id') for row in reviews]
    if len(people)<2 or any(not isinstance(name,str) or not name.strip() for name in people) or len(set(people))!=len(people):
        raise ValueError('Two real independent human review records are required')
    if not all(row.get('blind_to_predictions') is True for row in reviews):
        raise ValueError('Editors and label reviewers must be blind to predictions')
    if not pair.get('sample_id') or not pair.get('group_id') or not pair.get('gold_labels'):
        raise ValueError('Missing original sample/group/gold provenance')
    if pair['mechanism']=='M1':
        source=english_words(pair['original_text'])
        intervention=english_words(pair['intervention_text']);control=english_words(pair['control_text'])
        if not 5<=len(source)<=60:raise ValueError('M1 original word count is ineligible')
        if len(intervention)!=len(source) or len(control)!=len(source):
            raise ValueError('M1 requires substitution-only word replacements; insertion/deletion is ineligible')
        a,edited=_edit_distance_and_originals(source,intervention)
        b,control_edits=_edit_distance_and_originals(source,control)
        edited=[word for word,replacement in zip(source,intervention) if word!=replacement]
        control_edits=[word for word,replacement in zip(source,control) if word!=replacement]
        if a!=len(edited) or b!=len(control_edits):
            raise ValueError('M1 requires aligned word substitutions, not insertion/deletion alignments')
        if not 1<=a<=3 or a!=b or abs(len(intervention)-len(source))>2 or abs(len(control)-len(source))>2:
            raise ValueError('Intervention and control word edit distances/length limits differ')
        cues={word.casefold() for word in pair.get('cue_words',[])}
        if not edited or not all(word in cues for word in edited):
            raise ValueError('Intervention must replace registered training cue words')
        if not control_edits or any(word in cues for word in control_edits):
            raise ValueError('Control must edit noncue words at matched distance')
        if not all(row.get('intervention_content_words_replaced') is True and
                   row.get('control_content_words_replaced') is True for row in reviews):
            raise ValueError('Both human reviews must attest content-word replacements in both arms')
        if not all(row.get('intervention_label_preserved') is True and row.get('control_label_preserved') is True
                   and row.get('control_natural') is True for row in reviews):
            raise ValueError('Human label-preservation/natural-control review is incomplete')
    else:
        windows=pair['windows'];start,end=windows['evidence_span'];a=windows['intervention'];b=windows['control']
        if not a['char_start']<=start<end<=a['char_end'] or not (b['char_end']<=start or b['char_start']>=end):
            raise ValueError('M4 intervention/control evidence visibility is invalid')
        if a['token_end']-a['token_start']!=254 or b['token_end']-b['token_start']!=254:
            raise ValueError('M4 window budget changed')
        if not all(row.get('gold_full_supported') is True and row.get('intervention_evidence_valid') is True
                   and row.get('control_no_marked_evidence') is True and isinstance(row.get('control_has_other_cues'),bool)
                   for row in reviews):
            raise ValueError('Human full-label/evidence/control review is incomplete')


def pilot_candidate_packets(data,inputs,mechanism: str,seed: int=7301,cue_map=None):
    """Export algorithmically eligible D groups before any model-output reveal.

    Naturalness, label validity and M4 evidence positions remain human work.
    Random order is fixed so valid pairs are not selected by prediction effect.
    """
    if mechanism not in {'M1','M4'}:raise ValueError('Unknown human pilot mechanism')
    groups={}
    for row in data.partitions['discovery']:
        source=clean_text(row.text);visible=inputs[row.sample_id]
        if mechanism=='M1':
            if len(row.labels)!=1 or not 5<=len(english_words(source))<=60:continue
            cues=[record['word'] if isinstance(record,dict) else str(record)
                  for record in (cue_map or {}).get(row.labels[0],[])]
            present=sorted(set(english_words(visible.text))&set(cues))
            if not present:continue
        else:
            if not 320<=visible.full_tokens<=2048:continue
            present=[]
        groups.setdefault(row.group_id,[]).append((row,source,visible,present))
    rng=np.random.default_rng(seed);group_ids=sorted(groups)
    if not group_ids:return ()
    ordered=[group_ids[int(i)] for i in rng.permutation(len(group_ids))]
    result=[]
    for ordinal,group in enumerate(ordered):
        choices=sorted(groups[group],key=lambda item:item[0].sample_id)
        row,source,visible,present=choices[int(rng.integers(len(choices)))]
        pair={'mechanism':mechanism,'sample_id':row.sample_id,'group_id':group,
              'source_partition':'discovery','original_text':visible.text if mechanism=='M1' else source,
              'gold_labels':list(row.labels),'reviews':[]}
        if mechanism=='M1':
            pair.update({'intervention_text':None,'control_text':None,'cue_words':present})
        else:
            pair.update({'windows':None,'evidence_span':None})
        result.append({'candidate_order':ordinal,'sample_id':row.sample_id,'dataset_id':data.dataset_id,
            'group_id':group,'source_full_text_normalized':source,'common_visible_text':visible.text,
            'common_visible_end_char':visible.end_char,'full_content_tokens':visible.full_tokens,
            'gold_labels':list(row.labels),'gold_names':[data.label_names[c] for c in row.labels],
            'eligibility_status':'awaiting_human_review','algorithmic_eligibility_only':True,
            'sampling_frame_eligible_groups':len(group_ids),'pair_template':pair,
            'review_templates':[{'reviewer_id':None,'blind_to_predictions':None,
                **({'intervention_label_preserved':None,'control_label_preserved':None,'control_natural':None}
                   if mechanism=='M1' else {'gold_full_supported':None,'intervention_evidence_valid':None,
                                           'control_no_marked_evidence':None,'control_has_other_cues':None})}
                 for _ in range(2)]})
        if mechanism=='M1':
            for review in result[-1]['review_templates']:
                review.update(intervention_content_words_replaced=None,control_content_words_replaced=None)
    return tuple(result)

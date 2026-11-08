"""Group-probability audit packets and explicit, unfilled human review evidence."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any
import numpy as np

from .types import PreparedDataset


def draw_audit(data: PreparedDataset,purpose: str,n: int,seed: int,
               excluded_groups: set[str]) -> tuple[dict[str,Any],...]:
    if purpose not in {'discovery','audit','confirmation'} or n<1:
        raise ValueError('Unknown audit purpose or nonpositive group budget')
    partition='discovery' if purpose=='discovery' else 'test'
    groups=defaultdict(list)
    for row in data.partitions[partition]:
        if row.group_id not in excluded_groups:groups[row.group_id].append(row)
    group_ids=sorted(groups);total=len(group_ids);take=min(n,total)
    if not total:return ()
    rng=np.random.default_rng(seed)
    chosen=[group_ids[int(i)] for i in rng.choice(total,size=take,replace=False)]
    result=[]
    for group in chosen:
        candidates=sorted(groups[group],key=lambda row:row.sample_id)
        row=candidates[int(rng.integers(len(candidates)))]
        probability=(take/total)/len(candidates)
        template={'schema_version':1,'sample_id':row.sample_id,'dataset_id':data.dataset_id,
            'group_id':group,'partition':purpose,'reviewer_id':None,'review_status':'unreviewed',
            'gold_status':'not_checked','phenomena':[],'evidence_spans':[],
            'reason_hypotheses':[],'alternative_explanations':[],'conclusion_status':'unknown',
            'inclusion_probability':probability}
        result.append({'sample_id':row.sample_id,'dataset_id':data.dataset_id,'group_id':group,
            'purpose':purpose,'text':row.text,'fields':dict(row.fields),'labels':list(row.labels),
            'label_names':list(data.label_names),'group_population_size':len(candidates),
            'sampling_frame_groups':total,'sampled_groups':take,'inclusion_probability':probability,
            'analysis_weight':1/probability,'annotation_template':template})
    return tuple(result)


def validate_reviews(records: tuple[dict[str,Any],...]) -> dict[str,Any]:
    groups=defaultdict(list)
    required={'schema_version','sample_id','dataset_id','group_id','partition','reviewer_id',
              'review_status','gold_status','phenomena','evidence_spans','reason_hypotheses',
              'alternative_explanations','conclusion_status'}
    for record in records:
        if not required<=set(record) or record['schema_version']!=1:
            raise ValueError('Human review schema fields are missing')
        state=record['review_status']
        if state not in {'unreviewed','single_review','double_review','adjudicated'}:
            raise ValueError('Unknown human review state')
        reviewer=record['reviewer_id']
        if state!='unreviewed' and (not isinstance(reviewer,str) or not reviewer.strip()):
            raise ValueError('A real nonempty reviewer identifier is required')
        if record['conclusion_status']=='intervention_supported' and not record.get('control_record_ids'):
            raise ValueError('Intervention-supported reason requires linked control evidence')
        for span in record['evidence_spans']:
            if span['start_char']<0 or span['end_char']<span['start_char']:
                raise ValueError('Invalid human evidence span')
        key=(record['sample_id'],record.get('condition_id'),record.get('anonymous_model_id'))
        groups[key].append(record)
    summaries=[]
    for key,rows in groups.items():
        reviewed=[row for row in rows if row['review_status']!='unreviewed']
        people=[row['reviewer_id'] for row in reviewed]
        if len(people)!=len(set(people)):
            raise ValueError('Two independent human reviews cannot come from the same reviewer')
        claims_double=any(row['review_status'] in {'double_review','adjudicated'} for row in rows)
        if claims_double and len(set(people))<2:
            raise ValueError('Double review requires two real reviewer records')
        judgments={(row['gold_status'],tuple(sorted(row['phenomena']))) for row in reviewed}
        summaries.append({'sample_id':key[0],'reviewer_ids':people,'independent_reviews':len(people),
            'disagreement':len(judgments)>1,'double_review_complete':len(people)>=2,
            'original_records_preserved':True})
    return {'records':len(records),'samples':len(groups),'samples_summary':summaries,
            'double_review_complete':bool(summaries) and all(row['double_review_complete'] for row in summaries)}


def phenomenon_rates(records) -> dict[str,Any]:
    names=sorted({name for row in records for name in row['phenomena']})
    if any(not math.isfinite(float(row.get('analysis_weight',1))) or row.get('analysis_weight',1)<=0 for row in records):
        raise ValueError('Invalid inverse-inclusion analysis weight')
    total_errors=sum(float(row.get('analysis_weight',1)) for row in records if row['error'])
    result={}
    for name in names:
        present=[row for row in records if name in row['phenomena']]
        count=sum(float(row.get('analysis_weight',1)) for row in present)
        error_count=sum(float(row.get('analysis_weight',1)) for row in present if row['error'])
        result[name]={'p_error_given_phenomenon':error_count/count if count else None,
            'p_phenomenon_given_error':error_count/total_errors if total_errors else None,
            'phenomenon_rows':len(present),'error_rows_with_phenomenon':sum(bool(row['error']) for row in present),
            'weighted_phenomenon_denominator':count,'weighted_error_denominator':total_errors,
            'phenomena_may_overlap':True}
    return result

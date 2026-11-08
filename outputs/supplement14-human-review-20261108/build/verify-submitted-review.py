"""Audit submitted forms and draft structure without accepting or changing judgments."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[2]
sys.path.insert(0,str(PROJECT))
from research_v3.interventions import english_words

read=json.loads((HERE/'completed-review-readonly.json').read_text(encoding='utf-8'))
source=PROJECT/'research-v3/reports/overnight/supplement-1-4/human-review'
key={r['case_id']:r for r in map(json.loads,(source/'arm-key-do-not-open-before-review.jsonl').read_text(encoding='utf-8').splitlines())}
pairs={r['sample_id']:r for r in map(json.loads,(source/'AI-draft-pairs.jsonl').read_text(encoding='utf-8').splitlines())}
register={r['sample_id']:r for r in map(json.loads,(source/'candidate-register.jsonl').read_text(encoding='utf-8').splitlines())}
criteria=['原标签与原文相符，且意图可明确判断','A 保持原文含义、标签和关键槽位',
          'B 保持原文含义、标签和关键槽位','A 自然通顺，未新增歧义','B 自然通顺，未新增歧义',
          'A 替换的是内容词','B 替换的是内容词']
for report in read['reports']:
    md=(source/f"人审{report['reviewer']}.md").read_text(encoding='utf-8')
    chunks={chunk.splitlines()[0].split(' · ')[0]:chunk for chunk in md.split('## ')[1:]}
    for row,record in zip(report['rows'],report['records']):
        assert row[1]==', '.join(register[record['sampleId']]['gold_names'])
        chunk=chunks[record['caseId']]
        expected=[next(line for line in chunk.splitlines() if line.startswith('| '+c+' |')).split('|')[2].strip() for c in criteria]
        assert expected==record['decisions'],(report['reviewer'],record['caseId'])
        note=next(line for line in chunk.splitlines() if line.startswith('备注/拒绝原因/修改建议：')).split('：',1)[1]
        assert note==record['note']
notes=[]
for case in ['M1-001','M1-002','M1-003','M1-015','M1-073','M1-075','M1-076']:
    m=key[case];p=pairs[m['sample_id']];original=english_words(p['original_text'])
    replacements={}
    for arm in ['A','B']:
        edited=english_words(p[m[arm+'_role']+'_text'])
        replacements[arm]=[(a,b) for a,b in zip(original,edited) if a!=b]
    notes.append(dict(case_id=case,original=p['original_text'],A=p[m['A_role']+'_text'],B=p[m['B_role']+'_text'],replacements=replacements))
joint=read['joint']
disagreement_cases=sorted({r['caseId'] for r in joint['cell_disagreements']})
out=dict(status='submitted_forms_audited_not_frozen',
    xlsx_markdown_decisions_and_notes_match=True,source_text_and_gold_unchanged=True,
    submitted_rows_per_form=56,judgments_per_form=392,
    both_all_yes=len(joint['both_all_yes']),both_not_all_yes=len(joint['both_not_all_yes']),
    only_one_all_yes=len(joint['only_A_all_yes'])+len(joint['only_B_all_yes']),
    cell_disagreements=len(joint['cell_disagreements']),disagreement_case_count=len(disagreement_cases),
    additional_all_yes_pairs_needed_for50=50-len(joint['both_all_yes']),
    source_human_provenance='User confirmed: 由gemini补齐的. Preserve A first3 manual annotations; remaining A and all B are Gemini-assisted pre-review.',
    user_manual_annotation_cases=['M1-001','M1-002','M1-003'],
    gemini_completed_case_records=109,
    accepted_as_real_independent_reviews=False,confirmed_two_human_valid_pairs=0,
    earlier_no_draft_candidates_still_require_explicit_human_decisions=24,
    edits_frozen=False,edited_model_predictions_generated=False,examples=notes,
    submitted_file_sha256={report['reviewer']:hashlib.sha256(Path(report['file']).read_bytes()).hexdigest() for report in read['reports']})
(HERE/'submitted-review-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))

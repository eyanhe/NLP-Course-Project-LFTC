"""Prepare AI draft edits from the predeclared D order; never manufacture reviews."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT))
from research_v3.supplemental import check_draft
from research_v3.storage import hash_file,read_json

SOURCE=PROJECT/'research-v3/training/discovery-review/Bank_M1_SVM'
DEST=Path(__file__).resolve().parent/'human-review'
DEST.mkdir(parents=True,exist_ok=True)
manifest=read_json(SOURCE/'manifest.json')
assert hash_file(SOURCE/'candidates.jsonl')==manifest['candidate_file_sha256']
assert manifest['model_predictions_in_packet'] is False
candidates=[json.loads(line) for line in (SOURCE/'candidates.jsonl').read_text(encoding='utf-8').splitlines()][:80]

# Exactly one aligned substitution per arm. These are AI proposals, not accepted human pairs.
edits={
0:('fee','surcharge','hate','dislike'),
1:('payment','transaction','card','bankcard'),
2:('own','personal','want','wish'),
3:('countries','nations','get','obtain'),
4:('auto','automatically','multiple','several'),
5:('function','feature','activate','enable'),
8:('charged','billed','randomly','arbitrarily'),
9:('additional','extra','simple','straightforward'),
10:('person','recipient','happened','occurred'),
11:('delivered','received','help','assist'),
12:('transaction','transfer','did','made'),
13:('disposable','throwaway','own','possess'),
14:('funds','money','steps','procedures'),
15:('stopped','ceased','fix','resolve'),
16:('types','kinds','service','platform'),
17:('charges','fees','using','making'),
18:('refund','reimbursement','showing','appearing'),
19:('friend','mate','made','initiated'),
20:('check','cheque','mail','post'),
21:('new','replacement','need','require'),
22:('exchanged','converted','money','funds'),
25:('entered','typed','fix','resolve'),
26:('name','surname','want','wish'),
30:('limit','cap','have','include'),
32:('cancelled','reversed','reasons','factors'),
33:('ATM','cashpoint','get','receive'),
34:('declined','rejected','Help','Assist'),
35:('want','wish','inactivate','deactivate'),
36:('activate','enable','have','possess'),
38:('deposited','submitted','received','got'),
39:('blocked','locked','putting','typing'),
40:('payment','purchase','work','succeed'),
42:('pin','passcode','create','set'),
43:('exchange','conversion','explain','describe'),
44:('process','procedure','help','assist'),
45:('cheque','check','wrong','awry'),
49:('card','bankcard','get','add'),
51:('get','obtain','wanting','hoping'),
53:('reversed','reverted','app','application'),
54:('check','cheque','wrong','awry'),
55:('transaction','payment','know','understand'),
56:('delivered','received','card','bankcard'),
57:('change','update','just','recently'),
58:('pending','unprocessed','sure','certain'),
62:('exchange','conversion','know','understand'),
63:('identity','identification','need','require'),
64:('duplicate','repeated','made','performed'),
66:('funds','money','like','prefer'),
71:('duplicate','repeated','appreciate','welcome'),
72:('incorrect','wrong','product','item'),
73:('declined','rejected','tried','attempted'),
74:('another','spare','fees','charges'),
75:('information','details','think','believe'),
76:('order','request','card','bankcard'),
77:('refund','reimbursement','purchased','bought'),
79:('rejected','declined','subscription','membership'),
}
reasons={
6:'可明确替换的非线索内容词不足，不能用功能词凑控制。',
7:'现金/取款/ATM/拒绝等内容词均为线索，缺少自然的非线索控制。',
23:'可替换的核心内容词均为线索；不改否定或功能词作为控制。',
24:'原文为PIN问题，原标签为get_physical_card；应先由真人核对标签。',
27:'prove以外主要为功能词，缺少非线索内容词控制。',
28:'原文Auto top含义及省略不够明确，且缺少自然匹配控制。',
29:'wait等替换易改变等待激活的含义，尚无自然匹配控制。',
31:'仅命中up这一粒子，未找到可保持标签的内容词线索替换。',
37:'passcode的单词替换可能改变安全凭证含义，尚无自然双臂草稿。',
41:'deposit/check均为线索，其余为功能词，无内容词控制。',
46:'id/verifying以外主要为功能词，缺少内容词控制。',
47:'verify/identity以外没有可自然对齐替换的内容词控制。',
48:'cancel/purchase均为线索，缺少非线索内容词控制。',
50:'long/wait尚未找到自然的一词线索替换，不能改成费用或主动延迟。',
52:'currency/different尚无自然对齐替换，不改变货币概念或语法。',
59:'主要内容词均在训练线索集中，缺少非线索内容词控制。',
60:'Visa/Mastercard为品牌槽位，不替换品牌或用功能词凑干预。',
61:'仅top/up命中线索，未找到符合逐词替换约束的自然干预。',
65:'app/exchange/currencies均为线索，其余为功能词。',
67:'主要内容词均为线索，缺少非线索内容词控制。',
68:'主要内容词均为线索；不改变US地域槽位。',
69:'可替换内容词均为线索，缺少自然的非线索控制。',
70:'changed疑为charged的原始笔误；修正笔误会引入流畅性混杂。',
78:'非线索内容词主要为euro；不能改变币种槽位作为控制。',
}

def substitute(text,old,new):
    result,n=re.subn(r"(?<![A-Za-z0-9'])"+re.escape(old)+r"(?![A-Za-z0-9'])",lambda _:new,text,count=1,flags=re.I)
    assert n==1,(old,text)
    return result

pairs=[];registry=[];mapping=[]
for candidate in candidates:
    i=candidate['candidate_order'];pair=copy.deepcopy(candidate['pair_template'])
    row=dict(candidate_order=i,sample_id=pair['sample_id'],gold_names=candidate['gold_names'],
             source_partition='discovery',human_eligibility='pending')
    if i not in edits:
        row.update(draft_status='no_safe_draft',reason=reasons[i]);registry.append(row);continue
    cue,newcue,noncue,newnoncue=edits[i]
    pair['intervention_text']=substitute(pair['original_text'],cue,newcue)
    pair['control_text']=substitute(pair['original_text'],noncue,newnoncue)
    structural=check_draft(pair)
    pair['reviews']=[]
    swapped=bool(int(hashlib.sha256(pair['sample_id'].encode()).hexdigest(),16)%2)
    a='control' if swapped else 'intervention';b='intervention' if swapped else 'control'
    case=f'M1-{i+1:03d}'
    pairs.append(pair)
    mapping.append(dict(case_id=case,candidate_order=i,sample_id=pair['sample_id'],
                        A_role=a,B_role=b,structural_check=structural))
    row.update(case_id=case,draft_status='AI_draft_pending_two_humans',reason=None)
    registry.append(row)

for name,rows in [('AI-draft-pairs.jsonl',pairs),('candidate-register.jsonl',registry),('arm-key-do-not-open-before-review.jsonl',mapping)]:
    (DEST/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

byid={p['sample_id']:p for p in pairs};labels={c['sample_id']:c['gold_names'] for c in candidates}
for reviewer in ['A','B']:
    lines=[f'# M1 独立人审表 {reviewer}', '',
        f'共 {len(pairs)} 对 AI 编辑草稿，来自原预注册候选顺序的前80条。实际完成真人复核：0。目标为50个双人同意的有效对；当前不保证达到。', '',
        '复核者真实姓名/代号：待填；日期：待填；复核时是否见过这些样本的模型预测：待填。', '',
        '请独立填写本文件，不看另一位复核者的表、模型预测或 arm-key。A/B文本顺序按样本固定打乱，未标出哪一臂是线索干预。标签是数据集原标签，也需要你核对。', '',
        '每项请将“待填”改为“是 / 否 / 不确定”。未填写和不确定均不视为通过。两种编辑都应保持原文含义和原标签，不改变动作、对象、否定、数额、币种及其他关键槽位；两臂都应自然，替换应是内容词。', '',
        '如果需要修改文本，请在备注给出建议，不直接改表中的原始草稿。修改后须重新确认两臂，先冻结最终文本再进行模型预测。完成后把此文件发回即可。', '',
        '本文本中的大小写、拼写和原始语法问题保留；它们也可能令候选不适合实验，请据实拒绝。', '']
    for m in mapping:
        pair=byid[m['sample_id']]
        lines.extend([f"## {m['case_id']} · {m['sample_id']}", '',f"原标签：{', '.join(labels[m['sample_id']])}", '',
             '原文：'+pair['original_text'],'','文本 A：'+pair[m['A_role']+'_text'],'','文本 B：'+pair[m['B_role']+'_text'],'',
             '| 复核项目 | 判断 |','| --- | --- |',
             '| 原标签与原文相符，且意图可明确判断 | 待填 |',
             '| A 保持原文含义、标签和关键槽位 | 待填 |',
             '| B 保持原文含义、标签和关键槽位 | 待填 |',
             '| A 自然通顺，未新增歧义 | 待填 |',
             '| B 自然通顺，未新增歧义 | 待填 |',
             '| A 替换的是内容词 | 待填 |',
             '| B 替换的是内容词 | 待填 |',
             '','备注/拒绝原因/修改建议：待填',''])
    (DEST/f'人审{reviewer}.md').write_text('\n'.join(lines),encoding='utf-8')

summary=dict(source_candidate_sha256=manifest['candidate_file_sha256'],source_manifest_sha256=hash_file(SOURCE/'manifest.json'),
             fixed_source_candidates=80,AI_draft_pairs=len(pairs),no_safe_draft=80-len(pairs),
             human_reviews_completed=0,human_accepted_pairs=0,model_predictions_in_review_material=False,
             edit_pair_structure_checked=True,label_preservation_verified=False,
             note='AI draftability exclusions are proposals pending human judgment, not adjudicated eligibility. No model scoring until two real independent reviews and text freeze.')
(DEST/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))

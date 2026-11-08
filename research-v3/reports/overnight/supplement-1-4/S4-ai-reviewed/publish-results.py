"""Independently reconcile saved predictions and draw the paired intervention contrast."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[4]
sys.path.insert(0,str(PROJECT))
from research_v3.ai_reviewed_m1 import summarize_contrast
from research_v3.config import canonical_hash
from research_v3.storage import hash_file,read_json

frozen=read_json(HERE/'inputs-frozen.json');manifest=read_json(HERE/'result-manifest.json');result=read_json(HERE/'results.json')
assert hash_file(HERE/'results.json')==manifest['results_sha256']
assert frozen['freeze_hash']==canonical_hash({k:v for k,v in frozen.items() if k!='freeze_hash'})==result['freeze_hash']==manifest['freeze_hash']
assert len(frozen['pairs'])==37 and result['formal_human_pilot'] is False
source={r['case_id']:r['pair'] for r in frozen['pairs']}
for model in result['models']:
    rows=model['pairs'];assert len(rows)==37
    for row in rows:
        p=source[row['case_id']]
        assert row['sample_id']==p['sample_id'] and row['group_id']==p['group_id'] and row['gold_label']==p['gold_labels'][0]
        for role in ['original','cue','control']:
            assert row[role]['prediction']==int(np.argmax(row[role]['scores']))
            assert row[role]['correct']==(row[role]['prediction']==row['gold_label'])
    y=np.asarray([r['gold_label'] for r in rows]);values=[np.asarray([r[k]['scores'] for r in rows]) for k in ['original','cue','control']]
    assert summarize_contrast(y,*values)==model['summary']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(11.6,5.1),dpi=180,sharey=True)
for ax,model in zip(axes,result['models']):
    s=model['summary'];denom=s['introduced_error_denominator']
    nums=[s['cue_introduced_errors'],s['control_introduced_errors']]
    rates=[s['cue_introduced_error_rate'],s['control_introduced_error_rate']]
    ax.bar([0,1],rates,width=.52,color=['#BB4A49','#4C759B'])
    for x,n,r in zip([0,1],nums,rates):
        ax.text(x,r+.016,f'{n}/{denom} ({r:.1%})',ha='center',fontsize=13,color='#26374D')
    label='SVM' if model['model_id']=='tfidf_svm' else 'DistilBERT'
    ax.set_title(f'{label}：原文答对 {denom} 条',fontproperties=font,fontsize=15,pad=15)
    ax.set_xticks([0,1],['替换线索词','替换非线索词\n（匹配控制）'],fontproperties=font,fontsize=12)
    ax.set_ylim(0,.4);ax.set_yticks([0,.1,.2,.3,.4]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#BEC7D3')
axes[0].set_ylabel('原文答对后，被改错的比例',fontproperties=font,fontsize=12)
fig.suptitle('替换训练线索词，比匹配控制更容易引入错误吗？',fontproperties=font,fontsize=18,y=.97)
fig.text(.5,.875,'BANKING77 · 固定37个基例 · 原6,000条训练模型',ha='center',fontproperties=font,fontsize=12,color='#66758A')
fig.text(.07,.095,'审核来源：Gemini，探索性D子集。分母是各模型原文答对的样本数。',fontproperties=font,fontsize=10,color='#66758A')
fig.text(.07,.048,'两模型的条件样本集合不同；本图描述各自干预对照，不作模型能力上限结论。',fontproperties=font,fontsize=10,color='#66758A')
fig.subplots_adjust(top=.75,bottom=.25,wspace=.16,left=.1,right=.97)
fig.savefig(HERE/'线索替换与匹配控制.png',facecolor='white');plt.close(fig)
print(json.dumps(dict(status='verified',pairs=37,models=len(result['models']),job_id=manifest['job_id'],
      result_sha256=manifest['results_sha256'],chart='线索替换与匹配控制.png'),ensure_ascii=False))

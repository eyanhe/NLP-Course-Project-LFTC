"""Research-question figures from verified saved results; no training or SSH."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

FOLDER=Path(__file__).resolve().parent
OUT=FOLDER/'charts-research'
OUT.mkdir(exist_ok=True)
status=json.loads((FOLDER/'research-status.json').read_text(encoding='utf-8'))
with (FOLDER/'charts-v1/all-completed-records.csv').open(encoding='utf-8-sig',newline='') as f:
    records=list(csv.DictReader(f))
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':12,'axes.titlesize':15,'axes.titleweight':'bold','axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})
DATASETS=('yelp','go_emotions','banking77','yahoo_answers_topics')
NAMES={'yelp':'情感分类 · Yelp','go_emotions':'多标签情绪 · GoEmotions',
    'banking77':'意图分类 · BANKING77','yahoo_answers_topics':'主题分类 · Yahoo'}
BLUE,GREEN,ORANGE='#286DA8','#168B78','#DC9B30'
BOTTOM='已保存的探索性结果；尚未完成正式Q评估。图中不使用未完成训练的结果。'
figures=[]
pdf=PdfPages(OUT/'能力边界与失败机制-研究主图.pdf')
def heading(fig,title,subtitle,footer=BOTTOM):
    fig.suptitle(title+'\n'+subtitle,fontsize=19,linespacing=1.65)
    fig.supxlabel(footer,fontsize=10,color='#596B7B')
def save(fig,name,description):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight')
    pdf.savefig(fig,bbox_inches='tight')
    figures.append({'file':name+'.png','question':description})
    plt.close(fig)

# A: The research contrast itself changes with the data condition.
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for ax,ds in zip(axes.flat,DATASETS):
    rows=sorted((r for r in status['boundary_point_estimates'] if r['dataset']==ds),key=lambda r:r['n'])
    assert [r['n'] for r in rows]==[1000,6000]
    values=[100*r['delta_distilbert_minus_svm'] for r in rows]
    ax.axhspan(0,25,color=GREEN,alpha=.045)
    ax.axhspan(-53,0,color=BLUE,alpha=.045)
    ax.axhline(0,color='#647584',linestyle='--',linewidth=1.2)
    ax.plot([0,1],values,color='#405A74',linewidth=2.7,marker='o',markersize=10)
    for x,v in enumerate(values):
        ax.annotate(f'{v:+.2f}点',(x,v),xytext=(0,13 if v>=0 else -22),textcoords='offset points',
            ha='center',weight='bold',fontsize=14,color=GREEN if v>=0 else BLUE)
    ax.set_xticks([0,1],['1,000条训练文本','6,000条训练文本'])
    ax.set_xlim(-.32,1.32)
    ax.set_ylim(-55,28)
    ax.set_yticks([-50,-25,0,25])
    ax.set_ylabel('DistilBERT − SVM（百分点）')
    ax.set_title(NAMES[ds]+'\n'+('比较Macro-AP差值' if ds=='go_emotions' else '比较Macro-F1差值'),loc='left',pad=12)
    ax.grid(axis='y',alpha=.12)
heading(fig,'① 固定训练配方下：训练数据增多，模型差距怎样变化？',
    'DistilBERT固定5轮；SVM按求解器收敛准则结束。零线上方为DistilBERT较高。',
    '尚未建立神经模型充分优化证据；差值包含训练预算影响，不能直接认定模型能力边界或稳定反转。')
save(fig,'01-样本量改变模型差距','样本量从1,000到6,000时，DistilBERT相对SVM的差距如何变化')

# B: Absolute degradation, with the low-baseline trap disclosed.
models=('tfidf_svm','bilstm','distilbert')
model_names={'tfidf_svm':'SVM','bilstm':'BiLSTM','distilbert':'DistilBERT'}
colors={'tfidf_svm':BLUE,'bilstm':ORANGE,'distilbert':GREEN}
fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
for ax,ds in zip(axes,('yelp','banking77','yahoo_answers_topics')):
    pairs=[next(r for r in status['core_imbalance_pairs'] if r['dataset']==ds and r['model']==m) for m in models]
    for y,r in enumerate(pairs):
        ax.barh(y,r['drop_percentage_points'],color=colors[r['model']],height=.5)
        ax.text(r['drop_percentage_points']+.5,y,f"下降 {r['drop_percentage_points']:.2f} 点",va='center',fontsize=12)
    ax.set_yticks(range(3),['SVM','BiLSTM *' if ds=='banking77' else 'BiLSTM','DistilBERT'])
    ax.invert_yaxis()
    ax.set_xlim(0,33)
    ax.set_xlabel('Macro-F1下降（百分点，越少越好）',fontsize=10)
    ratio='约8:1' if ds=='banking77' else '约10:1'
    ax.set_title(NAMES[ds]+f"\n总训练量 {pairs[0]['n']:,}；均衡 → {ratio}",loc='left',fontsize=13,pad=16)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
heading(fig,'② 固定训练配方下：不平衡条件带来多大退化？',
    '每个任务内：训练总量与验证集不变，比较各模型相对自身均衡基线的降幅。',
    '* BiLSTM均衡基线仅0.0132，小降幅不能称稳健；尚未排除优化不足，也未分离标注支持与目标偏置。')
save(fig,'02-分布倾斜后的退化','固定每个任务的训练总量，比较均衡与倾斜训练的退化幅度')

# C: Individual error transitions, directly counted on matched D inputs.
fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
cell_colors=np.array([[[.96,.86,.86],[.83,.94,.87]],[[1.,.90,.73],[.90,.93,.96]]])
for ax,ds in zip(axes.flat,DATASETS):
    r=next(r for r in status['discovery_transitions'] if r['dataset']==ds and r['before']=='N1000' and r['after']=='N6000')
    cells=((('持续错误',r['wrong_both']),('被纠正',r['corrected'])),
           (('新增错误',r['new_error']),('持续正确',r['correct_both'])))
    assert sum(v for row in cells for _,v in row)==r['paired_D_rows']
    ax.imshow(cell_colors,extent=(-.5,1.5,1.5,-.5),aspect='auto')
    for i,row in enumerate(cells):
        for j,(text,count) in enumerate(row):
            ax.text(j,i,f'{text}\n{count} 条',ha='center',va='center',fontsize=19,weight='bold',linespacing=1.8,color='#233B4F')
    ax.set_xticks([0,1],['仍预测错误','预测正确'])
    ax.set_yticks([0,1],['原来预测错误','原来预测正确'])
    ax.set_xlabel('增加到6,000条训练文本后',fontsize=11)
    ax.set_ylabel('只有1,000条训练文本时',fontsize=11)
    ax.set_title(NAMES[ds]+f" · 同一批D样本 n={r['paired_D_rows']}",loc='left',fontsize=14,pad=12)
    ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True)
    ax.grid(which='minor',color='white',linewidth=5)
    ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
heading(fig,'③ 失败轨迹：增加数据以后，具体错误被修复了吗？',
    '模型固定为SVM；每个格子是同一样本从1,000条训练到6,000条训练后的状态。',
    '条件化D样本框，不能估计总体错误率；BANKING77仅23条。GoEmotions以完整标签集合匹配判断对错，非Macro-AP。')
save(fig,'03-同一样本的失败轨迹','在同一批D样本上统计持续错误、纠错、新增错误和持续正确')

# D: A completed factorial mechanism comparison, limited to its own recipe.
arms=(('pretrained_frozen','pretrained_unfrozen'),('random_frozen','random_unfrozen'))
means=np.zeros((2,2));ranges={}
for i,row in enumerate(arms):
    for j,arm in enumerate(row):
        values=[float(r['score']) for r in records if r['arm']==arm]
        assert len(values)==9
        means[i,j]=np.mean(values);ranges[i,j]=(min(values),max(values))
fig,ax=plt.subplots(figsize=(12,6.5),layout='constrained')
ax.imshow(means,cmap='GnBu',vmin=0,vmax=.9,aspect='auto')
for i in range(2):
    for j in range(2):
        low,high=ranges[i,j]
        color='white' if means[i,j]>.6 else '#17394F'
        ax.text(j,i,f'{means[i,j]:.4f}\n9次平均\n范围 {low:.3f}–{high:.3f}',ha='center',va='center',
            fontsize=17,weight='bold',color=color,linespacing=1.8)
ax.set_xticks([0,1],['只训练分类头\n编码器冻结','分类头＋编码器都训练\n编码器解冻'],fontsize=13)
ax.set_yticks([0,1],['使用预训练权重','随机初始化\n无预训练权重'],fontsize=13)
ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True)
ax.grid(which='minor',color='white',linewidth=6)
ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():spine.set_visible(False)
heading(fig,'④ 给定训练预算下：预训练 × 编码器适配对照',
    'BANKING77，1,000条训练文本；同架构、对齐分类头初始化；每格3个数据种子×3个训练种子。',
    '数值为V Macro-F1，越高越好；范围不是置信区间。只支持这套M2训练配方，不能替代主学习曲线或完整失败原因。')
save(fig,'04-预训练与适配四臂机制','同架构四臂，分别比较初始化与编码器适配')
audit_figure=OUT/'05-训练预算与验证曲线.png'
if audit_figure.exists():
    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    ax.imshow(plt.imread(audit_figure));ax.axis('off')
    pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    figures.append({'file':audit_figure.name,'question':'充分训练尚未确认：验证曲线是否仍受预算上限限制'})
pdf.close()
manifest={'figures':figures,'source_snapshot_utc':status['last_verified_cluster_snapshot_utc'],
    'scope':'matched_data_conditions_error_transitions_and_local_mechanism_evidence',
    'new_training_started':False,'official_Q_generated':False,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (FOLDER/'research-status.json',FOLDER/'charts-v1/all-completed-records.csv',FOLDER/'failure-transition-summary.csv',FOLDER/'convergence-audit.json')},
    'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.png','.pdf')}}
(OUT/'plot-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
guide=['# 数据条件、失败轨迹及训练预算核查','',
    '所有图只使用已核验数据。前两张为固定训练配方下的观察；第三张追踪个体错误；第四张展示给定预算下的机制对照；第五张核查验证曲线。',
    '神经模型充分优化尚未确认，当前结果不能直接认定模型能力边界；也没有正式Q结果。','']
for r in figures:
    guide.extend(['## '+r['question'],'',f"![{r['question']}]({r['file']})",''])
guide.extend(['[研究主图PDF](能力边界与失败机制-研究主图.pdf)','',
    '[研究问题、实际实验与证据缺口](../results-analysis-v1-zh.md)',''])
(OUT/'README.md').write_text('\n'.join(guide),encoding='utf-8')
print(json.dumps({'figures':len(figures),'boundary_points':8,'imbalance_pairs':9,
    'paired_D_samples':1276,'mechanism_runs':36,'output':str(OUT)},ensure_ascii=False))

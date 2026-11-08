"""Draw the research argument from saved evidence; no training or inference."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[3]
REPORT = PROJECT/'research-v3/reports/overnight'
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyhbd.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.unicode_minus':False})


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


sources = [REPORT/'convergence-v1/experiment-inventory.json',
    REPORT/'convergence-v1/cluster-snapshot/reports/overnight/convergence-v1/test-results.json',
    REPORT/'supplement-1-4/cluster-snapshot/S2-summary.json',
    REPORT/'supplement-1-4/S4-ai-reviewed/results.json',
    REPORT/'supplement-1-4/manifest.json',
    REPORT/'results-analysis-v1-zh.md']
inventory, tests, trajectories, edits, supplement = [read(p) for p in sources[:5]]
assert tests['status']=='complete' and len(tests['models'])==18
q = {}
for dataset in ['banking77','go_emotions']:
    metric = 'macro_f1' if dataset=='banking77' else 'macro_ap'
    q[dataset] = {model:statistics.mean(m['metrics'][metric] for m in tests['models']
        if m['dataset_id']==dataset and m['model_id']==model and m['role']=='primary')
        for model in ['tfidf_svm','bilstm','distilbert']}
    assert q[dataset]['distilbert']>q[dataset]['tfidf_svm']>q[dataset]['bilstm']
support = {}
for item in supplement['fits']:
    if item['supplement_id']!='S3' or item['model_id']!='fixed_tfidf_svm':continue
    hp=item['entry']['run']['hparams']; weights=hp['target_q']
    if max(weights)-min(weights)>1e-12:continue
    path=REPORT/'supplement-1-4/cluster-snapshot/runs'/item['entry']['run_id']/'status.json'
    status=read(path);sources.append(path)
    support[status['training_unique_rows']]=status['validation']['macro_f1']
assert set(support)=={781,555} and support[781]>support[555]
analysis_path=REPORT/'analysis-v2/analysis-data.json';analysis=read(analysis_path);sources.append(analysis_path)
support_q={r['train_rows']:r['macro_f1'] for r in analysis['q_records']
    if r['stage']=='S3' and r['model']=='fixed_tfidf_svm' and r['risk']=='uniform'}
assert set(support_q)=={781,555}
m2={}
for row in inventory['rows']:
    if row['model']=='m2_distilbert' and row['dataset']=='banking77' and row['train_rows']==1000:
        status_path=PROJECT/row['status_source']
        hp=read(status_path)['run']['hparams']
        if abs(hp['encoder_lr']-.00002)>1e-12:continue
        sources.append(status_path)
        m2.setdefault(row['arm'],[]).append(row['V_macro_f1'])
m2={arm:{'n':len(values),'mean':statistics.mean(values)} for arm,values in m2.items()}
assert all(m2[a]['n']==9 for a in ['pretrained_unfrozen','pretrained_frozen','random_unfrozen','random_frozen'])
assert m2['pretrained_unfrozen']['mean']>max(m2[a]['mean'] for a in
    ['pretrained_frozen','random_unfrozen','random_frozen'])
drops={m['model_id']:-100*m['summary']['cue_minus_control_accuracy'] for m in edits['models']}
assert drops['tfidf_svm']>drops['distilbert']

fig=plt.figure(figsize=(22.6,13.8),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,22.6);ax.set_ylim(0,13.8);ax.axis('off')
NAVY='#17365D'; INK='#203448'; MUTED='#566B7E'; LINE='#BDCAD5'
STAGES=['#FFF3DB','#F1EBF8','#E7F0FA','#E9F3ED']
checks=[]


def text(x,y,value,size=12,color=INK,weight='normal',ha='left',va='center'):
    return ax.text(x,y,value,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.55)


def box(x,y,w,h,value,fill='white',size=12.2,border=LINE,weight='normal',align='left',color=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.055',
                 linewidth=.85,edgecolor=border,facecolor=fill))
    tx=x+.19 if align=='left' else x+w/2
    t=text(tx,y+h/2,value,size,color,weight,ha=align)
    checks.append((t,(x+.12,y+.07,x+w-.12,y+h-.07)))
    return t


def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
                                color='#74899B',linewidth=1.25))


text(.55,13.24,'文本分类模型在不同数据条件下的能力边界与失败机制',25,NAVY,'bold')
text(.57,12.72,'承接交接前常规分类、MultiNLI与AG News长尾比较：从条件性优势的发现，推进到规模、优化与机制解释。',13.5,MUTED)
box(.55,11.66,10.55,.72,'主问题一 · 能力边界：什么数据条件下，模型占优、退化或持续出错？',
    '#F0F4F8',13.3,weight='bold')
box(11.48,11.66,10.55,.72,'主问题二 · 失败机制：优化、表征、标注支持与词汇依赖如何影响失败？',
    '#F0F4F8',13.3,weight='bold')
arrow(6.1,11.59,6.1,11.29);arrow(16.7,11.59,16.7,11.29)
box(.55,10.5,21.48,.72,'统一起点：在同一数据条件内，比较 TF-IDF + SVM、BiLSTM、DistilBERT；再观察数据量、类别分布与输入表达改变后的表现。',
    '#F6F8FA',13.3,align='center')
text(.56,10.04,'以下五条是并列的研究分支；每一行从左向右阅读。',12.3,MUTED)

xs=[2.05,6.25,10.37,16.02];ws=[3.72,3.64,5.17,6.01]
headers=['① 从比较中发现的现象','② 由现象追问的子问题','③ 对应的实验与证据','④ 目前得到的解释']
for x,w,label,fill in zip(xs,ws,headers,STAGES):
    box(x,9.34,w,.46,label,fill,12.8,border=fill,weight='bold',align='center')

rows=[
    ('数据规模',
     '从1k增至6k后，\n部分错误得到纠正，\n也存在持续错误与新增错误。',
     '哪些错误可以靠加数据缓解？\n哪些错误需要进一步解释？',
     'E1 规模对照 ＋ S2 错误轨迹\n四任务、三模型、24个模型的D预测\n形成12组同样本1k→6k轨迹。',
     '数据规模效应具有选择性：加数据能纠错，\n也可能引入新错。持续错误已定位；\n具体语义失败模式仍待系统归纳。',
     'D轨迹已完成 · 语义归因待补'),
    ('优化配方',
     '早期一些条件下，\nSVM分数超过神经模型；\n延长训练后部分排序改变。',
     '是模型能力差异，还是\n训练预算、学习率和调度\n影响了比较结果？',
     'S1 学习率×预算 ＋ 新配方网格\n动态早停、最优配方3个种子\n18个最终/参考模型完成独立Q测试。',
     f"优化配方能改变部分早期排序。两已测条件\nQ均为 DistilBERT ＞ SVM ＞ BiLSTM。\nBank F1：{q['banking77']['distilbert']:.3f} / {q['banking77']['tfidf_svm']:.3f} / {q['banking77']['bilstm']:.3f}；Go AP：{q['go_emotions']['distilbert']:.3f} / {q['go_emotions']['tfidf_svm']:.3f} / {q['go_emotions']['bilstm']:.3f}。",
     'S1与新配方均已有Q结果'),
    ('预训练表征',
     '预训练模型与从头训练模型\n在低数据条件下，\n表现存在明显差距。',
     '优势来自预训练表征，\n还是来自编码器在\n下游任务中的适配？',
     'M2 同架构、同分类头初始化\n预训练/随机初始化 × 冻结/解冻\nBANKING77 1k四臂重复对照。',
     '本配方下，初始化与编码器适配都影响结果；\n预训练＋解冻表现最好。\n这为低数据表现差异提供局部表征解释。',
     'V上的受控证据 · 未做此分支Q'),
    ('长尾与支持',
     '类别分布倾斜后，\n不同模型的退化幅度不同，\n这种相对优势还随任务改变。',
     '尾类失败是因为独立标注少，\n还是因为训练中\n尾类的贡献较小？',
     'E3 长尾对照 ＋ S3 固定表示干预\n标注支持完整/删减 × 贡献均匀/偏斜\n两种固定表示，共8格。',
     '固定TF-IDF+SVM、贡献均匀时，删减尾类标注\n'+f'使Q Macro-F1从{support_q[781]:.3f}降至{support_q[555]:.3f}；\n尾类召回65.6%→10.4%，本组均匀贡献不能替代标注支持。',
     'S3独立Q已完成8格'),
    ('词汇变化',
     '原文分类表现较好，\n仍可能因词汇表达变化\n而发生预测翻转。',
     '模型是否依赖表层线索？\n线索词替换是否比普通\n词汇替换更容易造成失败？',
     'S4 / M1 匹配词汇干预\n37对Gemini审核样本；两个固定模型\n原文 / 线索词替换 / 控制词替换。',
     '这组样本中，DistilBERT对线索替换更稳定。\n相对控制替换，准确率降幅：\n'+f"SVM {drops['tfidf_svm']:.1f}个百分点，DistilBERT {drops['distilbert']:.1f}个百分点。",
     'AI审核样本上的配对干预证据'),
]
row_h=1.47;gap=.18;top=9.08
for i,(topic,phenomenon,question,experiment,explanation,scope) in enumerate(rows):
    y=top-row_h-i*(row_h+gap)
    text(.6,y+row_h*.59,topic,14,NAVY,'bold')
    for j,value in enumerate([phenomenon,question,experiment,explanation]):
        box(xs[j],y,ws[j],row_h,value,STAGES[j],12.1,border='#D8E1E8')
        if j<3:arrow(xs[j]+ws[j]+.055,y+row_h/2,xs[j+1]-.055,y+row_h/2)
    text(xs[3]+.19,y+.14,scope,9.6,MUTED)

text(.6,.48,'证据口径：V＝验证集；D＝探索样本；Q＝正式测试集。Q中的神经模型分数为3个种子均值，SVM为确定性单次结果。',10.5,MUTED)
text(.6,.19,'这些分支共同回答“优势何时改变、失败与哪些因素有关”；各分支结论限于已测任务、数据条件和配方，不能跨任务拼成统一因果解释。',10.5,MUTED)

fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t,(x1,y1,x2,y2) in checks:
    bounds=t.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert bounds.x0>=x1 and bounds.x1<=x2 and bounds.y0>=y1 and bounds.y1<=y2, t.get_text()
base=FOLDER/'当前研究问题与实验逻辑链'
for suffix in ['png','svg','pdf']:
    fig.savefig(base.with_suffix('.'+suffix),dpi=150,facecolor='white',metadata={'Title':'当前研究问题与实验逻辑链'} if suffix=='pdf' else None)
sources += [PROJECT/'HANDOFF.md',PROJECT/'docs/RESEARCH-CONTINUITY.md']
evidence={'created_at_sgt':datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat(),
    'Q_primary_means':q,'S3_uniform_contribution_V':support,'S3_uniform_contribution_Q':support_q,'M2_Banking77_1k_V':m2,
    'S4_control_minus_cue_accuracy_percentage_points':drops,
    'source_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(FOLDER/'figure-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figure':str(base),'bounds_checked':len(checks),'Q_primary_means':q},ensure_ascii=False))

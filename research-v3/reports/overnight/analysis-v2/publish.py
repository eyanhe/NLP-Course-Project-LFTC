"""Publish the reconstructed research analysis and three focused Q figures."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

FOLDER=Path(__file__).resolve().parent
data=json.loads((FOLDER/'analysis-data.json').read_text(encoding='utf-8'))
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','pdf.fonttype':42,'svg.fonttype':'none',
    'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
names={'tfidf_svm':'SVM','bilstm':'BiLSTM','distilbert':'DistilBERT',
    'fixed_tfidf_svm':'固定TF-IDF + SVM','frozen_generic_distilbert_head':'冻结编码器 + 分类头'}
datasets={'banking77':'BANKING77','go_emotions':'GoEmotions'}
colors=['#3378A8','#D28A37','#348369']
def save(fig,pdf,name):
    fig.savefig(FOLDER/(name+'.png'),dpi=160,facecolor='white')
    pdf.savefig(fig,facecolor='white');plt.close(fig)
def heading(fig,title,subtitle):
    fig.suptitle(title,x=.07,y=.965,ha='left',fontsize=17,weight='bold',color='#17365D')
    fig.text(.07,.905,subtitle,fontsize=10.5,color='#536779')

with PdfPages(FOLDER/'Q测试与机制对照分析.pdf') as pdf:
    fig,axes=plt.subplots(1,2,figsize=(12.6,6.2));fig.subplots_adjust(left=.08,right=.97,bottom=.18,top=.76,wspace=.3)
    heading(fig,'优化后的模型优势：两组正式Q测试','BANKING77训练1,000条；GoEmotions训练6,000条。各任务使用自己的主指标。')
    for ax,ds in zip(axes,['banking77','go_emotions']):
        rows=[r for r in data['primary'] if r['dataset']==ds]
        x=np.arange(3);means=[r['mean'] for r in rows]
        ax.bar(x,means,color=colors,width=.55,yerr=[r['sd'] or 0 for r in rows],capsize=4)
        for i,row in enumerate(rows):
            vals=[r['score'] for r in data['q_records'] if r['run_id'] in row['run_ids']]
            ax.scatter(np.linspace(i-.12,i+.12,len(vals)) if len(vals)>1 else [i],vals,color='#25394B',s=22,zorder=4)
            ax.text(i,row['mean']+(row['sd'] or 0)+.028,f"{row['mean']:.4f}",ha='center',fontsize=12)
        ax.set_xticks(x,[names[r['model']] for r in rows]);ax.set_ylim(0,1 if ds=='banking77' else .55)
        ax.set_title(datasets[ds]+('，Q 3,079条' if ds=='banking77' else '，Q 5,401条'))
        ax.set_ylabel('Macro-F1' if ds=='banking77' else 'Macro-AP');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.08,.07,'柱为神经模型3种子均值，误差线为训练种子样本SD，散点为各种子；SVM为确定性单次。SD不是测试置信区间。',fontsize=9.6,color='#536779')
    save(fig,pdf,'01-优化后Q模型比较')

    fig,axes=plt.subplots(2,2,figsize=(12.6,8.8));fig.subplots_adjust(left=.09,right=.96,bottom=.13,top=.79,hspace=.5,wspace=.3)
    heading(fig,'S1：训练预算改善可以迁移到Q，但并非单调','每幅图比较同一任务、同一模型内的两档学习率。子图y轴范围各自标示。')
    for ax,(ds,model) in zip(axes.ravel(),[(d,m) for d in ['banking77','go_emotions'] for m in ['bilstm','distilbert']]):
        rows=[r for r in data['q_records'] if r['stage']=='S1' and r['dataset']==ds and r['model']==model]
        lrs=sorted({r['lr'] for r in rows});max_y=max(r['score'] for r in rows)
        for color,lr in zip(colors,lrs):
            group=sorted([r for r in rows if r['lr']==lr],key=lambda r:r['budget'])
            ax.plot([r['budget'] for r in group],[r['score'] for r in group],marker='o',color=color,label=f'lr={lr:g}',lw=2)
            for r in group:ax.annotate(f"{r['score']:.3f}",(r['budget'],r['score']),xytext=(0,8 if lr==lrs[-1] or r['score']<max_y*.15 else -15),textcoords='offset points',ha='center',fontsize=10)
        ax.set_title(datasets[ds]+' / '+names[model]);ax.set_xlabel('总训练预算（轮）')
        ax.set_ylabel('Q Macro-F1' if ds=='banking77' else 'Q Macro-AP')
        ax.set_xticks(sorted({r['budget'] for r in rows}));ax.set_ylim(-max_y*.05,max_y*1.22)
        ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.18)
    fig.text(.09,.045,'单seed42；预算改变同时改变线性衰减计划，因此解释为优化配方/预算效应。GoEmotions高学习率下5→10轮AP近似持平。',fontsize=9.5,color='#536779')
    save(fig,pdf,'02-S1预算与学习率Q对照')

    fig,axes=plt.subplots(2,2,figsize=(12.6,9));fig.subplots_adjust(left=.09,right=.96,bottom=.15,top=.79,hspace=.5,wspace=.3)
    heading(fig,'S3：独立标注支持和类别贡献影响不同的失败','各固定表示内部比较2×2条件。左列整体Macro-F1，右列原训练定义的尾类Macro-Recall。')
    for i,model in enumerate(['fixed_tfidf_svm','frozen_generic_distilbert_head']):
        for j,key in enumerate(['macro_f1','tail_recall']):
            ax=axes[i,j];rows=[r for r in data['q_records'] if r['stage']=='S3' and r['model']==model]
            for k,(risk,label,color) in enumerate([('uniform','类别贡献均匀',colors[0]),('skew','类别贡献偏斜',colors[1])]):
                vals=[next(r[key] for r in rows if r['support']==s and r['risk']==risk) for s in ['full','tail_reduced']]
                xpos=np.arange(2)+(k-.5)*.33
                ax.bar(xpos,vals,width=.31,label=label,color=color)
                for x,v in zip(xpos,vals):ax.text(x,v+(0.012 if i==0 else .004),f'{v:.3f}',ha='center',fontsize=10)
            ax.set_xticks([0,1],['标注支持完整（781条）','尾部删减（555条）']);ax.set_title(names[model])
            ax.set_ylim(0,.82 if i==0 else (.15 if j==0 else .26));ax.set_ylabel('Q Macro-F1' if j==0 else 'Q 尾类Macro-Recall')
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if i==0 and j==0:ax.legend(frameon=False,fontsize=9)
    fig.text(.09,.065,'冻结表示模型在支持删减时：贡献偏斜使整体F1从0.044升至0.103，尾类召回却从4.8%降至0.4%。',fontsize=11,color='#17365D')
    fig.text(.09,.027,'不同表示的绝对分数不作同等优化后的架构排名。BANKING77 Q共3,079条，固定同一尾类集合；各格仅一个训练seed。',fontsize=9.5,color='#536779')
    save(fig,pdf,'03-S3支持与贡献Q机制对照')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
primary_table=table(['条件','SVM','BiLSTM（均值±SD）','DistilBERT（均值±SD）'],[
    [datasets[ds]+(' 1k / Q Macro-F1' if ds=='banking77' else ' 6k / Q Macro-AP')]+
    [f"{next(r for r in data['primary'] if r['dataset']==ds and r['model']==m)['mean']:.4f}"+
        (f" ± {next(r for r in data['primary'] if r['dataset']==ds and r['model']==m)['sd']:.4f}" if m!='tfidf_svm' else '')
     for m in ['tfidf_svm','bilstm','distilbert']] for ds in ['banking77','go_emotions']])
s1_table=table(['数据集/模型','学习率','较短预算Q','较长预算Q','变化（百分点）'],[
    [r['context'].split(' / ')[0]+'/'+r['context'].split(' / ')[1],r['context'].split(' / ')[2],
     f"{r['before']:.4f}",f"{r['after']:.4f}",f"{r['difference_pp']:+.2f}"]
    for r in data['contrasts'] if r['section']=='S1预算'])
s3_table=table(['固定表示','标注支持','类别贡献','整体Q F1','尾类Q F1','尾类Q召回'],[
    [names[r['model']],'完整781' if r['support']=='full' else '删减555','均匀' if r['risk']=='uniform' else '偏斜',
     f"{r['macro_f1']:.4f}",f"{r['tail_f1']:.4f}",f"{r['tail_recall']:.3%}"]
    for r in data['q_records'] if r['stage']=='S3'])
m2_table=table(['初始化/编码器','V Macro-F1均值','配置间SD'],[
    [r['arm'],f"{r['mean']:.4f}",f"{r['sd']:.4f}"] for r in data['m2_summary']])
imbalance_table=table(['任务','模型','均衡V F1','倾斜V F1','降幅（百分点）'],[
    [r['dataset'],names[r['model']],f"{r['balanced']:.4f}",f"{r['skewed']:.4f}",f"{r['drop_pp']:.2f}"] for r in data['imbalance']])
s4_table=table(['模型','原文准确率','线索替换准确率','控制替换准确率','线索相对控制降幅'],[
    [names[r['model']],f"{r['original']:.2%}",f"{r['cue']:.2%}",f"{r['control']:.2%}",f"{r['drop_pp']:.2f}个百分点"] for r in data['s4']])
trajectory_table=table(['任务','模型','D样本数','持续错误','被纠正','新增错误'],[
    [r['dataset'],names[r['model']],r['n'],r['persistent_error'],r['corrected'],r['new_error']] for r in data['trajectories']])
report=f'''# 文本分类模型能力边界与失败机制：基于现有结果的统一分析

分析日期：{data['generated_at_sgt'][:10]}（新加坡时间）。本版纳入42个已完成Q评估的固定模型，另结合既有V对照、D错误轨迹和37对词汇干预。

**当前主要结论：模型的优势取决于数据条件、优化配方和预训练表征；尾类失败既受独立标注支持影响，也受训练贡献分配影响。整体分数、逐标签的样本排序与具体错误稳定性需要分别观察。**

本分析从保存的42份逐样本预测重新计算全部指标，核对文件SHA256、Q样本ID、标签、group_id和固定阈值，指标误差小于1e-12。[计算与来源记录](analysis-v2/analysis-data.json)。没有启动新训练、推理或Q选参。

[研究主线与六组数据的含义](../../RESEARCH.md)逐项说明“观察现象—子问题—对照证据—核心问题答案”。[查看三页Q分析图](analysis-v2/Q测试与机制对照分析.pdf)。本版替换旧的执行中状态；[Q完成前的分析原文](analysis-v2/results-analysis-before-42Q.md)保留。

## 1. 核心问题与分析路线

### 1.1 承接交接前的研究发现

原项目已比较常规分类、MultiNLI和AG News类别不平衡，发现模型优势具有任务条件性，并留下预训练归因、尾类样本稀缺与类别偏置未分离等问题。当前研究承接这些发现：E1/S1与后续优化检验规模和配方因素，M2分解预训练/适配，S3分离标注支持与类别贡献，S2/S4把总体差异连接到具体错误。[交接前后的连续论证](../../../docs/RESEARCH-CONTINUITY.md)按原交接清单核实来源。

这是从性能比较推进到局部机制解释；BANKING77的M2/S4不直接解释原MultiNLI分差，原四线程batch1延迟也不与新八线程批预测均摊耗时直接求加速比。[完整图册](complete-atlas-v1/完整研究图册.pdf)以接续图开篇，再展示11张结果图。

### 1.2 当前问题与分析路线

能力边界问题是：在什么任务、数据规模和类别分布下，模型优势出现、消失或反转？失败机制问题是：优化、预训练表征、标注支持和词汇线索分别怎样影响失败？

分析顺序为：基准比较发现现象 → 提出候选解释 → 用匹配对照/干预检验 → 报告效应大小与具体错误 → 回答这两个问题。短预算结果用于发现现象，不能直接充当架构上限。V用于选择与探索，D用于错误观察与干预，Q用于固定检查点的独立评估。

## 2. 现象一：优化后的两组Q测试有一致排序

{primary_table}

BANKING77训练1,000条、Q 3,079条；GoEmotions训练6,000条、Q 5,401条。神经模型为seed42/43/44的均值与样本SD，SVM为确定性单次；训练种子SD不是测试置信区间。所有配方由V选定，Q没有用来重新选模型。

两组条件均为 **DistilBERT > SVM > BiLSTM**，且三个神经种子分别都保持这个相对SVM的方向。DistilBERT相对SVM分别高15.81个Macro-F1百分点和6.20个Macro-AP百分点；BiLSTM分别低14.80和13.50个百分点。

这说明：在这两组已测条件和优化范围内，预训练Transformer的优势稳定出现；同属深度学习的BiLSTM仍低于稀疏线性SVM。不能用一个“深度模型”标签概括二者的能力边界，也不能从这两个点位推出精确的样本量临界值。

![优化后Q模型比较](analysis-v2/01-优化后Q模型比较.png)

## 3. 子问题一：早期低分有多少来自优化配方？

**候选解释：** 神经模型的低分可能部分来自学习率和训练预算设置，而不是数据条件下已达到的能力上限。

**实验：** S1在相同数据、输入和CPU设置下，对BiLSTM做10/20轮、DistilBERT做5/10轮的学习率×预算对照；全部16个固定检查点现已评估Q。

{s1_table}

**结果与解释：** BANKING77的DistilBERT在lr=5e-5时，预算5→10轮使Q Macro-F1从0.3600升至0.6689，提高30.88个百分点。lr=2e-5也提高14.77个百分点。GoEmotions在lr=2e-5时，Q Macro-AP提高5.98个百分点。优化因素确实解释了一部分早期低分。

但GoEmotions在lr=5e-5时，5→10轮的Q AP为0.4050→0.4030，基本持平并略降；同一对照的Q Macro-F1却从0.3057升至0.3878。增加预算不是单调获益，逐标签的样本排序指标（Macro-AP）与固定阈值决策指标也会给出不同变化。

后续学习率网格与动态早停使BiLSTM明显优于其短预算检查点，但上述两组Q条件仍低于SVM。因此“部分旧低分受优化影响”得到支持，“训练更久就能让所有深度模型超过SVM”不被当前结果支持。

S1的总预算同时改变线性衰减计划，效应解释为配方/预算变化；新网格另改变调度与学习率范围。新旧配方差距不全部归于epoch或early stopping中的某一个因素。S1仅seed42，0.4050这个Q值只作既定对照展示，不据它重新选择最终配方。

![S1对照](analysis-v2/02-S1预算与学习率Q对照.png)

## 4. 子问题二：预训练表征与下游适配各起什么作用？

**实验：** M2在BANKING77 1,000条条件下固定架构和分类头初始化，比较预训练/随机初始化×冻结/解冻。下表只取相同编码器学习率2e-5的四臂，排除额外的随机初始化学习率敏感性记录。每臂3个数据抽样×3个训练种子，共9个配置。

{m2_table}

**解释：** 在解冻条件下，预训练相对随机初始化提高50.08个V Macro-F1百分点；在预训练条件下，解冻相对冻结提高45.74个百分点。这两种比较在9/9成对配置中方向一致。预训练表征和任务适配共同构成优势，不能把它全部归于模型架构本身。

这是该受控机制设置下的V证据，未包含在42个Q模型中。九个配置包含两类种子，SD描述配置波动，不把九条记录当作九个独立任务。

## 5. 现象二与子问题三：类别倾斜为什么造成失败？

旧固定预算的E3对照显示，类别倾斜鲁棒性具有任务条件性：

{imbalance_table}

Yelp上SVM下降24.51个百分点，BiLSTM/DistilBERT约下降7.98/7.99；BANKING77上DistilBERT下降22.23个百分点，反而大于SVM的6.37。上述为V探索，包含原短预算设置，不能据此给所有任务排一个统一的鲁棒性顺序。

**候选解释：** 尾类的独立标注减少与尾类的训练贡献降低，可能分别造成损失；给尾类增加权重是否能替代缺失的标注支持？

**实验：** S3固定表示，分别改变完整/尾部删减支持和均匀/偏斜贡献。完整支持781条，删减后555条；SVM总权重固定781，冻结编码器分类头固定245次更新。尾类集合沿用训练条件定义，Q后不重新挑选类别。

{s3_table}

**Q结论一：均匀类别贡献不能替代独立标注支持。** 固定TF-IDF+SVM、贡献均匀时，删减尾部标注使整体F1从0.6541降至0.4531，损失20.10个百分点；尾类F1从0.6508降至0.1567，尾类召回从65.6%降至10.4%。贡献偏斜条件下删减支持也损失17.50个整体F1百分点。这个Q结果复现了原V对照的方向。

**Q结论二：贡献偏斜也能独立伤害尾类。** 标注支持完整时，SVM改为偏斜贡献，整体F1下降3.57个百分点，尾类召回从65.6%降至50.0%。这两个因素有各自的作用，不能把长尾退化只解释为单一的样本比例。

**Q结论三：整体Macro-F1提高可以伴随尾类进一步失败。** 在冻结编码器分类头、支持删减条件下，均匀→偏斜贡献使整体F1从0.0436升至0.1033，而尾类召回从4.8%降至0.4%、尾类F1从0.0316降至0.0073；头类F1则从0.0494升至0.1495。整体改善主要伴随头类收益，不能当作尾类获救。

这支持“标签信息不足”和“训练贡献分配”两类局部失败解释，也说明即使报告Macro-F1，仍需检查预先定义的尾类指标。两种固定表示各自做内部对照；冻结编码器分类头不等于端到端微调DistilBERT，这里的绝对分数不作架构优劣比较。八格仅一个训练seed，结论限于该设置和干预强度。

![S3独立Q机制对照](analysis-v2/03-S3支持与贡献Q机制对照.png)

## 6. 子问题四：原文高分是否意味着对词汇表达稳定？

**实验：** S4/M1使用BANKING77的两个固定6k模型，在37对Gemini审核样本上比较原文、线索词替换和同编辑预算控制替换。这批干预样本与上述1k正式Q条件不同。

{s4_table}

**解释：** 在这组可改写样本中，SVM对线索替换更敏感，DistilBERT相对更稳定。原文SVM准确率更高，但线索替换后排序反转；常规分数优势不保证词汇扰动稳定性。

**案例M1-013：近邻意图混淆。** 原文“I just did a money transaction but its not showing they received it”两模型均正确识别“收款人未收到转账”。transaction→transfer后，SVM转为pending_transfer，DistilBERT保持正确；控制did→made后两者仍正确。这展示了线索形式变化诱发的局部近邻标签混淆。

**案例M1-019：预训练模型也会失稳。** “No refund showing in my statement.”中refund→reimbursement后，两模型都错；控制showing→appearing后两者仍正确。因此这里得到的是相对稳定性差异，不能把预训练表征解释为对同义表达完全不敏感。

37对属于Gemini审核后选出的子集，按实际来源报告为AI辅助审核；两个表格版本不作为双真人独立复核。案例用于解释可能的失败方式，不把两条案例当作整体发生率估计。

## 7. 子问题五：加数据后，哪些错误仍然存在？

S2在固定未预留D样本上比较1k/6k检查点。以下为实际同样本轨迹计数：

{trajectory_table}

数据增加能纠正部分错误，也会引入新错误；规模收益具有样本选择性。DistilBERT在这四组D条件均有持续错误，现有证据已经定位了这些样本，但没有系统证明它们都来自某一个机制。

这些轨迹使用原短预算模型，不代表优化后仍不可解。BANKING77只有23个D样本；GoEmotions按固定0阈值的完整标签集合判断整条对错，不能把其持续错误比例等同于1−Macro-AP，也不与单标签错误率混合排名。

## 8. 对核心研究问题的回答

**能力边界：** 在已测BANKING77 1k与GoEmotions 6k优化配方下，DistilBERT在Q上持续领先SVM，BiLSTM持续落后SVM。旧长尾对照又显示相对鲁棒性随任务变化，词汇干预显示原文排名与扰动后排名可以不同。能力应按“任务×数据条件×优化配方×评价维度”描述，当前没有一个适用于所有条件的模型总榜。

**失败机制：** 现有受控证据分别支持优化配方不合适、预训练表征/任务适配不足、独立尾类标注不足、类别贡献偏斜及词汇形式敏感性这些局部解释。S1/S3已有Q结果，M2主要是V证据，S4是AI审核子集上的配对干预。它们可以解释各自对应的失败，不能跨任务拼成统一的内部机制。

当前可以肯定陈述：①部分早期神经模型低分受优化设置影响；②预训练Transformer在两已测优化条件下优于SVM，但BiLSTM没有同样优势；③均匀贡献不能弥补本组尾类独立标注删减；④整体F1改善可以掩盖尾类恶化；⑤DistilBERT在本组词汇干预中比SVM更稳定。

剩余未回答的是精确的样本量临界点、未测条件上的普遍性、持续错误的系统语义分类及M4窗口信息机制。已有结果不需要等整套原始矩阵跑完才能形成上述条件明确的结论。

## 9. 结果与复核入口

- 42个Q模型：18个优化/历史参考、16个S1、8个S3；均在读取Q前固定模型和规则。
- [重算脚本](analysis-v2/analyse.py)、[完整分析数据及逐源SHA](analysis-v2/analysis-data.json)。本次42份指标均从预测重建并与保存结果核对。
- [三页Q分析图](analysis-v2/Q测试与机制对照分析.pdf)，分别展示优化后模型比较、S1、S3。
- [Excel对照与42模型明细](../../../outputs/research-analysis-20261108/研究问题定量分析.xlsx)。
- [S4原始结果](supplement-1-4/S4-ai-reviewed/results.json)、[S2轨迹结果](supplement-1-4/cluster-snapshot/S2-summary.json)。
- 原模型训练数量不等于独立结论数量；428个完成记录包含选参与工程检查，当前研究口径232个，其中42个已有Q评估。
'''
(FOLDER.parent/'results-analysis-v1-zh.md').write_text(report,encoding='utf-8')
print(json.dumps({'report_updated':True,'Q_figures':3,'pdf_pages':3},ensure_ascii=False))

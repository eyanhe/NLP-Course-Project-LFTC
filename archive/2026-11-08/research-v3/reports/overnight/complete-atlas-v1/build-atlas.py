"""Publish one coherent atlas from saved evidence; never load or run models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import textwrap
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
OVERNIGHT = HERE.parent
SUPPORT = HERE/'support'
SUPPORT.mkdir(exist_ok=True)
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(PROJECT))] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


data = read(OVERNIGHT/'analysis-v2/analysis-data.json')
timing = read(OVERNIGHT/'convergence-v1/training-inference-timings.json')
for source in [data, timing]:
    for relative, expected in source['source_sha256'].items():
        path = PROJECT/relative
        assert sha(path) == expected, relative
        SOURCES[relative] = expected
assert len(data['q_records']) == len({r['run_id'] for r in data['q_records']}) == 42
assert len(data['scale']) == 24 and len(data['imbalance']) == 9
assert len(data['trajectories']) == 12 and len(data['m2']) == 36

# Verify that the historical stage really belonged to the original handoff.
package=read(PROJECT/'FULL-PACKAGE-MANIFEST.json')
handoff_entries={r['path']:r for r in package['files']}
historical_files=['HANDOFF.md','report/final-report.md','report/final-synthesis-zh.md',
    'results-v2/metrics.csv','results-hard/metrics.csv','results-imbalance/comparison.csv',
    'results-imbalance/mitigation.csv','results-v2/benchmark.json']
for relative in historical_files:
    assert relative in handoff_entries, relative
    actual=sha(PROJECT/relative)
    assert actual==handoff_entries[relative]['sha256'], relative
    SOURCES[relative]=actual
SOURCES['docs/RESEARCH-CONTINUITY.md']=sha(PROJECT/'docs/RESEARCH-CONTINUITY.md')


def historical_csv(relative):
    with (PROJECT/relative).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


old_conventional=historical_csv('results-v2/metrics.csv')
old_nli=historical_csv('results-hard/metrics.csv')
old_imbalance=historical_csv('results-imbalance/comparison.csv')
old_benchmark=json.loads((PROJECT/'results-v2/benchmark.json').read_text(encoding='utf-8'))
legacy_summary={'conventional':[],'nli':[],'severe_imbalance':[]}
for ds in ['imdb','rotten_tomatoes','yelp','ag_news']:
    group=[r for r in old_conventional if r['source']==r['target']==ds]
    lr=next(float(r['macro_f1']) for r in group if r['model']=='logistic_regression')
    deep=[float(r['macro_f1']) for r in group if r['model']=='distilbert']
    assert len(deep)==3
    legacy_summary['conventional'].append({'dataset':ds,'LR_macro_f1':lr,
        'DistilBERT_macro_f1_mean':statistics.mean(deep),'difference_pp':100*(statistics.mean(deep)-lr)})
for target in ['mnli_matched','mnli_mismatched']:
    group=[r for r in old_nli if r['target']==target]
    svm=next(float(r['macro_f1']) for r in group if r['model']=='linear_svm')
    deep=[float(r['macro_f1']) for r in group if r['model']=='distilbert']
    assert len(deep)==3
    legacy_summary['nli'].append({'target':target,'paired_SVM_macro_f1':svm,
        'DistilBERT_macro_f1_mean':statistics.mean(deep)})
for model in ['linear_svm','distilbert']:
    r=next(r for r in old_imbalance if r['model']==model and r['condition']=='severe')
    legacy_summary['severe_imbalance'].append({'model':model,'macro_f1':float(r['macro_f1']),
        'drop_pp':float(r['macro_f1_drop_pp'])})
assert all(r['threads']==4 and r['batch_size']==1 for r in old_benchmark['rows'])
legacy_summary['timing_scope']={'cpu_threads':4,'batch_size':1,'current_timing_comparable_as_speedup':False}

BRIDGE_ROWS=[
    {'branch':'性能与成本',
     'before':'四套常规任务：LR与DistilBERT\n在IMDb/Yelp差距不足1点，\nRT/AG News差距更大；传统推理更快。',
     'question':'优势为什么随任务改变？\n数据量和优化配方各起什么作用？\n更高性能需要付出多少成本？',
     'now':'E1规模 ＋ S1/早停 ＋ 最终Q ＋ 计时\n两组优化Q：DistilBERT > SVM > BiLSTM。\n按各自条件保留性能与成本的权衡。'},
    {'branch':'表征与适配',
     'before':'MultiNLI matched：\n配对SVM 0.4990，DistilBERT 0.6580。\n原报告指出外部预训练与架构差异并存。',
     'question':'预训练初始化和下游适配\n分别贡献什么？\n不能把优势全部归于架构名称。',
     'now':'M2：BANKING77同架构四臂\n初始化 × 冻结/解冻，匹配9个配置。\nV证据支持预训练与任务适配共同贡献。'},
    {'branch':'长尾与补救',
     'before':'AG News严重倾斜：F1降幅\nSVM 77.76点，DistilBERT 12.86点。\n加权有收益，但未分离稀缺与类别偏置。',
     'question':'独立标注不足与训练贡献不足\n能否分别解释退化？\n增加权重是否能替代增加信息？',
     'now':'E3任务扩展 ＋ S3支持×贡献Q对照\n均匀贡献不能补回本组缺失标注；\n平均分改善仍可能伴随尾类恶化。'},
    {'branch':'具体错误与线索',
     'before':'已有逐样本预测、待复核错误队列，\nhypothesis-only诊断提示可用线索。\n现象观察尚未构成统一机制验证。',
     'question':'加数据后哪些错误持续或新增？\n线索形式变化能否触发错误？\n如何把总体分数连接到具体失败？',
     'now':'S2同样本轨迹 ＋ S4配对词汇干预\n定位错误转移；37对AI审核子集中，\nSVM对线索替换比DistilBERT更敏感。'},
]

MODELS = ['tfidf_svm', 'bilstm', 'distilbert']
MODEL_NAME = {'tfidf_svm': 'SVM', 'bilstm': 'BiLSTM', 'distilbert': 'DistilBERT',
              'fixed_tfidf_svm': '固定TF-IDF + SVM',
              'frozen_generic_distilbert_head': '冻结编码器 + 分类头'}
DATASET_NAME = {'yelp': 'Yelp', 'banking77': 'BANKING77',
                'go_emotions': 'GoEmotions', 'yahoo_answers_topics': 'Yahoo'}
DATASETS = ['yelp', 'banking77', 'go_emotions', 'yahoo_answers_topics']
COLORS = ['#3178A8', '#D48B31', '#2E856F']
MODEL_COLOR = dict(zip(MODELS, COLORS))
NAVY, MUTED, GRID = '#17365D', '#526779', '#E4EAF0'
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'pdf.fonttype': 42,
    'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
PAGES = []
QA = []


def page(number, title, subtitle, shape=(1, 1), bottom=.24, top=.77):
    fig, axes = plt.subplots(*shape, figsize=(14, 9), squeeze=False)
    fig.subplots_adjust(left=.10, right=.95, bottom=bottom, top=top,
                        wspace=.32, hspace=.53)
    fig.text(.065, .949, f'{number:02d}  {title}', fontsize=21,
             weight='bold', color=NAVY, va='top')
    fig.text(.065, .887, subtitle, fontsize=11, color=MUTED, va='top')
    fig.text(.95, .04, f'结果图 {number:02d} / 11', ha='right', fontsize=10, color=MUTED)
    return fig, axes


def footer(fig, takeaway, scope):
    fig.text(.065, .143, '结论  '+takeaway, fontsize=12, weight='bold', color=NAVY, va='top')
    fig.text(.065, .091, scope, fontsize=9.7, color=MUTED, va='top', linespacing=1.6)


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def score_name(ds, split):
    return split+' Macro-'+('AP' if ds == 'go_emotions' else 'F1')


def legend(fig, handles, labels, ncol=3, y=.826):
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.51, y),
               ncol=ncol, frameon=False, fontsize=10.5)


def finish(fig, pdf, number, slug, question, takeaway, scope, values, source_groups):
    # Flag clipped text before export; axes labels and figure footers are checked too.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = fig.findobj(match=matplotlib.text.Text)
    for item in texts:
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        # Invisible/out-of-range tick positions are not rendered by their parent axes.
        if item.axes is not None and item not in item.axes.texts:
            continue
        assert box.x0 >= -2 and box.y0 >= -2, (slug, item.get_text())
        assert box.x1 <= fig.bbox.width+2 and box.y1 <= fig.bbox.height+2, (slug, item.get_text())
    filename = f'{number:02d}-{slug}.png'
    fig.savefig(HERE/filename, dpi=160)
    pdf.savefig(fig)
    PAGES.append({'page': number, 'pdf_page':number+1,'file': filename, 'question': question,
        'takeaway': takeaway, 'scope': scope, 'source_groups': source_groups, 'data': values})
    QA.append({'page': number, 'text_bounds_checked': True,
               'figure_width_inches': 14, 'figure_height_inches': 9})
    plt.close(fig)


# Read exactly the twelve optimized neural states and their selected-epoch histories.
curves = []
for row in data['q_records']:
    if row['role'] != 'primary' or row['model'] not in ['bilstm', 'distilbert']:
        continue
    path = OVERNIGHT/'convergence-v1/cluster-snapshot/runs'/row['run_id']/'status.json'
    state = read(path)
    metric = 'macro_ap' if row['dataset'] == 'go_emotions' else 'macro_f1'
    history = [{'epoch': h['epoch'], 'metric': h['validation'][metric],
                'train_loss': h['train_loss']} for h in state['history']]
    assert [h['epoch'] for h in history] == list(range(1, state['epochs_completed']+1))
    chosen = next(h for h in history if h['epoch'] == state['selected_epoch'])
    assert abs(chosen['metric']-state['validation'][metric]) < 1e-12
    curves.append({'run_id': row['run_id'], 'dataset': row['dataset'], 'model': row['model'],
        'seed': row['seed'], 'lr': row['lr'], 'selected_epoch': state['selected_epoch'],
        'epochs_completed': state['epochs_completed'], 'stop_reason': state['stop_reason'],
        'selected_V': chosen['metric'], 'history': history})
assert len(curves) == 12
selected_ids = {r['run_id'] for r in data['q_records'] if r['role'] == 'primary'}
cost_rows = [r for r in timing['rows'] if r['run_id'] in selected_ids]
assert len(cost_rows) == 14
assert all(r['training_cpu_threads'] == r['inference_cpu_threads'] == 8 for r in cost_rows)
assert all(r['training_device'] == 'cpu' and r['predict_seconds'] > 0 and
           r['training_pipeline_wall_seconds'] > 0 for r in cost_rows)

with PdfPages(HERE/'完整研究图册.pdf', metadata={'Title': '文本分类能力边界与失败机制：完整研究图册'}) as pdf:
    # One context page connects inherited findings to the new studies. It is not a twelfth result figure.
    bridge=plt.figure(figsize=(14,9),facecolor='white')
    bax=bridge.add_axes([0,0,1,1]);bax.set_xlim(0,1);bax.set_ylim(0,1);bax.axis('off')
    bax.text(.055,.95,'研究接续：从交接前的比较，走向当前的机制解释',fontsize=21,weight='bold',color=NAVY,va='top')
    bax.text(.055,.884,'同一主线逐步深入：先发现模型表现具有条件性，再检验规模、优化、表征和监督信息的作用。',fontsize=11,color=MUTED)
    xs=[.125,.413,.70];widths=[.26,.26,.27]
    for x,w,title in zip(xs,widths,['交接前已经得到的发现','由此继续追问','当前实验怎样推进']):
        bax.add_patch(FancyBboxPatch((x,.79),w,.049,boxstyle='round,pad=0.006',facecolor=NAVY,edgecolor='none'))
        bax.text(x+w/2,.8145,title,ha='center',va='center',fontsize=11.4,color='white',weight='bold')
    bridge_text_checks=[]
    for i,row in enumerate(BRIDGE_ROWS):
        y=.65-i*.14;h=.115
        bax.text(.023,y+h/2,row['branch'],fontsize=10.5,weight='bold',color=NAVY,va='center')
        for j,key in enumerate(['before','question','now']):
            x=xs[j];w=widths[j]
            bax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',facecolor=['#F0F4F8','#FFF3DD','#E8F2ED'][j],edgecolor='#D4DEE7',linewidth=.6))
            item=bax.text(x+.01,y+h/2,row[key],fontsize=9.8,color=NAVY,va='center',linespacing=1.65)
            bridge_text_checks.append((item,(x,y,x+w,y+h)))
            if j<2:bax.annotate('',(xs[j+1]-.008,y+h/2),(x+w+.008,y+h/2),arrowprops={'arrowstyle':'->','color':'#71889D'})
    bax.text(.055,.145,'整体论证：任务/分布/成本比较 → 条件性优势与失败 → 可检验假设 → 匹配对照与干预 → 定量/定性解释',fontsize=11.5,weight='bold',color=NAVY)
    bax.text(.055,.093,'原结果按11月6日交接清单验SHA；后续问题是当前的梳理，不倒写为旧阶段已完成的机制验证。',fontsize=9.6,color=MUTED)
    bax.text(.055,.065,'MultiNLI与BANKING77不是同一任务；旧batch1延迟与新批预测均摊耗时不直接拼接。后续11张结果图保持原证据口径。',fontsize=9.6,color=MUTED)
    bridge.canvas.draw();renderer=bridge.canvas.get_renderer()
    for item,(x1,y1,x2,y2) in bridge_text_checks:
        bounds=item.get_window_extent(renderer).transformed(bax.transData.inverted())
        assert bounds.x0>=x1 and bounds.x1<=x2 and bounds.y0>=y1 and bounds.y1<=y2,item.get_text()
    bridge.savefig(HERE/'00-交接前后研究接续.png',dpi=160);pdf.savefig(bridge);plt.close(bridge)
    # 1. The fixed-budget scale comparison introduces the observed phenomenon.
    fig, axes = page(1, '加数据后，模型差距怎样变化？',
        'E1规模对照｜同任务训练1,000→6,000条；四个任务分别比较，使用各自V主指标。', (2, 2))
    for ax, ds in zip(axes.flat, DATASETS):
        for model in MODELS:
            rows = sorted([r for r in data['scale'] if r['dataset'] == ds and r['model'] == model], key=lambda r:r['n'])
            assert [r['n'] for r in rows] == [1000, 6000]
            ax.plot([0, 1], [r['score'] for r in rows], 'o-', color=MODEL_COLOR[model], lw=2.3, label=MODEL_NAME[model])
        ax.set_xticks([0, 1], ['1,000条', '6,000条'])
        ax.set_xlim(-.15, 1.15)
        ax.set_ylim(0, 1 if ds != 'go_emotions' else .5)
        ax.set_title(DATASET_NAME[ds], loc='left', weight='bold')
        ax.set_ylabel(score_name(ds, 'V')); grid(ax)
    legend(fig, *axes.flat[0].get_legend_handles_labels())
    take = '数据规模会改变模型差距；BANKING77的BiLSTM、DistilBERT在6k时明显改善。'
    scope = '旧固定预算探索：BiLSTM 10轮、DistilBERT 5轮；单配置无误差条。这里不是优化后Q学习曲线。\n两个规模点不能定位精确临界样本量；各任务指标不合并排名。'
    footer(fig, take, scope)
    finish(fig, pdf, 1, '数据规模与模型差距', '增加数据能否改变模型优势？', take, scope, data['scale'], ['scale / V'])

    # 2. S1 diagnoses optimization sensitivity, with the independent Q outcomes.
    fig, axes = page(2, '早期低分有多少来自优化配方？',
        'S1独立Q｜同任务、同模型，比较两档学习率×两档总预算；共16个固定模型。', (2, 2))
    contexts = [(ds, model) for ds in ['banking77', 'go_emotions'] for model in ['bilstm', 'distilbert']]
    for ax, (ds, model) in zip(axes.flat, contexts):
        rows = [r for r in data['q_records'] if r['stage'] == 'S1' and r['dataset'] == ds and r['model'] == model]
        maximum = max(r['score'] for r in rows)
        for color, lr in zip(COLORS[:2], sorted({r['lr'] for r in rows})):
            group = sorted([r for r in rows if r['lr'] == lr], key=lambda r:r['budget'])
            ax.plot([0, 1], [r['score'] for r in group], 'o-', color=color, lw=2.3, label=f'lr={lr:g}')
            for x, row in enumerate(group):
                ax.annotate(f"{row['score']:.3f}", (x, row['score']), xytext=(0, 9 if color==COLORS[1] else -15),
                            textcoords='offset points', ha='center', fontsize=10)
        ax.set_xticks([0, 1], [f"{n}轮" for n in sorted({r['budget'] for r in rows})]); ax.set_xlim(-.16, 1.16)
        ax.set_ylim(-maximum*.09, maximum*1.26); ax.set_ylabel(score_name(ds, 'Q'))
        ax.set_title(DATASET_NAME[ds]+' / '+MODEL_NAME[model], loc='left', weight='bold')
        ax.legend(frameon=False, fontsize=9,
                  loc='lower right' if ds=='go_emotions' and model=='distilbert' else 'upper left'); grid(ax)
    take = 'BANKING77 DistilBERT在lr=5e-5时提高30.88点；GoEmotions高学习率下AP基本持平。'
    scope = '单seed42；预算变化同时改变线性衰减计划，解释为配方/预算效应，不是纯轮数效应。\n不同子图y轴范围不同；所有检查点在Q评估前已固定。'
    footer(fig, take, scope)
    finish(fig, pdf, 2, '优化配方的Q敏感性', '延长预算是否总能改善正式测试？', take, scope,
           [r for r in data['q_records'] if r['stage']=='S1'], ['S1 / Q'])

    # 3. Actual optimized learning curves; selected checkpoint != stopping epoch.
    fig, axes = page(3, '最终配方实际训练到哪里才停止？',
        '后续优化轮｜每个神经模型3个训练种子；实线为逐轮V指标，圆点为选中轮，叉号为停止轮。', (2, 2))
    for ax, (ds, model) in zip(axes.flat, contexts):
        rows = sorted([r for r in curves if r['dataset']==ds and r['model']==model], key=lambda r:r['seed'])
        for color, row in zip(COLORS, rows):
            epochs = [h['epoch'] for h in row['history']]; values = [h['metric'] for h in row['history']]
            ax.plot(epochs, values, color=color, lw=1.8, label=f"seed{row['seed']}：选{row['selected_epoch']} / 停{row['epochs_completed']}")
            ax.scatter(row['selected_epoch'], row['selected_V'], color=color, edgecolor='white', s=54, zorder=4)
            ax.scatter(epochs[-1], values[-1], color=color, marker='x', s=55, zorder=5)
        ax.set_ylim(0, 1 if ds=='banking77' else .55)
        ax.set_title(DATASET_NAME[ds]+' / '+MODEL_NAME[model]+f" / lr={rows[0]['lr']:g}", loc='left')
        ax.set_ylabel(score_name(ds, 'V')); ax.set_xlabel('实际训练轮次')
        ax.legend(frameon=False, fontsize=8.3, loc='lower right'); grid(ax)
    stopped = sum(r['stop_reason']=='patience' for r in curves)
    assert stopped == 12
    take = '12个最终神经模型均触发patience早停；选中轮次与实际训练终止轮次分别记录。'
    scope = 'BANKING77 1k、GoEmotions 6k；仅展示V选定配方的三个种子，未展示全部未选学习率。\n早停表示满足既定停止准则，不等于证明全局最优。未用Q逐轮挑选检查点。'
    footer(fig, take, scope)
    finish(fig, pdf, 3, '实际训练与早停曲线', '最终模型是否只是被固定轮数截断？', take, scope, curves, ['optimized states / V'])

    # 4. Independent generalization comparison after optimization.
    fig, axes = page(4, '优化后，正式测试中的模型差距还在吗？',
        '最终模型Q｜BANKING77训练1k、Q 3,079条；GoEmotions训练6k、Q 5,401条。', (1, 2))
    for ax, ds in zip(axes.flat, ['banking77','go_emotions']):
        rows = [next(r for r in data['primary'] if r['dataset']==ds and r['model']==model) for model in MODELS]
        for x, row in enumerate(rows):
            ax.bar(x, row['mean'], width=.57, color=MODEL_COLOR[row['model']], yerr=row['sd'] or 0, capsize=5)
            values = [r['score'] for r in data['q_records'] if r['run_id'] in row['run_ids']]
            ax.scatter(np.linspace(x-.11,x+.11,len(values)) if len(values)>1 else [x], values, s=23, color=NAVY, zorder=4)
            ax.text(x,row['mean']+(row['sd'] or 0)+.027,f"{row['mean']:.4f}",ha='center',fontsize=12)
        references=[r for r in data['q_records'] if r['dataset']==ds and r['role']=='fixed_budget_reference']
        assert len(references)==2
        for reference in references:
            x=MODELS.index(reference['model'])
            ax.scatter(x,reference['score'],marker='D',color='#727E88',edgecolor='white',s=54,zorder=5)
            ax.annotate(f"旧{reference['score']:.4f}",(x,reference['score']),xytext=(0,11),textcoords='offset points',
                        ha='center',fontsize=9,color='#53606B',bbox={'facecolor':'white','alpha':.88,'edgecolor':'none','pad':1})
        ax.set_xticks(range(3), [MODEL_NAME[m] for m in MODELS]); ax.set_ylim(0,1 if ds=='banking77' else .55)
        ax.set_title(DATASET_NAME[ds], weight='bold', loc='left'); ax.set_ylabel(score_name(ds,'Q')); grid(ax)
    handle=plt.Line2D([],[],marker='D',color='#727E88',linestyle='none',markersize=6)
    legend(fig,[handle],['灰菱形：旧固定预算seed42参考；彩柱：V选定优化配方'],ncol=1)
    take = '两组已测条件均为DistilBERT > SVM > BiLSTM，三个神经训练种子保持相同排序。'
    scope = '彩柱为均值；神经n=3、误差条为训练种子样本SD，散点为每个种子；SVM确定性单次。SD不是测试置信区间。\n灰菱形共4个历史Q参考；新旧配方同时改变学习率/预算/调度，不把差距全归于训练轮数。'
    footer(fig,take,scope)
    finish(fig,pdf,4,'优化后的正式Q比较','优化能否消除所有模型差距？',take,scope,
           {'primary':data['primary'],'historical_Q':[r for r in data['q_records'] if r['role']=='fixed_budget_reference']},['primary + historical reference / Q'])

    # 5. Controlled representation/init mechanism, not a ranking of architecture names.
    fig, axes = page(5,'优势来自预训练，还是下游适配？',
        'M2匹配四臂｜BANKING77 1k；相同DistilBERT架构与分类头初始化，编码器学习率固定2e-5。',(1,2))
    arms = [['random_frozen','pretrained_frozen'],['random_unfrozen','pretrained_unfrozen']]
    for ax, pair, title in zip(axes.flat, arms, ['编码器冻结：只训练分类头','编码器解冻：同时适配任务']):
        left,right = pair
        for i,arm in enumerate(pair):
            row=next(r for r in data['m2_summary'] if r['arm']==arm)
            vals=[r['macro_f1'] for r in sorted([v for v in data['m2'] if v['arm']==arm], key=lambda r:(r['data_seed'],r['training_seed']))]
            ax.bar(i,row['mean'],color=COLORS[0 if i==0 else 2],width=.54,yerr=row['sd'],capsize=5,alpha=.85)
            ax.scatter(i+np.linspace(-.12,.12,9),vals,s=23,color=NAVY,zorder=4)
            ax.text(i,max(vals)+.035,f"{row['mean']:.4f}",ha='center',fontsize=12)
        aa={(r['data_seed'],r['training_seed']):r['macro_f1'] for r in data['m2'] if r['arm']==left}
        bb={(r['data_seed'],r['training_seed']):r['macro_f1'] for r in data['m2'] if r['arm']==right}
        assert aa.keys()==bb.keys()
        ax.set_xticks([0,1],['随机初始化','预训练初始化']);ax.set_ylim(0,1);ax.set_ylabel('V Macro-F1')
        ax.set_title(title,loc='left',weight='bold');grid(ax)
    take='解冻时预训练提高50.08点；预训练后解冻提高45.74点，两个方向均在9/9成对配置中成立。'
    scope='每臂3个数据抽样×3个训练种子，点为配置、柱为均值、误差条为配置间样本SD。\n这是V机制证据，未包含在42个Q模型中；不把9个配置当作9个独立任务。'
    footer(fig,take,scope)
    finish(fig,pdf,5,'预训练与任务适配','预训练与适配分别贡献什么？',take,scope,data['m2'],['M2 / V'])

    # 6. Use both absolute levels and drops to expose the low-baseline trap.
    fig,axes=page(6,'类别倾斜时，哪个模型更容易退化？',
        'E3固定预算探索｜同任务训练总量与V集不变，比较均衡与倾斜条件；左端为倾斜，右端为均衡。',(1,3))
    for ax,ds in zip(axes.flat,['yelp','banking77','yahoo_answers_topics']):
        rows=[next(r for r in data['imbalance'] if r['dataset']==ds and r['model']==m) for m in MODELS]
        for y,row in enumerate(rows):
            color=MODEL_COLOR[row['model']]
            ax.plot([row['skewed'],row['balanced']],[y,y],color=color,lw=4,alpha=.65)
            ax.scatter(row['balanced'],y,color=color,s=70,zorder=4)
            ax.scatter(row['skewed'],y,facecolor='white',edgecolor=color,s=70,zorder=4,linewidth=2)
            ax.text(.01,y+.23,f"{row['balanced']:.3f} → {row['skewed']:.3f}  (−{row['drop_pp']:.2f}点)",fontsize=8.8,color=MUTED)
        ax.set_yticks(range(3),[MODEL_NAME[m] for m in MODELS]);ax.invert_yaxis();ax.set_ylim(2.6,-.55)
        ax.set_xlim(-.04,1.04);ax.set_xlabel('V Macro-F1');ax.set_title(DATASET_NAME[ds]+f" / N={rows[0]['n']:,}",loc='left',weight='bold');grid(ax,'x')
    take='Yelp上SVM退化更大，BANKING77上DistilBERT退化更大；鲁棒性排序随任务变化。'
    scope='实心为均衡、空心为倾斜；原固定预算V结果，包含优化因素。\nBANKING77 BiLSTM均衡F1仅0.0132，小降幅不能称稳健；标注支持和类别贡献尚需S3分开检验。'
    footer(fig,take,scope)
    finish(fig,pdf,6,'类别倾斜的任务差异','是否存在跨任务一致的长尾鲁棒模型？',take,scope,data['imbalance'],['E3 / V'])

    # 7. S3 provides the controlled mechanism and its tail diagnostic.
    fig,axes=page(7,'补训练权重，能否替代缺失的尾类标注？',
        'S3独立Q｜每种固定表示内，比较支持完整/删减×贡献均匀/偏斜；Q 3,079条。',(2,2))
    for i,model in enumerate(['fixed_tfidf_svm','frozen_generic_distilbert_head']):
        for j,key in enumerate(['macro_f1','tail_recall']):
            ax=axes[i,j]
            rows=[r for r in data['q_records'] if r['stage']=='S3' and r['model']==model]
            for k,(risk,label,color) in enumerate([('uniform','贡献均匀',COLORS[0]),('skew','贡献偏斜',COLORS[1])]):
                vals=[next(r[key] for r in rows if r['support']==s and r['risk']==risk) for s in ['full','tail_reduced']]
                xx=np.arange(2)+(k-.5)*.34
                ax.bar(xx,vals,width=.31,color=color,label=label)
                for x,value in zip(xx,vals):
                    ax.text(x,value+(.018 if i==0 else .005),f'{value:.3f}',ha='center',fontsize=9.5)
            ax.set_xticks([0,1],['完整781条','尾部删减555条']);ax.set_title(MODEL_NAME[model],loc='left')
            ax.set_ylim(0,.84 if i==0 else (.16 if j==0 else .28));ax.set_ylabel('Q Macro-F1' if j==0 else 'Q尾类Macro-Recall');grid(ax)
    legend(fig,*axes[0,0].get_legend_handles_labels(),ncol=2)
    take='均匀贡献无法补回独立标注；冻结头整体F1提升0.044→0.103时，尾类召回4.8%→0.4%。'
    scope='SVM固定总权重781，冻结头固定245次更新；尾类集合事先定义，各格单seed。\n不同固定表示不作端到端模型排名；整体Macro-F1改善也可能掩盖局部失效。'
    footer(fig,take,scope)
    finish(fig,pdf,7,'标注支持与尾类失效','监督信息不足与训练贡献不足有何区别？',take,scope,
           [r for r in data['q_records'] if r['stage']=='S3'],['S3 / Q'])

    # 8. Same inputs, all four transitions, with absolute denominators and counts.
    fig,axes=page(8,'加数据后，究竟纠正了谁、又错了谁？',
        'S2样本轨迹｜同一批D文本，比较原1k→6k模型；条内数字为样本数，条长为该D集占比。',(2,2))
    states=[('always_correct','持续正确','#C5D4E3'),('corrected','由错变对','#378573'),
            ('persistent_error','持续错误','#B25B5B'),('new_error','由对变错','#E3A24A')]
    for ax,ds in zip(axes.flat,DATASETS):
        rows=[next(r for r in data['trajectories'] if r['dataset']==ds and r['model']==m) for m in MODELS]
        n=rows[0]['n'];assert all(r['n']==n for r in rows)
        for y,row in enumerate(rows):
            assert sum(row[k] for k,_,_ in states)==n
            left=0
            for key,label,color in states:
                width=100*row[key]/n
                ax.barh(y,width,left=left,color=color,height=.55,label=label if y==0 else None)
                if width>=5:
                    ax.text(left+width/2,y,str(row[key]),ha='center',va='center',color='white' if key in ['corrected','persistent_error'] else NAVY,fontsize=9)
                left+=width
        ax.set_yticks(range(3),[MODEL_NAME[m] for m in MODELS]);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.xaxis.set_major_formatter(PercentFormatter(100));ax.set_title(DATASET_NAME[ds]+f' / D n={n}',loc='left',weight='bold');grid(ax,'x')
    legend(fig,*axes.flat[0].get_legend_handles_labels(),ncol=4)
    take='增加数据能纠错，也会引入新错；四任务的DistilBERT都仍有持续错误样本。'
    scope='旧固定预算检查点；持续错误不等于优化后不可解。BANKING77 D仅23条。\nGoEmotions以0阈值完整标签集合判断整条对错，其他任务为单标签；不能跨任务比较“错误率”。'
    footer(fig,take,scope)
    finish(fig,pdf,8,'逐样本错误迁移','平均分改善背后发生了哪些样本转移？',take,scope,data['trajectories'],['S2 / D'])

    # 9. Paired lexical interventions show both original and matched-control scores.
    fig,axes=page(9,'原文准确，是否代表表达改变后仍稳定？',
        'S4/M1配对干预｜BANKING77两个固定6k模型；同37个样本比较原文、匹配控制替换和线索替换。',(1,2))
    ax=axes[0,0]
    for row in data['s4']:
        model=row['model'];keys=['original','control','cue'];values=[row[k] for k in keys]
        ax.plot(range(3),values,'o-',color=MODEL_COLOR[model],lw=2.5,markersize=8,label=MODEL_NAME[model])
        for x,value in enumerate(values):
            above = (model=='tfidf_svm') if x<2 else (model=='distilbert')
            ax.annotate(f'{round(value*37)}/37  ({value:.1%})',(x,value),xytext=(0,12 if above else -20),textcoords='offset points',ha='center',fontsize=9.5)
    ax.set_xticks(range(3),['原文','匹配控制替换','线索替换']);ax.set_xlim(-.25,2.25)
    ax.set_ylim(.5,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('37例子集准确率');ax.legend(frameon=False);grid(ax)
    ax=axes[0,1]
    for x,row in enumerate(data['s4']):
        ax.bar(x,row['drop_pp'],color=MODEL_COLOR[row['model']],width=.55)
        ax.text(x,row['drop_pp']+1,f"{row['drop_pp']:.2f}点",ha='center',fontsize=13)
    ax.set_xticks([0,1],[MODEL_NAME[r['model']] for r in data['s4']]);ax.set_ylim(0,32)
    ax.set_ylabel('控制准确率 − 线索替换准确率（百分点）');ax.set_title('线索替换相对控制的损失',loc='left');grid(ax)
    take='本组线索替换使SVM损失24.32点、DistilBERT损失8.11点；原文排名不能代表扰动稳定性。'
    scope='37对由Gemini两个表格版本审核，按AI辅助子研究报告；不是双真人独立复核。\n这是筛选出的可改写D子集，不代表整个任务的扰动发生率；误差条未作推断。'
    footer(fig,take,scope)
    finish(fig,pdf,9,'词汇干预的配对比较','原文表现与词汇稳定性是否相同？',take,scope,data['s4'],['S4 / AI-reviewed D'])

    # 10. A qualitative figure connects the aggregate intervention effect to examples.
    fig,axes=page(10,'哪些具体错误解释了词汇敏感性？',
        '定性案例｜同一输入的三臂结果；标签判断依据原业务意图，不用情绪正负替代意图。',(1,2),bottom=.245,top=.79)
    translations={'transfer_not_received_by_recipient':'收款人未收到转账','pending_transfer':'转账处理中',
                  'refund_not_showing_up':'退款未到账','extra_charge_on_statement':'账单额外收费',
                  'direct_debit_payment_not_recognised':'不认识的自动扣款'}
    for ax,case in zip(axes.flat,data['cases']):
        ax.axis('off');edit=case['text_edits'];pred=case['predictions'];first=pred['tfidf_svm']
        ax.text(0,1,case['case_id']+'  '+translations.get(first['gold_name'].lower(),first['gold_name']),fontsize=13,weight='bold',color=NAVY,va='top',transform=ax.transAxes)
        original='\n'.join(textwrap.wrap(edit['original_text'],width=48))
        ax.text(0,.89,'原文\n'+original,fontsize=10.3,linespacing=1.55,va='top',transform=ax.transAxes,color=MUTED)
        words=('transaction → transfer','did → made') if case['case_id']=='M1-013' else ('refund → reimbursement','showing → appearing')
        ax.text(0,.61,'线索替换：'+words[0]+'\n控制替换：'+words[1],fontsize=10.3,linespacing=1.7,transform=ax.transAxes,color=NAVY)
        cell=[];cell_colors=[]
        for model in ['tfidf_svm','distilbert']:
            cell.append([MODEL_NAME[model]]+['正确' if pred[model][key]['correct'] else '错误' for key in ['original','control','cue']])
            cell_colors.append(['#F0F4F8']+['#DDEFE7' if pred[model][key]['correct'] else '#F5DFDF' for key in ['original','control','cue']])
        table=ax.table(cellText=cell,colLabels=['模型','原文','控制替换','线索替换'],cellColours=cell_colors,
                       cellLoc='center',colWidths=[.31,.21,.24,.24],bbox=[0,.25,1,.25])
        table.auto_set_font_size(False);table.set_fontsize(10)
        for (r,c),cell_item in table.get_celld().items():
            cell_item.set_edgecolor('white')
            if r==0:cell_item.set_facecolor(NAVY);cell_item.get_text().set_color('white')
        wrong=[]
        for model in ['tfidf_svm','distilbert']:
            if not pred[model]['cue']['correct']:
                prediction=pred[model]['cue']['prediction_name']
                wrong.append(MODEL_NAME[model]+'误判：'+translations.get(prediction.lower(),prediction))
        for j,line in enumerate(wrong):
            ax.text(0,.16-j*.09,line,fontsize=8.8,va='top',transform=ax.transAxes,color='#9D4444')
    take='转账案例中DistilBERT保持正确，退款同义替换却令两模型都错：相对稳定不等于完全不敏感。'
    scope='案例M1-013、M1-019来自同一37对AI审核干预集；仅解释失败形式，不估计总体频率。\n正确/错误由保存的逐例预测与金标签重建，未重新运行模型。'
    footer(fig,take,scope)
    finish(fig,pdf,10,'词汇失败的具体案例','相同编辑预算下，错误怎样发生？',take,scope,data['cases'],['S4 / qualitative D'])

    # 11. Cost is separated into measured training pipeline and prediction throughput.
    fig,axes=page(11,'取得这些分数，训练与推理分别花多久？',
        '最终14个模型实测｜相同CPU 8线程；上行训练流程总秒数，下行批预测均摊毫秒/条；均使用对数轴。',(2,2))
    cost_summary=[]
    for j,ds in enumerate(['banking77','go_emotions']):
        for i,key in enumerate(['training_pipeline_wall_seconds','milliseconds_per_row']):
            ax=axes[i,j]
            for x,model in enumerate(MODELS):
                rows=[r for r in cost_rows if r['dataset_id']==ds and r['model_id']==model]
                assert len(rows)==(1 if model=='tfidf_svm' else 3)
                if key=='milliseconds_per_row':
                    assert all(abs(r[key]-1000*r['predict_seconds']/r['test_rows'])<1e-12 for r in rows)
                vals=[r[key] for r in rows];median=statistics.median(vals)
                ax.vlines(x,min(vals),max(vals),color=MODEL_COLOR[model],lw=2)
                xx=np.linspace(x-.10,x+.10,len(vals)) if len(vals)>1 else [x]
                ax.scatter(xx,vals,color=MODEL_COLOR[model],s=36,zorder=4)
                ax.plot([x-.18,x+.18],[median,median],color=NAVY,lw=2)
                label=f'{median:.3g}'
                ax.annotate(label,(x,max(vals)),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
                cost_summary.append({'dataset':ds,'model':model,'measure':key,'n':len(vals),'median':median,'min':min(vals),'max':max(vals),'run_ids':[r['run_id'] for r in rows]})
            ax.set_xticks(range(3),[MODEL_NAME[m] for m in MODELS]);ax.set_xlim(-.45,2.45);ax.set_yscale('log')
            values=[r[key] for r in cost_rows if r['dataset_id']==ds]
            ax.set_ylim(min(values)/2,max(values)*4)
            ax.set_ylabel('训练流程（秒，对数轴）' if i==0 else '批预测均摊（毫秒/条，对数轴）')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));grid(ax)
            if i==0:ax.set_title(DATASET_NAME[ds]+(' / 训练1k' if ds=='banking77' else ' / 训练6k'),loc='left',weight='bold')
    take='本轮SVM成本最低；DistilBERT取得更高Q分数，同时付出更高的训练与批预测成本。'
    scope='点为各种子，横线为中位数；训练为既有资产下单模型流程，含特征/验证/保存，不含预训练与全网格搜索。\n预测不含加载/校验/清理，均摊时间不是在线延迟。共享CPU实测；BANKING77 DistilBERT seed43训练含续跑重建。'
    footer(fig,take,scope)
    finish(fig,pdf,11,'训练与推理成本','预测优势需要付出多少计算代价？',take,scope,
           {'summary':cost_summary,'records':cost_rows},['optimized timing / CPU8'])

assert len(PAGES)==11 and [r['page'] for r in PAGES]==list(range(1,12))
plotted_Q_ids={r['run_id'] for r in data['q_records'] if r['stage'] in ['S1','S3'] or r['role'] in ['primary','fixed_budget_reference']}
assert plotted_Q_ids=={r['run_id'] for r in data['q_records']}
assert all(sha(PROJECT/p)==h for p,h in SOURCES.items()), 'Source changed during build'
manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'new_training_or_inference':False,'official_Q_models':42,'thematic_figures':11,
    'pdf_pages':12,'context_pages':1,'Q_models_represented':len(plotted_Q_ids),'existing_Q_topics_included':3,'additional_topics':8,
    'handoff_lineage':{'manifest_created_utc':package['created_utc'],'original_files_verified':historical_files,
        'legacy_summary':legacy_summary,'bridge_rows':BRIDGE_ROWS,'context_image':'00-交接前后研究接续.png'},
    'source_sha256':SOURCES,'pages':PAGES,'layout_checks':QA}
(HERE/'figure-data.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
guide=['# 完整研究图册：阅读入口','',
    '[打开12页PDF](完整研究图册.pdf)：前置1页研究接续图，后接原11张结果主题图。新增接续页不算新实验。',
    '', '[先读交接前后的连续论证](../../../../docs/RESEARCH-CONTINUITY.md)。交接前比较常规任务、MultiNLI和AG News长尾；当前继续追问规模、优化、预训练、标注支持与线索依赖。',
    '', '![交接前后研究接续](00-交接前后研究接续.png)',
    '', '随后按“观察现象 → 检查优化 → 独立泛化 → 分解机制 → 观察错误 → 解释案例 → 计算成本”阅读。',
    '', '| PDF页码 | 结果图 | 比较的问题 | 证据 |', '| --- | --- | --- | --- |']
for row in PAGES:
    guide.append(f"| {row['pdf_page']:02d} | {row['page']:02d} | [{row['question']}]({row['file']}) | {', '.join(row['source_groups'])} |")
guide+=['','V为验证/选择集，D为探索与干预集，Q为固定模型的正式测试。旧固定预算结果用于发现现象，新优化Q用于已测条件下的泛化比较。',
    '', '## 逐图解释','']
for row in PAGES:
    guide += [f"### {row['page']:02d} {row['question']}",'',row['takeaway'],'',row['scope'].replace('\n',' '),'',f"![{row['question']}]({row['file']})",'']
guide+=['## 来源与复现','',
    '[逐图数据及SHA](figure-data.json)保存每张图实际使用的数据、模型ID和全部输入指纹。[生成脚本](build-atlas.py)仅读取已保存结果。',
    '', '[现有完整数值分析](../results-analysis-v1-zh.md)、[研究主线](../../../RESEARCH.md)。',
    '', '`support/`只存PDF渲染与排版核验，不计作研究图。']
(HERE/'README.md').write_text('\n'.join(guide)+'\n',encoding='utf-8')
print(json.dumps({'result_figures':11,'context_pages':1,'pdf_pages':12,'sources':len(SOURCES),'neural_curves':len(curves),
                  'all_selected_neural_runs_stopped_by_patience':stopped,'output':str(HERE)},ensure_ascii=False))

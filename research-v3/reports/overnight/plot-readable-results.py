"""A four-question reading guide for the existing result snapshot; no ML runs."""
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

FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER / 'charts-v1/all-completed-records.csv'
OUT = FOLDER / 'charts-v2'
OUT.mkdir(exist_ok=True)
with SOURCE.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r['n'] = int(r['n'])
    r['score'] = float(r['score']) if r['score'] else None
    r['ratio'] = float(r['ratio']) if r['ratio'] else None
    r['actual_ratio'] = float(r['actual_ratio']) if r['actual_ratio'] else None
    r['data_seed'] = int(r['data_seed'])
    r['training_seed'] = int(r['training_seed']) if r['training_seed'] else None
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white', 'pdf.fonttype': 42})
DATASETS = ('yelp', 'go_emotions', 'banking77', 'yahoo_answers_topics')
TITLES = {'yelp': '情感分类 · Yelp', 'go_emotions': '多标签情绪 · GoEmotions',
          'banking77': '意图分类 · BANKING77', 'yahoo_answers_topics': '主题分类 · Yahoo'}
BLUE, GREEN, GREY = '#286DA8', '#14917F', '#9AA7B6'
MODEL = {'tfidf_svm': '传统 SVM', 'distilbert': 'DistilBERT'}
COLOR = {'tfidf_svm': BLUE, 'distilbert': GREEN}
FOOT = '所有分数来自验证集，已用于选检查点；并非独立测试成绩。主图为固定种子的单次结果。'
coverage, comparisons = {}, []
pdf = PdfPages(OUT / '四个问题看懂当前结果.pdf')

def primary(r):
    return r['data_seed'] == 2026 and r['training_seed'] in (None, 42)

def select(ds, model, n=None, experiment='E1', ratio=None):
    hits = [r for r in rows if r['dataset'] == ds and r['model'] == model and primary(r)
            and r['experiment'] == experiment and r['score'] is not None
            and (n is None or r['n'] == n) and (ratio is None or r['ratio'] == ratio)]
    return sorted(hits, key=lambda r: r['n'])

def one(hits):
    assert len(hits) == 1, len(hits)
    return hits[0]

def mname(ds):
    return 'Macro-AP' if ds == 'go_emotions' else 'Macro-F1'

def decorate(fig, question, fixed, foot=FOOT):
    fig.suptitle(question + '\n' + fixed, fontsize=17, linespacing=1.8)
    fig.supxlabel(foot, fontsize=9, color='#647184')

def save(fig, name, ids):
    fig.savefig(OUT / (name + '.png'), dpi=160, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    coverage[name] = list(dict.fromkeys(ids))
    plt.close(fig)

# Q1: One fair, matched model contrast. No other experiments or seeds here.
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
ids = []
for ax, ds in zip(axes.flat, DATASETS):
    a, b = [one(select(ds, model, 6000)) for model in ('tfidf_svm', 'distilbert')]
    for y, r in enumerate((a, b)):
        ax.barh(y, r['score'], color=COLOR[r['model']], height=.48)
        ax.text(r['score'] + .015, y, f"{r['score']:.4f}", va='center', fontsize=12, weight='bold')
        ids.append(r['run_id'])
    delta = (b['score'] - a['score']) * 100
    direction = '高' if delta >= 0 else '低'
    ax.set_yticks((0, 1), ('传统方法：TF-IDF + SVM', '预训练模型：DistilBERT'), fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(1.8, -.7)
    ax.set_xlim(0, 1.08)
    ax.set_xticks((0, .25, .5, .75, 1))
    ax.set_xlabel(f"验证集 {mname(ds)}（越高越好）")
    ax.set_title(TITLES[ds], loc='left', pad=12)
    ax.text(.02, .07, f'DistilBERT 比 SVM {direction} {abs(delta):.2f} 个百分点',
            transform=ax.transAxes, fontsize=11, color='#263C50',
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': '#EEF3F7', 'edgecolor': 'none'})
    ax.grid(axis='x', alpha=.12)
    ax.set_axisbelow(True)
    comparisons.append({'question': 1, 'dataset': ds, 'svm': a['score'], 'distilbert': b['score'],
                        'difference_percentage_points': delta, 'run_ids': [a['run_id'], b['run_id']]})
decorate(fig, '① 同样的训练数据，传统 SVM 和 DistilBERT 差多少？',
         '固定：每个任务内，同样6,000条训练文本、同一个验证集。只改变模型。')
save(fig, '01-同样数据比较两种模型', ids)

# Q2: Model stays fixed. Categories are explicit sample-size levels.
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
ids = []
N = {1000: '1千条', 6000: '6千条', 20000: '2万条', 100000: '10万条'}
for ax, ds in zip(axes.flat, DATASETS):
    rs = select(ds, 'tfidf_svm')
    assert len({r['n'] for r in rs}) == len(rs)
    x = np.arange(len(rs))
    ax.plot(x, [r['score'] for r in rs], color=BLUE, linewidth=2.8, marker='o', markersize=8)
    for i, r in enumerate(rs):
        ax.text(i, r['score'] + .035, f"{r['score']:.3f}", ha='center', fontsize=11)
        ids.append(r['run_id'])
    gain = (rs[-1]['score'] - rs[0]['score']) * 100
    ax.set_title(TITLES[ds] + f'\n最少到最多样本：增加 {gain:.2f} 个百分点', loc='left', pad=12)
    ax.set_xticks(x, [N[r['n']] for r in rs])
    ax.set_xlim(-.3, len(rs) - .7)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel('训练文本数量（按已完成的样本量分档）')
    ax.set_ylabel(f"验证集 {mname(ds)}（越高越好）")
    ax.grid(axis='y', alpha=.15)
decorate(fig, '② 如果一直用传统 SVM，增加训练文本有帮助吗？',
         '固定：模型都是 TF-IDF + SVM，同一任务使用同一个验证集。只改变训练样本量。')
save(fig, '02-同一模型增加训练数据', ids)

# Q3: Two models, balanced versus skewed. Four directly labelled bars per task.
fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), layout='constrained')
ids = []
for ax, ds in zip(axes, ('yelp', 'banking77', 'yahoo_answers_topics')):
    pairs = []
    for model in ('tfidf_svm', 'distilbert'):
        a, b = [one(select(ds, model, experiment='E3', ratio=ratio)) for ratio in (1, 10)]
        assert a['n'] == b['n']
        pairs.append((model, a, b))
    for j, (model, a, b) in enumerate(pairs):
        for y, r in ((j * 3, a), (j * 3 + 1, b)):
            ax.barh(y, r['score'], color=COLOR[model], alpha=.4 if r is a else 1, height=.65)
            ax.text(r['score'] + .015, y, f"{r['score']:.3f}", va='center', fontsize=10)
            ids.append(r['run_id'])
        drop = (a['score'] - b['score']) * 100
        ax.text(.02, .17 - j * .14, f'{MODEL[model]}：下降 {drop:.2f} 个百分点',
                transform=ax.transAxes, fontsize=11, color=COLOR[model], weight='bold')
        comparisons.append({'question': 3, 'dataset': ds, 'model': model,
                            'balanced': a['score'], 'skewed': b['score'],
                            'drop_percentage_points': drop, 'run_ids': [a['run_id'], b['run_id']]})
    n, ratio = pairs[0][1]['n'], pairs[0][2]['actual_ratio']
    ax.set_yticks((0, 1, 3, 4), ('SVM · 均衡', 'SVM · 不均衡', 'DistilBERT · 均衡', 'DistilBERT · 不均衡'))
    ax.set_ylim(6.5, -.8)
    ax.set_xlim(0, 1.09)
    ax.set_title(TITLES[ds] + f'\n训练量均为{n:,}条；倾斜后最大/最小类约{ratio:.1f}倍', loc='left', fontsize=12, pad=15)
    ax.set_xlabel('验证集 Macro-F1（越高越好）')
    ax.grid(axis='x', alpha=.12)
    ax.set_axisbelow(True)
decorate(fig, '③ 训练数据变得不平衡，两个模型各下降多少？',
         '固定：每个任务内训练总量不变，验证集不变。比较“均衡训练”和“不均衡训练”。',
         '本图为验证集结果；样本组成也随抽样改变，不能视作已隔离所有因素的因果效应。')
save(fig, '03-均衡与不均衡的变化', ids)

# Q4: Show means clearly; individual 36 runs remain in the appendix.
arms = ('pretrained_frozen', 'pretrained_unfrozen', 'random_frozen', 'random_unfrozen')
names = ('有预训练，只训练分类头', '有预训练，同时训练编码器',
         '无预训练，只训练分类头', '无预训练，同时训练编码器')
fig, ax = plt.subplots(figsize=(13, 6), layout='constrained')
ids = []
for i, arm in enumerate(arms):
    rs = [r for r in rows if r['arm'] == arm]
    assert len(rs) == 9
    values = np.array([r['score'] for r in rs])
    mean = values.mean()
    ax.barh(i, mean, color=GREEN if arm.startswith('pretrained') else GREY, height=.6)
    ax.errorbar(mean, i, xerr=[[mean-values.min()], [values.max()-mean]], fmt='none',
                ecolor='#253B4F', capsize=5, linewidth=1.2)
    ax.text(values.max() + .022, i, f'{mean:.4f}', va='center', fontsize=13, weight='bold')
    ids.extend(r['run_id'] for r in rs)
ax.set_yticks(range(4), names, fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel('验证集 Macro-F1（越高越好）')
ax.grid(axis='x', alpha=.12)
ax.set_axisbelow(True)
decorate(fig, '④ 预训练和继续训练编码器，分别有什么影响？',
         '固定：BANKING77任务，1,000条训练文本，四组使用同一套机制实验设置。',
         '柱子为9次平均；细线为最低到最高，不是置信区间。本图训练配置不同于前面的模型比较。')
save(fig, '04-预训练与继续训练编码器', ids)
pdf.close()

manifest = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'source_snapshot': json.loads((FOLDER/'charts-v1/plot-manifest.json').read_text(encoding='utf-8'))['snapshot_singapore'],
            'figure_count': 4, 'comparisons': comparisons, 'per_figure_run_ids': coverage,
            'readable_main_figures_are_selected_matched_comparisons': True,
            'all_completed_records_retained_in_appendix': '../charts-v1/README.md',
            'training_started': False, 'formal_Q_generated': False}
manifest['artifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.png', '.pdf')}
(OUT/'plot-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
text = '''# 先看这四个问题

本页使用同一份已核对的结果快照重新排版。主图选用固定条件的比较；全部120项记录仍保留在附录。
所有性能数字来自验证集，不是独立正式测试成绩。数据快照为2026年11月7日22:23，新加坡时间。

## 1. 同样的数据，传统SVM和DistilBERT差多少？

每个小图只比较两种模型。两根柱子使用同样6,000条训练文本和同一个验证集；柱子越长，分数越高。
蓝色是传统SVM，绿色是DistilBERT。只在同一个小图内比较，GoEmotions与其他任务的指标不同。

![两种模型的差距](01-同样数据比较两种模型.png)

## 2. 一直用SVM，增加训练文本有帮助吗？

模型保持不变，横轴是训练文本数量，纵轴是验证得分。每个小图只追踪一个任务。

![增加训练数据](02-同一模型增加训练数据.png)

## 3. 数据变得不平衡，分数下降多少？

分别比较SVM和DistilBERT在均衡与不均衡训练下的成绩；每个任务内总训练量保持相同。

![不平衡的影响](03-均衡与不均衡的变化.png)

## 4. 预训练和继续训练编码器，分别带来什么变化？

看同一个BANKING77机制实验中的四种训练方式。柱子是9次结果的平均，细线是最低到最高。
这是独立的机制实验配置，不能把它的分数拼入前面模型的学习曲线。

![预训练和编码器训练](04-预训练与继续训练编码器.png)

[全部逐条结果、其他模型及耗时：详细附录](../charts-v1/README.md)
'''
(OUT/'README.md').write_text(text, encoding='utf-8')
print(json.dumps({'figures': 4, 'matched_baseline_pairs': 4, 'imbalance_pairs': 6,
                  'mechanism_repeats': len(coverage['04-预训练与继续训练编码器']),
                  'source_unchanged': True}, ensure_ascii=False))

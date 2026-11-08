"""Derive robustness analysis and integrated conclusions from saved predictions."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imbalance as exp
from pipeline import read_json,write_json

ROOT=exp.ROOT;OUT=exp.OUT;REPORT=ROOT/'report';FIG=REPORT/'figures-imbalance'
NAMES={'logistic_regression':'TF-IDF + LR','linear_svm':'TF-IDF + SVM','word2vec_svm':'Word2Vec + SVM','bilstm':'Word2Vec + BiLSTM','distilbert':'DistilBERT'}
METRICS=['accuracy','macro_precision','macro_recall','macro_f1','mcc','minority_recall','minority_f1','worst_class_recall','simulated_prior_weighted_accuracy']

def md(frame):
    # Keep report generation independent of optional pandas/tabulate versions.
    def cell(value):
        if pd.isna(value):return '—'
        if isinstance(value,(float,np.floating)):return f'{value:.4f}'
        return str(value).replace('|','/').replace('\n',' ')
    rows=[[str(c) for c in frame.columns],['---']*len(frame.columns)]+[[cell(x) for x in row] for row in frame.itertuples(index=False,name=None)]
    return '\n'.join('| '+' | '.join(row)+' |' for row in rows)
def bootstrap(predictions):
    result=[];rng=np.random.default_rng(2026)
    for name in exp.MODELS:
        f=predictions[(predictions.model==name)&~predictions.weighted]
        if name in ['bilstm','distilbert']:f=f[f.seed==42]
        b=f[f.condition=='balanced'].set_index('sample_id');s=f[f.condition=='severe'].set_index('sample_id').loc[b.index]
        assert (b.label==s.label).all();y=b.label.to_numpy();bp=b.prediction.to_numpy();sp=s.prediction.to_numpy()
        groups=[np.flatnonzero(y==c) for c in range(4)];deltas=[]
        for _ in range(2000):
            ids=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            deltas.append(f1_score(y[ids],bp[ids],average='macro',zero_division=0)-f1_score(y[ids],sp[ids],average='macro',zero_division=0))
        result.append({'model':name,'degradation_macro_f1':float(f1_score(y,bp,average='macro')-f1_score(y,sp,average='macro')),
            'ci95_low':float(np.quantile(deltas,.025)),'ci95_high':float(np.quantile(deltas,.975))})
    write_json(OUT/'bootstrap.json',{'rows':result,'replicates':2000,'seed':2026,'contrast':'Balanced minus severe, fixed traditional fit or deep seed42',
        'scope':'Paired, class-stratified bootstrap on 1000 fixed test examples. Conditional on trained checkpoints, no training/sampling uncertainty or multiple comparison correction.'})
    return result

def main():
    FIG.mkdir(parents=True,exist_ok=True);obj=read_json(OUT/'evaluation.json');f=pd.read_csv(OUT/'metrics.csv');pred=pd.read_csv(OUT/'predictions.csv')
    main=f[~f.weighted];summary=main.groupby(['condition','model'])[METRICS].mean().reset_index();sd=main.groupby(['condition','model']).macro_f1.std().rename('macro_f1_training_sd').reset_index()
    summary=summary.merge(sd,on=['condition','model']);base=summary[summary.condition=='balanced'].set_index('model')
    summary['macro_f1_drop_pp']=[100*(base.loc[r.model,'macro_f1']-r.macro_f1) for r in summary.itertuples()]
    summary['minority_recall_drop_pp']=[100*(base.loc[r.model,'minority_recall']-r.minority_recall) for r in summary.itertuples()]
    paired_sd=[]
    for r in summary.itertuples():
        if r.model in ['bilstm','distilbert']:
            b=main[(main.condition=='balanced')&(main.model==r.model)].set_index('seed').macro_f1
            c=main[(main.condition==r.condition)&(main.model==r.model)].set_index('seed').macro_f1
            paired_sd.append(float((100*(b-c)).std()))
        else:paired_sd.append(float('nan'))
    summary['macro_f1_drop_training_sd_pp']=paired_sd
    summary.to_csv(OUT/'comparison.csv',index=False)
    severe=summary[summary.condition=='severe'].set_index('model').loc[exp.MODELS].reset_index()
    mitigation=f[f.weighted].groupby('model')[METRICS].mean().reset_index()
    mitigation['macro_f1_gain_pp']=[100*(r.macro_f1-severe.set_index('model').loc[r.model,'macro_f1']) for r in mitigation.itertuples()]
    mitigation['minority_recall_gain_pp']=[100*(r.minority_recall-severe.set_index('model').loc[r.model,'minority_recall']) for r in mitigation.itertuples()]
    mitigation.to_csv(OUT/'mitigation.csv',index=False)
    class_rows=[]
    for r in obj['rows']:
        for label in exp.LABELS:class_rows.append({k:r[k] for k in ['condition','model','seed','weighted']}|{'class':label,**r['classification_report'][label]})
    class_frame=pd.DataFrame(class_rows);class_frame.to_csv(OUT/'class_metrics.csv',index=False)
    class_recall=class_frame[(class_frame.condition=='severe')&~class_frame.weighted].pivot_table(index='model',columns='class',values='recall').reindex(index=exp.MODELS,columns=exp.LABELS)
    class_recall.index=class_recall.index.map(NAMES);class_recall.index.name='Model'
    class_recall_text=md(class_recall.reset_index())
    boot=bootstrap(pred)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    conditions=list(exp.CFG['conditions']);colors=['#9aa8a0','#668477','#45605c','#56849a','#157e75']
    for metric,title,ylabel in [('macro_f1','Fixed-test Macro F1 under training imbalance','Macro F1 (0–1)'),('minority_recall','Minority-class recall under training imbalance','Mean recall: Sports / Business / Sci-Tech (0–1)')]:
        fig,ax=plt.subplots(figsize=(10.5,5.8))
        for name,color in zip(exp.MODELS,colors):
            g=summary[summary.model==name].set_index('condition').loc[conditions]
            ax.plot(range(4),g[metric],marker='o',color=color,label=NAMES[name],linewidth=2)
            if name in ['bilstm','distilbert']:
                spread=main[main.model==name].groupby('condition')[metric].std().reindex(conditions).to_numpy()
                ax.fill_between(range(4),g[metric].to_numpy()-spread,g[metric].to_numpy()+spread,color=color,alpha=.12)
        ax.set(title=title,xlabel='Training ratio: majority : each minority class',ylabel=ylabel,ylim=(0,1),xticks=range(4),xticklabels=['1:1\n25/25/25/25%','7:1\n70/10/10/10%','17:1\n85/5/5/5%','47:1\n94/2/2/2%'])
        ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',fontsize=10)
        fig.text(.08,.015,'Source: results-imbalance/evaluation.json. Train N=6,000, fixed balanced test N=1,000. Bands: deep training SD (3 seeds).',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,1])
        for suffix in ['png','pdf']:fig.savefig(FIG/f'{metric}.{suffix}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,(name,weighted,title) in zip(axes,[('linear_svm',False,'SVM, severe'),('distilbert',False,'DistilBERT, severe'),('distilbert',True,'DistilBERT, weighted severe')]):
        r=next(r for r in obj['rows'] if r['condition']=='severe' and r['model']==name and r['weighted']==weighted and (r['seed'] in [None,42]))
        matrix=np.asarray(r['confusion_matrix']);ax.imshow(matrix/250,vmin=0,vmax=1,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>125 else '#173c35')
        ax.set(title=title,xlabel='Predicted class',ylabel='True class',xticks=range(4),yticks=range(4),xticklabels=['World','Sports','Business','Sci/Tech'],yticklabels=['World','Sports','Business','Sci/Tech']);ax.tick_params(axis='x',rotation=30)
    fig.suptitle('Severe imbalance: confusion counts on the same balanced test');fig.tight_layout(rect=[0,.08,1,.93]);fig.text(.04,.015,'Source: saved predictions. Each true class N=250. Deep checkpoints use seed42. Color indicates row-normalized proportion.',fontsize=10)
    for suffix in ['png','pdf']:fig.savefig(FIG/f'confusions.{suffix}',dpi=180)
    plt.close(fig)
    counts=pd.DataFrame([{'Condition':c,**dict(zip(exp.LABELS,n)),'Total':sum(n),'Ratio':f'{n[0]/n[1]:g}:1'} for c,n in exp.CFG['conditions'].items()])
    readable={'model':'Model','macro_f1':'Severe Macro F1','macro_f1_training_sd':'F1 SD','macro_f1_drop_pp':'F1 drop (pp)',
        'macro_f1_drop_training_sd_pp':'Drop SD (pp)','minority_recall':'Minority Recall','minority_recall_drop_pp':'Recall drop (pp)',
        'mcc':'MCC','macro_f1_gain_pp':'F1 gain (pp)','minority_recall_gain_pp':'Recall gain (pp)'}
    result_text=md(severe[['model','macro_f1','macro_f1_training_sd','macro_f1_drop_pp','macro_f1_drop_training_sd_pp','minority_recall','minority_recall_drop_pp','mcc']].replace({'model':NAMES}).rename(columns=readable))
    mitigation_text=md(mitigation[['model','macro_f1','macro_f1_gain_pp','minority_recall','minority_recall_gain_pp']].replace({'model':NAMES}).rename(columns={**readable,'macro_f1':'Class-weighted model Macro F1'}))
    largest=severe.loc[severe.macro_f1_drop_pp.idxmax()];best_severe=severe.loc[severe.macro_f1.idxmax()]
    interpretation=(f"The largest Macro-F1 deterioration is {NAMES[largest['model']]} ({largest['macro_f1_drop_pp']:.2f} points). "
        f"The highest absolute severe-condition Macro F1 is {NAMES[best_severe['model']]} ({best_severe['macro_f1']:.4f}). "
        "These are different comparisons. Minority scarcity can weaken feature estimation and class-boundary learning, while changed priors encourage majority predictions. "
        "The experiment does not isolate these mechanisms or architecture from pretraining resources. Depth alone does not determine robustness.")
    report=f'''# Controlled class-imbalance robustness experiment

## Question and protocol

How do traditional and deep text classifiers deteriorate when minority labels become scarce, at a fixed total training budget? This is an exploratory extension requested after the original AG News test results were inspected. New ratios, World as the majority class, frozen hyperparameters and severe-only mitigation were specified in `config-imbalance.json` and `results-imbalance/design.freeze.json` before the new training runs.

{md(counts)}

Every condition has **6,000 unique training texts**. Validation and test reuse the exact balanced AG News splits, **1,000 texts each, 250 per class**. The balanced reference reuses the original checkpoints and saved predictions. Increasing World examples come from the eligible official training pool. Minority examples are nested subsets of the balanced training split. No repeated/oversampled texts and no normalized exact-text overlap with validation/test are allowed. Official test texts take precedence during deduplication, so even unsampled official test texts cannot enter training. Only one sampling realization and one majority-class orientation are tested.

Five families: TF-IDF + LR, TF-IDF + SVM, training-only Word2Vec mean + SVM, Word2Vec-initialized BiLSTM and generic pretrained DistilBERT. All classifiers retain hyperparameters chosen on balanced AG News validation. Each new feature extractor fits its own condition's training texts. BiLSTM and DistilBERT each use seeds 42/43/44, the original 5/3 epoch budgets and validation Macro F1 checkpoint selection. The identical subsets across seeds isolate training randomness, not uncertainty in the sampling design. Word vectors use 100 dimensions, SGNS, ten epochs and one worker. DistilBERT benefits from external pretraining, while local word vectors do not.

## Metrics and deterioration

Primary metrics use the fixed balanced test: Macro Precision/Recall/F1, Accuracy, MCC, mean minority Recall/F1 across Sports/Business/Sci-Tech, worst-class Recall and per-class scores. Balanced Accuracy equals Macro Recall here. Because test supports are equal, weighted and macro F1 coincide, and Accuracy equals balanced Accuracy. Reporting both is for clarity, not independent evidence. A positive deterioration equals `100 × (balanced score − condition score)` in percentage points. Negative changes remain visible. Deep summaries average three seeds and show sample training SD.

Severe condition compared with each model's own balanced reference:

{result_text}

{interpretation}

Mean per-class Recall at severe imbalance (each true class has 250 test examples):

{class_recall_text}

![Macro F1 decline](figures-imbalance/macro_f1.png)

![Minority recall decline](figures-imbalance/minority_recall.png)

Full condition/metric results: `results-imbalance/comparison.csv`; all 40 run records: `metrics.csv`; all 40,000 predictions: `predictions.csv`; per-class results: `class_metrics.csv`. The 40 records comprise nine reused balanced fits, 27 new unweighted fits and four severe weighted fits. F1 SD is the severe-score training SD; Drop SD pairs the same three training seeds between balanced and severe. Neither is a test confidence interval. Fixed traditional fits have no training SD.

## Predeclared class-weight mitigation

At severe imbalance only, TF-IDF SVM uses `class_weight='balanced'`, and DistilBERT retrains from the generic checkpoint with inverse-frequency cross-entropy weights `N / (4 n_c)`, approximately [0.2660, 12.5, 12.5, 12.5]. Same data, learning rate, epochs and validation criterion. This compares two representative models, without searching weights against test results. Positive gain means weighted minus unweighted severe score.

{mitigation_text}

![Severe confusion matrices](figures-imbalance/confusions.png)

## Conditional uncertainty

{md(pd.DataFrame(boot))}

These are 2,000 paired class-stratified bootstrap replicates on the fixed 1,000 test examples, balanced minus severe. Deep contrasts use seed42 only. Intervals condition on the trained checkpoints and do not cover training randomness, alternative class orientations or subset draws. No multiple-comparison correction. Three-seed SD answers a different uncertainty question.

## Accuracy under a changed deployment prior

The main test remains balanced. A **separate simulated prior-weighted Accuracy** reweights these same predictions by desired training prior / observed test prior. This assumes label shift with unchanged class-conditional text distributions. It is not an additional observed test set. At a 94/2/2/2 prior, an always-World rule has simulated Accuracy 94%, while its fixed balanced-test Accuracy is 25%, Macro F1 is 0.10 and minority Recall is zero. This mathematical diagnostic explains why aggregate Accuracy alone can hide minority failures. Compare observed fixed-test scores separately from this simulation.

## Limits

This experiment measures the joint effect of changed priors and minority scarcity at fixed N. Text composition also changes. Condition-specific features and validation checkpoint selection are part of the pipeline. One corpus, World as majority, a fixed sampling realization and small minority counts limit generalization. Pretraining resources differ across model families. Results cannot establish a universal robustness ordering for traditional ML versus all deep learning. The imbalance experiment introduces no new latency benchmark: architecture and tokenizer are unchanged, and the earlier common-workload CPU measurements remain the latency evidence.

## Sources

- AG News source and immutable revisions: `assets.lock.json`, `data-v2/audit.json`, `data-imbalance/audit.json`.
- [scikit-learn Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html): mean recall across classes.
- [scikit-learn class weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html): inverse-frequency balanced weighting.
- Local package versions and original protocols: README and existing experiment reports. Sources describe definitions; all scores above come from local saved predictions.
'''
    (REPORT/'imbalance-report.md').write_text(report,encoding='utf-8')
    v2=pd.read_csv(ROOT/'results-v2'/'metrics.csv');v2=v2[v2.source==v2.target].groupby(['source','model']).macro_f1.mean()
    hard=pd.read_csv(ROOT/'results-hard'/'metrics.csv');hard=hard[hard.model.isin(exp.MODELS)].groupby(['target','model']).macro_f1.mean()
    text=['# 文本分类项目：最终实验主线与结论','',
        '本项目比较五种模型，并通过常规文本分类、复杂句对推理和训练类别不平衡三个实验回答不同问题。以下结论来自真实运行结果，不把预期结论当作实验结果。','',
        '## 1. 常规任务：准确性与推理成本的取舍','',
        'IMDb 和 Yelp 上，TF-IDF + LR 与 DistilBERT 的 Macro-F1 相差不到 1 个百分点；Rotten Tomatoes 和 AG News 的差距更明显。因此建议写为“部分常规任务性能接近，差距依赖数据集”，而不是“所有简单任务都一样”。同一批评论输入、四个 CPU 线程和 batch=1 的已有测试显示传统模型推理更快。新闻检查点的该延迟测试仍使用评论输入，不能直接当成新闻生产流量的延迟。','']
    for source in ['imdb','rotten_tomatoes','yelp','ag_news']:
        a=v2.loc[(source,'logistic_regression')];b=v2.loc[(source,'distilbert')];text.append(f'- {source}: TF-IDF + LR {a:.4f}，DistilBERT {b:.4f}，差距 {(b-a)*100:.2f} 个百分点。')
    text+=['','## 2. MultiNLI：复杂语义任务中的 Transformer 优势','',
        '在 6,000 个训练句对的预算下，DistilBERT 在 matched/mismatched 两个保留评估集上的 Macro-F1 均高于所测传统基线和 BiLSTM；预先指定的 seed42 对 SVM 比较区间也均为正。这里支持“预训练 Transformer 在此复杂任务和预算下更好”，不支持把所有深度学习都视为同样有效。该任务评估使用官方 validation 子集，不是隐藏测试榜单；预训练资源也不同。','']
    for split in ['mnli_matched','mnli_mismatched']:
        text.append(f'- {split}: 配对 TF-IDF + SVM {hard.loc[(split,"linear_svm")]:.4f}，DistilBERT {hard.loc[(split,"distilbert")]:.4f}。')
    text+=['','## 3. 类别不平衡：比较各模型自身的衰退','',
        '训练总量固定为 6,000，从每类 1,500 条逐步变为 World 5,640 条、其他每类 120 条；验证和测试始终每类 250 条。主要看 Macro-F1 和三个少数类的平均 Recall，并相对各自均衡基线计算下降百分点。','']
    for r in severe.itertuples():text.append(f'- {NAMES[r.model]}：严重不平衡 Macro-F1 {r.macro_f1:.4f}，下降 {r.macro_f1_drop_pp:.2f} 个百分点；少数类 Recall {r.minority_recall:.4f}，下降 {r.minority_recall_drop_pp:.2f} 个百分点。')
    text+=['','类别加权补救实验：','']
    for r in mitigation.itertuples():text.append(f'- {NAMES[r.model]}：Macro-F1 变化 {r.macro_f1_gain_pp:+.2f} 个百分点，少数类 Recall 变化 {r.minority_recall_gain_pp:+.2f} 个百分点。')
    text+=['',f'本次下降最多的是 {NAMES[largest["model"]]}，而严重条件下绝对Macro-F1最高的是 {NAMES[best_severe["model"]]}。少数类样本变少会影响特征和分类边界的学习，先验改变也会鼓励多数类预测；本实验没有把这些机制逐一分离，也不能把预训练资源差异完全归于网络架构。','',
        '因此最终结论应把绝对能力、推理成本和分布变化下的鲁棒性分别讨论。少数类召回率下降不能被总体准确率掩盖，也不能由单次实验概括为某一大类方法永远更稳健。','',
        '为说明准确率的局限，另提供标签先验变化的加权模拟：94% World 的部署分布下，只预测 World 就能得到 94% 模拟准确率，但均衡测试的 Macro-F1 只有 0.10，少数类 Recall 为 0。该 94% 是数学示例，不是模型实测结果。','',
        '英文界面提供不平衡实验页，完整英文报告为 imbalance-report.md。所有指标、图和预测保存在 results-imbalance；不平衡子集没有与验证/测试重叠。']
    (REPORT/'final-synthesis-zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    slide={'summary':json.loads(summary.to_json(orient='records')),'severe':json.loads(severe.to_json(orient='records')),
        'mitigation':json.loads(mitigation.to_json(orient='records')),'bootstrap':boot,'counts':exp.CFG['conditions'],
        'normal':[{'source':a,'lr':float(v2.loc[(a,'logistic_regression')]),'distilbert':float(v2.loc[(a,'distilbert')])} for a in ['imdb','rotten_tomatoes','yelp','ag_news']],
        'hard':[{'target':a,**{m:float(hard.loc[(a,m)]) for m in exp.MODELS}} for a in ['mnli_matched','mnli_mismatched']]}
    write_json(REPORT/'slide-data-final.json',slide)
    normal_rows=[]
    for source in ['imdb','rotten_tomatoes','yelp','ag_news']:
        normal_rows.append({'Task':source,**{NAMES[m]:v2.loc[(source,m)] for m in exp.MODELS}})
    cost=[{k:r[k] for k in ['model','median_ms','p95_ms']} for r in read_json(ROOT/'results-v2'/'benchmark.json')['rows'] if r['source']=='imdb']
    hard_rows=[{'Evaluation':r['target'],**{NAMES[m]:r[m] for m in exp.MODELS}} for r in slide['hard']]
    integrated=f'''# Traditional NLP and deep learning for text classification

## Research question

When do traditional features with linear classifiers offer sufficient text classification performance, when does a pretrained Transformer provide a larger advantage, and how does training class imbalance change each model's behavior? The project separates predictive quality, inference cost and robustness under shifted training priors.

## Methods and common controls

Traditional methods are TF-IDF + Logistic Regression, TF-IDF + Linear SVM and local Word2Vec mean features + Linear SVM. Deep methods are Word2Vec-initialized BiLSTM and fine-tuned generic DistilBERT with a new classification head. DistilBERT is a Transformer in the BERT family. The experiment does not include every method listed as an option in the assignment. CRF models structured output sequences and does not match the document/pair labels here.

Each conventional task uses 6,000 training, 1,000 validation and 1,000 test texts. MultiNLI uses 6,000 training pairs, 999 source-validation pairs and 999 evaluation pairs for each genre condition. Model selection uses validation Macro F1 only. Deep results average seeds 42/43/44. Traditional methods use fixed fits. Feature extraction fits training text only. Saved protocols, sample manifests and checkpoint hashes support reproducibility. DistilBERT's external pretraining differs from training local word embeddings and limits attribution to architecture alone.

## Experiment 1: conventional tasks and inference cost

Macro F1, deep values averaged over three seeds:

{md(pd.DataFrame(normal_rows))}

The strongest traditional baseline is close to DistilBERT on IMDb and Yelp, with gaps below one percentage point. Rotten Tomatoes and AG News show larger gaps, around four to five points. Thus, **similar performance is task-dependent**, rather than a property of all conventional text classification. A BiLSTM does not automatically beat sparse features.

The following earlier CPU benchmark uses IMDb-trained checkpoints on the same 200 mixed review texts, batch1, four CPU threads, five repeats. Cleaning, feature conversion/tokenization and forward prediction are included. Loading is excluded. Values are milliseconds per text. These are common-workload timings from the original experiment, not new measurements concurrent with robustness training.

{md(pd.DataFrame(cost))}

Traditional classifiers offer substantially lower latency on this workload. Hardware, input lengths, batching and implementations affect the tradeoff. The full report retains p95 values, model sizes and selection details. News-trained models were also timed on the same review workload, so those values should not be described as news production latency.

## Experiment 2: MultiNLI sentence-pair semantics

Three labels describe entailment, neutral and contradiction. Pair-aware traditional features concatenate premise, hypothesis, absolute difference and elementwise product; a shared BiLSTM uses corresponding interactions. DistilBERT jointly encodes both sentences. Normalized premise groups stay disjoint across partitions. Reserved evaluation pools come from official validation_matched and validation_mismatched, not the hidden test leaderboard.

{md(pd.DataFrame(hard_rows))}

DistilBERT exceeds paired SVM by about 16 Macro-F1 points in both sampled genre conditions. A predeclared seed42 premise-cluster bootstrap gives positive 95% intervals for both differences; see `hard-benchmark-report.md` for exact contrasts and uncertainty limits. Hypothesis-only SVM obtains above-chance Accuracy, so annotation cues remain relevant. These results support the advantage of this pretrained Transformer under this budget, without proving universal deep-learning superiority or pure logical reasoning. Mismatched performance is slightly higher here; genre shift does not guarantee a decrease on every sampled evaluation.

## Experiment 3: fixed-budget class imbalance

Keep the four-class AG News training total at 6,000, but increase World support from 25% to 70%, 85% and 94%. Minority supports decrease from 1,500 each to 600, 300 and 120. Keep the same balanced validation/test sets and freeze the balanced-validation-selected hyperparameters. Refit features only on the corresponding condition. The severe comparison follows; positive decline means balanced minus severe.

{result_text}

{interpretation}

![Training imbalance and Macro F1](figures-imbalance/macro_f1.png)

![Training imbalance and minority recall](figures-imbalance/minority_recall.png)

The experiment compares deterioration relative to each model's own balanced baseline as well as absolute severe-condition quality. Both matter: a small drop from a weaker baseline does not establish the strongest classifier. The minority metrics and per-class confusion reveal failures that aggregate Accuracy can hide under an imbalanced deployment prior.

Predeclared inverse-frequency class weights at severe imbalance:

{mitigation_text}

See `imbalance-report.md` for full protocol, per-class scores, three-seed SD, conditional bootstrap intervals and changed-prior Accuracy simulation. The simulation is not an additional observed test set. One majority-class orientation and one subset realization limit generalization. Minority scarcity and changed priors remain combined even at fixed training N.

## Final interpretation

1. Conventional tasks can favor a low-latency traditional classifier when its predictive performance is sufficiently close. IMDb/Yelp support this practical tradeoff; other corpora show larger quality gains from DistilBERT.
2. Complex sentence-pair semantics reveal a larger advantage for the tested pretrained Transformer, while the tested BiLSTM does not show the same improvement.
3. Training imbalance is a separate robustness dimension. Report Macro F1 deterioration together with minority Recall and class-specific errors, then assess a predeclared mitigation. Do not infer a universal robustness ordering from a single corpus and majority class.

All empirical statements concern the local fixed-budget experiments. The English Streamlit application provides prediction, multi-metric comparisons, MultiNLI pair inputs and an imbalance robustness tab. Prediction demos retain balanced checkpoints. Previous reports remain historical detailed records; this document is the integrated reading order.

## Evidence and references

- Local detailed reports: `classification-report-v2.md`, `hard-benchmark-report.md`, `imbalance-report.md`.
- Exact model predictions and metadata: `results-v2`, `results-hard`, `results-imbalance`; source revisions: `assets.lock.json`.
- [Williams et al. (2018), MultiNLI](https://aclanthology.org/N18-1101/).
- [Gururangan et al. (2018), annotation artifacts](https://aclanthology.org/N18-2017/).
- [DistilBERT](https://arxiv.org/abs/1910.01108).
- [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [class weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).

Member contributions, personal signatures and a real narrated video remain for actual members, as deferred by the user. This project does not claim those human activities are complete.
'''
    (REPORT/'final-report.md').write_text(integrated,encoding='utf-8')
    print('Generated robustness report, final synthesis, figures and slide data.')
if __name__=='__main__':main()

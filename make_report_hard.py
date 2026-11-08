"""Actual MultiNLI results, cross-genre evaluation and a hypothesis-only diagnostic."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hard_benchmark as hard
from pipeline import read_json,write_json
ROOT=hard.ROOT;OUT=hard.OUT;REPORT=ROOT/'report';FIGS=REPORT/'figures-hard'
SHORT={'logistic_regression':'Pair TF-IDF + LR','linear_svm':'Pair TF-IDF + SVM','word2vec_svm':'Pair Word2Vec + SVM','bilstm':'Shared BiLSTM','distilbert':'DistilBERT','hypothesis_only_svm':'Hypothesis-only SVM'}
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)+'\n'
def save(fig,name):
    fig.savefig(FIGS/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(FIGS/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
def main():
    evaluation=read_json(OUT/'evaluation.json');hard.check(evaluation)
    bench=read_json(OUT/'benchmark.json');hard.check(bench)
    audit=read_json(hard.DATA/'audit.json');bootstrap=read_json(OUT/'bootstrap.json')
    f=pd.read_csv(OUT/'metrics.csv');cols=['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','mcc']
    grouped=f.groupby(['target','model'])[cols].mean();sd=f.groupby(['target','model']).macro_f1.std()
    FIGS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#bdc5c0','#899a91','#596f66','#4e7788','#1b766e']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axes,['matched','mismatched']):
        rows=grouped.loc['mnli_'+split].reindex(hard.MODELS)
        bars=ax.barh(range(5),rows.macro_f1,color=colors)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in rows.macro_f1],padding=4)
        ax.set(yticks=range(5),yticklabels=[SHORT[n] for n in hard.MODELS],xlim=(0,1),xlabel='Macro F1 (0–1)',title=split.title()+' genres');ax.invert_yaxis()
    fig.suptitle('MultiNLI: sentence-pair inference under a fixed training budget',fontsize=15)
    fig.text(.08,.02,'Source: frozen MultiNLI course experiment. 6,000 train / 999 source validation; 999 pairs per evaluation condition. Deep models: 3 seeds.',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.93]);save(fig,'mnli_comparison')
    old=pd.read_csv(REPORT/'comparison-v2.csv')
    fig,ax=plt.subplots(figsize=(13,5))
    x=np.arange(6);width=.15
    labels=['IMDb','Rotten\nTomatoes','Yelp','AG News','MultiNLI\nmatched','MultiNLI\nmismatched']
    for i,name in enumerate(hard.MODELS):
        values=[float(old[(old.source==source)&(old.model==name)].macro_f1.iloc[0]) for source in ['imdb','rotten_tomatoes','yelp','ag_news']]
        values.extend(grouped.loc[('mnli_'+split,name),'macro_f1'] for split in ['matched','mismatched'])
        ax.bar(x+(i-2)*width,values,width,color=colors[i],label=SHORT[name].replace('Pair ','').replace('Shared ',''))
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.02),ylabel='Macro F1 (0–1)',xlabel='Evaluation condition',title='Topic/sentiment classification and a harder semantic inference task')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.33)
    fig.text(.08,.015,'Source: V2 + MultiNLI results. MultiNLI uses paired variants; other tasks use single-text variants. Score changes do not isolate one cause.',fontsize=9)
    save(fig,'difficulty_context')
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for i,split in enumerate(['matched','mismatched']):
        for j,name in enumerate(['linear_svm','distilbert']):
            row=next(r for r in evaluation['rows'] if r['target']=='mnli_'+split and r['model']==name and (r['seed']==42 or r['seed'] is None))
            matrix=np.asarray(row['confusion_matrix']);ax=axes[i,j];ax.imshow(matrix,cmap='Blues',vmin=0,vmax=333)
            for a in range(3):
                for b in range(3):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if matrix[a,b]>165 else '#21383d')
            ax.set(xticks=range(3),xticklabels=['Entail','Neutral','Contradict'],yticks=range(3),yticklabels=['Entail','Neutral','Contradict'],xlabel='Predicted label',ylabel='True label',title=split.title()+': '+SHORT[name])
    fig.text(.1,.015,'Source: reserved MultiNLI evaluation pairs. DistilBERT uses seed42; each true class has 333 examples.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);save(fig,'mnli_confusions')
    scores=[];ci_rows=[];class_rows=[];cost_rows=[];diagnostics=[]
    for split in ['matched','mismatched']:
        for name in hard.MODELS:
            row=grouped.loc[('mnli_'+split,name)]
            scores.append([split,SHORT[name]]+[f'{row[c]:.4f}' for c in cols[:5]])
            selected=[r for r in evaluation['rows'] if r['target']=='mnli_'+split and r['model']==name]
            for label in hard.LABELS:
                reports=[r['classification_report'][label] for r in selected]
                class_rows.append([split,SHORT[name],label]+[f'{np.mean([r[k] for r in reports]):.4f}' for k in ['precision','recall','f1-score']]+[333])
        b=bootstrap[split];ci_rows.append([split,f'{b["delta_macro_f1"]:+.4f}',f'[{b["ci95_low"]:+.4f}, {b["ci95_high"]:+.4f}]'])
        diag=grouped.loc[('mnli_'+split,'hypothesis_only_svm')]
        full=grouped.loc[('mnli_'+split,'linear_svm')]
        diagnostics.append([split,f'{diag.accuracy:.4f}',f'{diag.macro_f1:.4f}',f'{full.macro_f1:.4f}'])
    for row in bench['rows']:cost_rows.append([SHORT[row['model']],f'{row["median_ms"]:.2f}',f'{row["p95_ms"]:.2f}'])
    results=grouped.reset_index();results['macro_f1_sd']=[None if pd.isna(sd.loc[(r.target,r.model)]) else float(sd.loc[(r.target,r.model)]) for _,r in results.iterrows()]
    results.to_csv(REPORT/'comparison-hard.csv',index=False)
    pd.DataFrame(class_rows,columns=['condition','model','class','precision','recall','f1','support']).to_csv(OUT/'class_metrics.csv',index=False)
    # Deterministic, non-random inspection queue, explicitly not human-reviewed.
    predictions=pd.read_csv(OUT/'predictions.csv');queue=[]
    for split in ['matched','mismatched']:
        data=hard.load_split(split).set_index('sample_id')
        svm=predictions[(predictions.split==split)&(predictions.model=='linear_svm')].set_index('sample_id')
        bert=predictions[(predictions.split==split)&(predictions.model=='distilbert')&(predictions.seed==42)].set_index('sample_id')
        selected=data[(svm.loc[data.index].prediction.to_numpy()!=data.label.to_numpy()) | (bert.loc[data.index].prediction.to_numpy()!=data.label.to_numpy())].sample(12,random_state=2026)
        for idx,row in selected.iterrows():queue.append({'sample_id':idx,'split':split,'premise':row.premise,'hypothesis':row.hypothesis,'genre':row.genre,'label':int(row.label),'svm':int(svm.loc[idx,'prediction']),'distilbert':int(bert.loc[idx,'prediction']),'review_status':'unreviewed','explanation':''})
    if not (OUT/'error_review.csv').exists():pd.DataFrame(queue).to_csv(OUT/'error_review.csv',index=False,encoding='utf-8-sig')
    findings=[]
    for split in ['matched','mismatched']:
        target='mnli_'+split;svm=grouped.loc[(target,'linear_svm')];bert=grouped.loc[(target,'distilbert')]
        findings.append(f'- {split.title()}: pair SVM Macro F1={svm.macro_f1:.4f}; DistilBERT={bert.macro_f1:.4f}; mean-score difference={(bert.macro_f1-svm.macro_f1)*100:+.2f} percentage points. This mean difference is distinct from the seed42 bootstrap contrast below.')
    report=f'''# MultiNLI: a harder benchmark for sentence-pair semantics

## Why this benchmark adds difficulty

The original tasks classify the sentiment or topic of one text. MultiNLI asks whether a hypothesis follows from, contradicts, or is neither established nor contradicted by a premise. Its matched and mismatched genre settings add an explicit generalization comparison. The original paper describes a broad-coverage inference challenge with diverse spoken and written genres; this is a different semantic task, not simply an increase in class count. We selected it before observing its final scores and did not replace it based on the resulting model gap. [Williams et al.,2018](https://aclanthology.org/N18-1101/)

## Fixed protocol and honest benchmark scope

We use6,000 training pairs,999 validation pairs sampled from the official training pool, and999 pairs each from `validation_matched` and `validation_mismatched`, balanced at333 examples per evaluation class. These reserved official validation samples serve as evaluation pools for the course experiment. They are **not official hidden-test leaderboard results**, full-data replication or a new state of the art. Label IDs follow the source data:0 entailment,1 neutral,2 contradiction. [Dataset card](https://huggingface.co/datasets/nyu-mll/multi_nli)

Training-only source validation selects hyperparameters and epochs. Before final evaluation, `results-hard/protocol.freeze.json` records nine main-model checkpoints and one hypothesis-only diagnostic. The original V1/V2 records remain unchanged. Matched/mismatched samples were fixed in advance, not chosen for favorable results.

### Leakage and genre controls

Exact pairs with contradictory labels are removed. Normalized premises are grouped across partitions, so a premise cannot appear in both train and validation/evaluation even with a different hypothesis. Official evaluation pools take precedence over training during grouping. Genre, pairID and parsing metadata are never model inputs. `data-hard/audit.json` records raw counts, genre distributions, group counts, hashes and truncation. Grouping does not rule out paraphrase or pretraining overlap.

{table(['Split','Pairs','Unique premises','Median paired subwords','Truncated at256'],[[s,v['n'],v['unique_premises'],v['median_pair_tokens'],f'{v["fraction_exceeding_256_subwords"]:.2%}'] for s,v in audit['splits'].items()])}

## Comparable pair-aware models

The five families remain TF-IDF+LR, TF-IDF+SVM, Word2Vec+SVM, BiLSTM and DistilBERT. Traditional methods receive both sentences and interaction features: concatenate premise representation, hypothesis representation, their absolute difference and elementwise product. TF-IDF fits a shared training-only vocabulary and searches two n-gram choices and C in0.1/1/10. Word2Vec uses100D locally trained vectors; its mean-vector features use StandardScaler and the same C choices. These are stronger pair baselines than simple unmarked text concatenation.

The shared BiLSTM uses one encoder for both sentences,100D trainable Word2Vec initialization and64 units in each direction. It concatenates the two sentence states, absolute difference and product, followed by a small classification head. Seed42 compares learning rates0.001/0.003 over five epochs, then the selected learning rate runs seeds43/44. DistilBERT jointly tokenizes premise/hypothesis with separator tokens, compares2e-5/5e-5 over three epochs, then adds seeds43/44. It uses the generic pretrained base, not an NLI-fine-tuned checkpoint.

BiLSTM allows256 word tokens **per sentence**; DistilBERT allows256 subwords **for the combined pair**, with longest-first truncation. Sparse and mean-vector models read full cleaned sentences. These are different budgets, and external pretraining differs. Training is supervised from the same6,000 labeled pairs but this is not a causal isolation of architecture.

## Matched and mismatched results

{table(['Condition','Model','Accuracy','Macro Precision','Macro Recall','Macro F1','Weighted F1'],scores)}

![MultiNLI comparison](figures-hard/mnli_comparison.png)

{chr(10).join(findings)}

Deep values average three seeds; traditional fits are fixed. Full per-seed outputs, SD, MCC, Macro/Micro/Weighted P/R/F1 and class metrics are saved. Balanced supports make Macro and Weighted coincide; single-label Micro P/R/F1 equal Accuracy. These are not independent evidence. Uniform random guessing has expected Accuracy1/3, not a guaranteed observed F1.

### Conditional uncertainty

{table(['Condition','DistilBERT seed42 minus pair SVM, Delta Macro F1','95% premise-cluster interval'],ci_rows)}

We resample normalized premise groups1,000 times, preserving dependent hypotheses sharing a premise. The percentile intervals condition on two fixed models and exclude training uncertainty. They do not test every model pair or correct multiple comparisons. A matched-to-mismatched difference also reflects different samples and genre difficulty, so it is not an isolated causal domain-shift estimate. A higher mismatched score is possible and must be reported as measured.

### Hypothesis-only diagnostic

{table(['Condition','Hypothesis-only Accuracy','Hypothesis-only Macro F1','Pair SVM Macro F1'],diagnostics)}

A fixed TF-IDF SVM (C=1, unigram+bigram) is trained only on training hypotheses and does not see premises. It is a diagnostic, not a sixth family selected for the headline comparison. Scores above chance can reveal that premise-free surface cues carry predictive information; they do not establish that any particular artifact caused an individual decision. Prior work motivates this check. [Gururangan et al.,2018](https://aclanthology.org/N18-2017/)

## What the harder task shows

![Context across tasks](figures-hard/difficulty_context.png)

The inference task has a much lower performance range than the earlier topic/sentiment samples in this particular protocol. That observation supports including it as a harder empirical comparison. It does not prove that semantic reasoning alone causes the gap: the label definitions, class count, data distribution and paired input design differ. A larger Transformer-versus-SVM gap would support a task-dependent advantage under this budget, not a universal claim that deep learning always wins. A small or reversed gap would also be a valid result. The benchmark and search space were fixed before evaluation.

## Per-class behavior and computational cost

![Class confusions](figures-hard/mnli_confusions.png)

Per-class Precision/Recall/F1 and support are in `results-hard/class_metrics.csv` and the English GUI, along with count or normalized3x3 confusion matrices. A24-case sampled error-review queue retains both sentences, genre, gold label and model predictions; its explanations and human-review status remain unfilled.

{table(['Model','CPU median (ms/pair)','CPU p95 (ms/pair)'],cost_rows)}

All models process the same200 matched pairs with four CPU threads, batch1,20 warm-ups and five repetitions. Timing includes preprocessing and forward computation but excludes loading/download. This differs from the earlier review-text workload; do not compare those timings as though the inputs were identical.

## Reproduce and demonstrate

Run `python download_assets.py`, `python hard_benchmark.py all`, `python make_report_hard.py`, then `python -m unittest -v test_project test_extension test_hard`. The English GUI adds Natural language inference, separate premise/hypothesis fields, matched/mismatched evaluation selection, all existing metrics and the premise-free diagnostic. Demonstration models consistently use seed42.

This report supplements `classification-report-v2.md`. Member information remains deferred. Real narration, signatures and official submission are outside the computational experiment.
'''
    for a,b in [('et al.,2018','et al., 2018'),('use6,000','use 6,000'),('pairs,999','pairs, 999'),('and999','and 999'),('at333','at 333'),('data:0','data: 0'),(',1 neutral',', 1 neutral'),(',2 contradiction',', 2 contradiction'),('at256','at 256'),('in0.1','in 0.1'),('uses100D','uses 100D'),('sentences,100D','sentences, 100D'),('and64','and 64'),('rates0.001','rates 0.001'),('compares2e','compares 2e'),('allows256','allows 256'),('same6,000','same 6,000'),('Accuracy1/3','Accuracy 1/3'),('groups1,000','groups 1,000'),('normalized3x3','normalized 3x3'),('A24-case','A 24-case'),('same200','same 200'),('batch1,20','batch 1, 20')]:report=report.replace(a,b)
    (REPORT/'hard-benchmark-report.md').write_text(report,'utf-8')
    write_json(REPORT/'slide-data-hard.json',{'summary':results.fillna(0).to_dict('records'),'bootstrap':bootstrap,'labels':hard.LABELS,
        'confusions':{split:{name:next(r['confusion_matrix'] for r in evaluation['rows'] if r['target']=='mnli_'+split and r['model']==name and (r['seed']==42 or r['seed'] is None)) for name in ['linear_svm','distilbert']} for split in ['matched','mismatched']},
        'display_rows':scores,'costs':bench['rows'],'findings':findings,'diagnostics':diagnostics})
    cn='# MultiNLI 更难语义任务结果\n\n新增前提／假设双输入，预测蕴含、中立、矛盾。分别评估 matched 与 mismatched 体裁，传统模型也使用成对交互特征。\n\n'+table(['条件','模型','Accuracy','Macro P','Macro R','Macro F1','Weighted F1'],scores)+'\n![结果](figures-hard/mnli_comparison.png)\n\n'+ '\n'.join(findings)+'\n\n本实验为固定预算抽样：训练6,000对、验证999对、两种评估条件各999对。评估样本来自官方validation池，并非官方隐藏测试榜单。旧任务与新任务输入和标签不同，分差不能全部归因于某一个因素。完整说明见hard-benchmark-report.md。\n'
    (REPORT/'hard-summary-zh.md').write_text(cn,'utf-8')
    print('Hard-benchmark report, three scientific figures, class metrics and an error-review queue generated.')
if __name__=='__main__':main()

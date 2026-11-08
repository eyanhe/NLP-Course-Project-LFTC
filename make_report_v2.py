"""Expanded, evidence-backed report for five models and four datasets."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extension import ROOT,OUT,CFG,SOURCES,MODELS,LABELS,check
from pipeline import read_json,write_json

REPORT=ROOT/'report';FIGS=REPORT/'figures-v2'
SHORT={'logistic_regression':'TF-IDF + LR','linear_svm':'TF-IDF + SVM','word2vec_svm':'Word2Vec + SVM','bilstm':'Word2Vec + BiLSTM','distilbert':'DistilBERT'}
TITLES={'imdb':'IMDb (binary)','rotten_tomatoes':'Rotten Tomatoes (binary)','yelp':'Yelp (binary)','ag_news':'AG News (4 classes)'}
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'
def save(fig,name):
    fig.savefig(FIGS/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white')
    fig.savefig(FIGS/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    evaluation=read_json(OUT/'evaluation.json');check(evaluation)
    benchmark=read_json(OUT/'benchmark.json');check(benchmark)
    all_rows=evaluation['rows'];within=[r for r in all_rows if r['source']==r['target']]
    metrics=pd.DataFrame(within);cols=['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','micro_f1','mcc']
    summary=metrics.groupby(['source','model'])[cols].mean().reset_index()
    sd=metrics.groupby(['source','model']).macro_f1.std()
    summary['macro_f1_sd']=[None if pd.isna(sd.loc[(r.source,r.model)]) else sd.loc[(r.source,r.model)] for _,r in summary.iterrows()]
    summary.to_csv(REPORT/'comparison-v2.csv',index=False)
    FIGS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,source in zip(axes.flat,SOURCES):
        values=summary[summary.source==source].set_index('model').reindex(MODELS)
        y=np.arange(5)
        for i,(metric,label,color) in enumerate([('macro_precision','Macro Precision','#a7b2b5'),('macro_recall','Macro Recall','#526c77'),('macro_f1','Macro F1','#1b766e')]):
            ax.barh(y+(i-1)*.22,values[metric],height=.22,color=color,label=label)
        ax.set(yticks=y,yticklabels=[SHORT[n] for n in MODELS],xlim=(0,1),xlabel='Test score (0–1)',title=TITLES[source]);ax.invert_yaxis()
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.045))
    fig.text(.08,.015,'Source: frozen V2 test predictions. Deep models: mean of 3 seeds. Traditional models: fixed fits. Sample size: 1,000 per corpus.',fontsize=9)
    fig.tight_layout(rect=[0,.11,1,1]);save(fig,'multiple_metrics')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    for ax,name in zip(axes,['linear_svm','bilstm','distilbert']):
        r=next(r for r in within if r['source']=='ag_news' and r['model']==name and (r['seed']==42 or r['seed'] is None))
        matrix=np.asarray(r['confusion_matrix']);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=250)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>125 else '#21383d')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS['ag_news'],yticklabels=LABELS['ag_news'],xlabel='Predicted class',ylabel='True class',title=SHORT[name])
        ax.tick_params(axis='x',rotation=30)
    fig.suptitle('AG News four-class confusion matrices (counts)',fontsize=15)
    fig.text(.08,.015,'Source: frozen test predictions. Deep matrices: seed42. Each true class has 250 examples.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.94]);save(fig,'news_confusion')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axes,['bilstm','distilbert']):
        rows=[r for r in within if r['source']=='ag_news' and r['model']==name];v=[]
        for label in LABELS['ag_news']:
            v.append([np.mean([r['classification_report'][label][k] for r in rows]) for k in ['precision','recall','f1-score']])
        ax.imshow(v,cmap='YlGnBu',vmin=.5,vmax=1,aspect='auto')
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{v[i][j]:.3f}',ha='center',va='center',color='white' if v[i][j]>.8 else '#21383d')
        ax.set(yticks=range(4),yticklabels=LABELS['ag_news'],xticks=range(3),xticklabels=['Precision','Recall','F1'],title=SHORT[name])
    fig.suptitle('AG News per-class scores (0–1)',fontsize=15)
    fig.text(.1,.02,'Source: frozen test results, mean over seeds 42/43/44; support 250 per class.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.93]);save(fig,'news_per_class')
    rows=[];conclusions=[]
    for source in SOURCES:
        current=summary[summary.source==source].set_index('model').reindex(MODELS)
        for name,r in current.iterrows():
            rows.append([TITLES[source],SHORT[name]]+[f'{r[c]:.4f}' for c in cols[:5]])
        winners={c:SHORT[current[c].idxmax()] for c in ['macro_precision','macro_recall','macro_f1']}
        conclusions.append(f'- **{TITLES[source]}**: highest mean Macro Precision: {winners["macro_precision"]}; Macro Recall: {winners["macro_recall"]}; Macro F1: {winners["macro_f1"]}.')
    class_rows=[]
    for name in MODELS:
        records=[r for r in within if r['source']=='ag_news' and r['model']==name]
        for label in LABELS['ag_news']:
            values=[r['classification_report'][label] for r in records]
            class_rows.append([SHORT[name],label]+[f'{np.mean([v[k] for v in values]):.4f}' for k in ['precision','recall','f1-score']]+[250])
    news=summary[summary.source=='ag_news'].set_index('model').reindex(MODELS)
    cm_dist=next(r['confusion_matrix'] for r in within if r['source']=='ag_news' and r['model']=='distilbert' and r['seed']==42)
    mis=np.array(cm_dist);np.fill_diagonal(mis,0);i,j=np.unravel_index(mis.argmax(),mis.shape)
    report=f'''# Traditional NLP and Deep Learning for Binary and Multiclass Text Classification

## Scope and course alignment

This V2 extends the completed sentiment project at the user's request. The GUI is English throughout. Four datasets now cover binary sentiment (IMDb, Rotten Tomatoes, Yelp) and four-class news categorization (AG News: World, Sports, Business, Sci/Tech). The comparison includes TF-IDF + Linear SVM, training-only Word2Vec mean vectors + Linear SVM, Word2Vec-initialized BiLSTM and fine-tuned DistilBERT. TF-IDF Logistic Regression is an additional baseline. The course examples offer alternatives; this project does not claim to implement all listed methods.

Every dataset uses 6,000 training,1,000 validation and1,000 test examples. The original three binary splits and checkpoints remain unchanged. AG News samples 1,500 training and 250 validation/test examples per class from official splits, reserves test examples before training deduplication, and records hashes in `data-v2/audit.json`. There is no exact overlap across the prepared news partitions or with the selected sentiment samples. Near duplicates remain a limitation. Raw news data have 120,000 training and 7,600 test examples; only the stated sample is evaluated.

V2 was specified after the original sentiment test results were inspected. Added binary models are exploratory extensions of that benchmark. AG News has a newly reserved official-test sample. Validation Macro F1 selects all new hyperparameters and checkpoints, and V2 freezes 36 final model artifacts before any new test evaluation. Original cross-corpus transfer remains restricted to the original sentiment model families. Sentiment and news labels are incompatible; no cross-task transfer score is invented.

## Methods and tuning budgets

TF-IDF uses up to 30,000 training-only features, min_df=2, sublinear TF, two n-gram choices and C in 0.1/1/10. Word2Vec uses 100-dimensional skip-gram negative-sampling vectors, window 5, min_count 2, 10 epochs, a single worker and seed 2026. Its vocabulary contains at most 29,998 training words, with padding/unknown entries added for the sequence model. Mean-vector SVM averages known vectors, fits a training-only StandardScaler, and selects C in 0.1/1/10. Averaging loses word order; zero-known-word inputs map to zeros before scaling.

BiLSTM initializes trainable word embeddings from the same source Word2Vec vectors, uses one bidirectional layer with 64 units per direction, packs variable-length inputs, concatenates final forward/backward states and applies dropout 0.3 plus a classification head. It reads up to 256 word tokens. Seed42 compares learning rates 0.001 and 0.003 over five epochs, then the selected rate trains seeds43/44. The highest validation Macro F1 epoch wins, with the first epoch breaking ties.

News DistilBERT starts from the same generic base model as V1 and has a newly initialized four-class head. It compares learning rates2e-5 and5e-5 with seed42 over three epochs, then trains seeds43/44 at the selected rate. AdamW, batch 16, weight decay 0.01, warmup 10% and first 256 subword tokens follow V1. CUBLAS_WORKSPACE_CONFIG is set before CUDA initialization in the extension. The original V1 CuBLAS limitation remains documented in its report.

The extension adds four Word2Vec fits, 24 new classical candidate fits (12 news TF-IDF candidates and 12 mean-vector SVM candidates),16 BiLSTM runs and four news DistilBERT runs. These are different search spaces and computational budgets; we compare complete pipelines, not isolated architecture effects. Local Word2Vec and externally pretrained DistilBERT have unequal prior resources.

## Multiple metrics

Accuracy counts all correct predictions. Per-class Precision penalizes false positives, Recall penalizes false negatives, and F1 is their harmonic mean. Macro averages class scores equally; Weighted averages by true support; Micro pools all decisions. We save Macro, Micro and Weighted Precision/Recall/F1, Accuracy, MCC, class support and confusion matrices for every run.

Balanced tests make Macro and Weighted scores coincide here. In single-label classification, Micro Precision/Recall/F1 equal Accuracy. These equalities do not provide independent evidence of quality. Macro F1 is the validation selection criterion and a useful primary comparison because each class matters. Per-class Recall and the confusion matrix reveal missed minority or difficult categories that an overall score can hide in other class distributions.

{table(['Dataset','Model','Accuracy','Macro P','Macro R','Macro F1','Weighted F1'],rows)}

![Multiple evaluation metrics](figures-v2/multiple_metrics.png)

Deep-model entries average three seeds. Full per-seed metrics and sample SD appear in `comparison-v2.csv` and `results-v2/metrics.csv`. Traditional models are fixed fits. Do not interpret training SD as a test confidence interval or treat the highest point estimate as a statistically established win over every other model.

## Which method is better?

{chr(10).join(conclusions)}

The winners above reflect different loss consequences. If false positives are costly, inspect Precision for the affected class; if missed relevant items are costly, inspect Recall; use F1 for balance. Cost and input coverage also matter. The original paired bootstrap tests only SVM versus DistilBERT seed42 on the sentiment datasets: its interval excludes zero on Rotten Tomatoes and spans zero on IMDb/Yelp. No new significance claim is made for every V2 pair.

In this protocol, TF-IDF SVM exceeds the BiLSTM mean Macro F1 on all four corpora, despite BiLSTM being a deep sequential model. Word2Vec mean-vector SVM is especially weak on short Rotten Tomatoes snippets relative to TF-IDF. This pattern is consistent with limited local embedding training and loss of word order, but the experiment does not isolate those causes. It does not establish that BiLSTM or Word2Vec are generally inferior under larger training budgets, pretrained vectors or different optimization. DistilBERT has both the highest point estimates here and a substantial external pretraining advantage. A TF-IDF model remains a useful low-cost baseline, rather than an obsolete method merely because it is traditional.

## Four-class news analysis

{table(['Model','Class','Precision','Recall','F1','Support'],class_rows)}

![News class-level metrics](figures-v2/news_per_class.png)

![Four-class confusion matrices](figures-v2/news_confusion.png)

For DistilBERT seed42, the largest directed off-diagonal confusion is true **{LABELS['ag_news'][i]}** predicted **{LABELS['ag_news'][j]}** ({int(mis[i,j])} examples). This localizes a concrete source of errors without pretending that a label pair alone explains the model's internal decision. Topic overlap and wording can be examined in individual texts. Four-class news and binary sentiment are different corpora, so their score difference does not isolate the effect of class count.

## Computational cost

`results-v2/benchmark.json` combines the original nine measurements with eleven newly timed source/model combinations. All use the same 200 review texts, four CPU threads, batch 1, 20 warm-ups and five repetitions, including cleaning and feature conversion but excluding loading. This common workload enables a like-for-like input comparison but is not a news-serving workload. Legacy measurements occurred in an earlier session. Per-run training and checkpoint sizes remain recorded in metadata; Word2Vec training is a shared preprocessing cost and should not be silently ignored.

## GUI and reproducibility

The English GUI switches between Binary sentiment and Multiclass news. It predicts with all five models, distinguishes SVM margins from uncalibrated probabilities, selects Macro/Weighted/Micro averaging, exposes class metrics and count/row-normalized confusion matrices, and explains the ranking under multiple metrics. Prediction demonstrations use seed42 consistently.

Reproduce V1 first if missing, then run `python extension.py all`, `python make_report_v2.py`, and `python -m unittest -v test_project test_extension`. Install `requirements-v2.txt`. New model/data files live in `models-v2` and `data-v2`; original files remain as the V1 evidence trail. V2 evaluation contains 66 records and 66,000 individual predictions, including 30 original off-diagonal sentiment records. There are 36 within-dataset run records across 20 source/model combinations.

## Limitations and remaining submission work

Binary classes force neutral sentiment into two labels. BiLSTM and DistilBERT truncate at different token units. Sparse models and mean vectors receive the full cleaned text. Local Word2Vec trained on 6,000 examples is not equivalent to a large pretrained embedding model. Model family, pretraining, tuning budget and text coverage all differ. Exact and semantic duplication, benchmark reuse and hardware timing variability limit conclusions.

Member information remains deliberately unfilled at the user's request. Actual members must verify contributions, sign required material, use the official GAI form and record the presentation using real voices. This update does not generate signatures or claim submission.

## Sources

- [AG News dataset card](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Word2Vec implementation](https://radimrehurek.com/gensim/models/word2vec.html)
- [LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [DistilBERT](https://arxiv.org/abs/1910.01108)
- Original sentiment sources and results: `experiment-report.md`.
'''
    # Keep ordinary English spacing in generated prose.
    (REPORT/'classification-report-v2.md').write_text(report,'utf-8')
    write_json(REPORT/'slide-data-v2.json',{'summary':summary.fillna(0).to_dict('records'),'news_confusions':{r['model']:r['confusion_matrix'] for r in within if r['source']=='ag_news' and (r['seed']==42 or r['seed'] is None)},'labels':LABELS['ag_news'],'models':MODELS,'sources':SOURCES})
    cn='''# 二分类与多分类更新结果

GUI已全部改为英文。新增AG News四分类（World、Sports、Business、Sci/Tech），并加入Word2Vec均值+SVM和Word2Vec初始化的BiLSTM。

每个数据集均使用6,000训练、1,000验证、1,000测试。现在共四数据集、五种方法，36个最终模型运行记录；深度模型分别运行三个种子。所有方法同时报告Accuracy、Macro/Micro/Weighted Precision、Recall、F1、MCC，以及每类指标、支持数和混淆矩阵。

## AG News实测结果

'''+table(['Model','Accuracy','Macro Precision','Macro Recall','Macro F1'],[[SHORT[n]]+[f'{news.loc[n,c]:.4f}' for c in ['accuracy','macro_precision','macro_recall','macro_f1']] for n in MODELS])+'''
![多指标比较](figures-v2/multiple_metrics.png)

详细英文报告见classification-report-v2.md。原有二分类结果保持不变，新增二分类方法属于查看旧测试结果后的扩展比较；新闻测试集为新保留的独立样本。不能把新闻与影评的分数差异全部归因于类别数不同。
'''
    (REPORT/'results-summary-v2-zh.md').write_text(cn,'utf-8')
    print('V2 report, comparison CSV, three figure families and slide data generated.')

if __name__=='__main__':main()

"""Generate scientific figures and a report from actual frozen experiment files."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT, CONFIG, NAMES, MODEL_NAMES, read_json, write_json, model_dir, assert_provenance

DISPLAY = {'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes', 'yelp': 'Yelp'}
SHORT = {'logistic_regression':'TF-IDF + LR','linear_svm':'TF-IDF + SVM','distilbert':'DistilBERT'}
COLORS = ['#adb5bd','#586774','#1b766e']
REPORT = ROOT / 'report'
FIGS = REPORT / 'figures'

def save(fig, name):
    fig.savefig(FIGS / f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGS / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def md_table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + '\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows) + '\n'

def main():
    evaluation = read_json(ROOT / 'results' / 'evaluation.json'); assert_provenance(evaluation)
    benchmark = read_json(ROOT / 'results' / 'benchmark.json'); assert_provenance(benchmark)
    audit = read_json(ROOT / 'data' / 'audit.json')
    selection = read_json(ROOT / 'results' / 'transformer_selection.json')
    bootstrap = read_json(ROOT / 'results' / 'bootstrap.json')
    env = read_json(ROOT / 'environment.json')
    frame = pd.read_csv(ROOT / 'results' / 'metrics.csv')
    predictions = pd.read_csv(ROOT / 'results' / 'predictions.csv')
    within = frame[frame.source == frame.target]
    grouped = within.groupby(['source','model']).agg(mean_f1=('macro_f1','mean'), sd_f1=('macro_f1','std'),
                accuracy=('accuracy','mean'), runs=('macro_f1','count')).reset_index()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    x = np.arange(3); width = .24
    fig, ax = plt.subplots(figsize=(9,4.7))
    for i,name in enumerate(MODEL_NAMES):
        rows=grouped[grouped.model==name].set_index('source').reindex(NAMES)
        bars=ax.bar(x+(i-1)*width,rows.mean_f1,width,label=SHORT[name],color=COLORS[i],
                    yerr=rows.sd_f1.to_numpy() if name=='distilbert' else None,capsize=4 if name=='distilbert' else 0)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in rows.mean_f1],padding=5,fontsize=9)
    ax.set(xticks=x,xticklabels=[DISPLAY[n] for n in NAMES],ylim=(0,1.05),
           ylabel='Test Macro-F1 (0–1)', xlabel='Dataset',title='Within-dataset sentiment classification')
    ax.legend(loc='lower right',frameon=False)
    fig.subplots_adjust(bottom=.24)
    fig.text(.1,.025,'Source: frozen local experiments. DistilBERT: mean ± sample SD over seeds 42/43/44. Linear models: one fixed fit.',fontsize=8)
    save(fig,'within_dataset')

    matrices={}
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,name in zip(axes,MODEL_NAMES):
        matrix=frame[frame.model==name].pivot_table(index='source',columns='target',values='macro_f1').reindex(index=NAMES,columns=NAMES)
        matrices[name]=matrix.values.tolist()
        ax.imshow(matrix.values,vmin=.5,vmax=1,cmap='YlGnBu')
        for r in range(3):
            for c in range(3):
                val=matrix.iloc[r,c]
                ax.text(c,r,f'{val:.3f}',ha='center',va='center',color='white' if val>.78 else '#182126',fontsize=12)
        ax.set(xticks=range(3),xticklabels=['IMDb','RT','Yelp'],yticks=range(3),yticklabels=['IMDb','RT','Yelp'],
               xlabel='Test target',ylabel='Training source',title=SHORT[name])
    fig.suptitle('Cross-dataset Macro-F1 (0–1)',fontsize=15)
    fig.text(.1,.025,'Source: frozen local experiments. Diagonal = within-dataset. DistilBERT cells average three seeds. RT = Rotten Tomatoes.',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.94]);save(fig,'transfer')

    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
    for i,source in enumerate(NAMES):
        meta=read_json(model_dir(source,'distilbert',42)/'metadata.json')
        epochs=[v['epoch'] for v in meta['history']]
        axes[0,i].plot(epochs,[v['train_loss'] for v in meta['history']],marker='o',color=COLORS[2])
        axes[1,i].plot(epochs,[v['validation']['macro_f1'] for v in meta['history']],marker='o',color=COLORS[2])
        axes[0,i].set(title=DISPLAY[source],ylabel='Training cross-entropy')
        axes[1,i].set(xlabel='Epoch',ylabel='Validation Macro-F1 (0–1)',xticks=epochs,ylim=(.5,1))
    fig.suptitle('Selected learning rate, seed 42',fontsize=15)
    fig.text(.1,.015,'Source: saved training and validation logs, selected source learning rates, seed 42.',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.95]);save(fig,'training_curves')

    fig,ax=plt.subplots(figsize=(8,4.5))
    vals=[audit['datasets'][n]['test']['truncated_fraction']*100 for n in NAMES]
    bars=ax.bar([DISPLAY[n] for n in NAMES],vals,color=COLORS[2])
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=4)
    ax.set(ylabel='Test reviews exceeding 256 tokens (%)',xlabel='Dataset',ylim=(0,max(vals+[1])*1.25),title='DistilBERT input truncation')
    fig.subplots_adjust(bottom=.23)
    fig.text(.1,.02,'Source: frozen data audit, 1,000 test reviews per corpus.',fontsize=9)
    save(fig,'truncation')

    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,source in zip(axes,NAMES):
        row=next(r for r in evaluation['rows'] if r['source']==source and r['target']==source and r['model']=='distilbert' and r['seed']==42)
        matrix=np.asarray(row['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=500)
        for r in range(2):
            for c in range(2): ax.text(c,r,str(matrix[r,c]),ha='center',va='center',color='white' if matrix[r,c]>250 else '#182126',fontsize=15)
        ax.set(title=DISPLAY[source],xlabel='Predicted label',ylabel='True label',xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])
    fig.suptitle('DistilBERT confusion matrices, seed 42 (counts)',fontsize=14)
    fig.text(.1,.015,'Source: saved within-dataset test predictions, 1,000 reviews per corpus.',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.91]);save(fig,'confusion_matrices')

    costs=pd.DataFrame(benchmark['rows'])
    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,name in enumerate(MODEL_NAMES):
        rows=costs[costs.model==name].set_index('source').reindex(NAMES)
        bars=ax.bar(x+(i-1)*width,rows.median_ms,width,label=SHORT[name],color=COLORS[i])
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in rows.median_ms],padding=4,fontsize=9)
    ax.set(xticks=x,xticklabels=[DISPLAY[n] for n in NAMES],yscale='log',ylabel='Median latency (ms/review)\nlogarithmic scale',xlabel='Training source',title='CPU inference on the same 200 reviews')
    ax.legend(frameon=False,loc='upper left')
    fig.subplots_adjust(bottom=.24)
    fig.text(.1,.025,'Source: local benchmark, 4 CPU threads, batch=1, 20 warm-up reviews and 5 repetitions. Model loading excluded.',fontsize=8)
    save(fig,'cpu_latency')

    data_rows=[]; result_rows=[]; parameter_rows=[]; cost_rows=[]; bootstrap_rows=[]
    training_summary=[]
    for source in NAMES:
        data_rows.append([DISPLAY[source]]+[audit['datasets'][source][s]['n'] for s in ['train','validation','test']]+[f'{audit["datasets"][source]["test"]["truncated_fraction"]:.1%}'])
        for name in MODEL_NAMES:
            row=grouped[(grouped.source==source)&(grouped.model==name)].iloc[0]
            display=f'{row.mean_f1:.4f} ± {row.sd_f1:.4f}' if row.runs>1 else f'{row.mean_f1:.4f}'
            result_rows.append([DISPLAY[source],SHORT[name],display,f'{row.accuracy:.4f}',int(row.runs)])
            if name!='distilbert':
                meta=read_json(model_dir(source,name)/'metadata.json')
                parameter_rows.append([DISPLAY[source],SHORT[name],str(meta['selected']['ngram']),meta['selected']['C'],'—'])
                fitting=meta['selected']['fit_seconds']; search=meta['search_wall_seconds']; size=meta['model_bytes']
                peak=None
            else:
                meta=read_json(model_dir(source,name,42)/'metadata.json')
                parameter_rows.append([DISPLAY[source],SHORT[name],'—','—',f'{meta["learning_rate"]:g}, epoch {meta["best_epoch"]} (seed42)'])
                fitting=meta['fit_wall_seconds']
                search=sum(v['fit_wall_seconds'] for v in selection['selections'][source]['candidate_metadata'])
                size=meta['model_bytes'];peak=meta['peak_gpu_allocated_bytes']
            latency=costs[(costs.source==source)&(costs.model==name)].iloc[0]
            cost_rows.append([DISPLAY[source],SHORT[name],f'{fitting:.1f}',f'{search:.1f}',f'{size/1024**2:.2f}',f'{latency.median_ms:.2f}',f'{latency.p95_ms:.2f}'])
            training_summary.append({'source':source,'model':name,'fit_seconds':fitting,'search_seconds':search,'model_mib':size/1024**2,'peak_gpu_allocated_mib':None if peak is None else peak/1024**2,'median_ms':latency.median_ms,'p95_ms':latency.p95_ms})
        b=bootstrap[source]
        bootstrap_rows.append([DISPLAY[source],f'{b["delta_macro_f1"]:+.4f}',f'[{b["ci95_low"]:+.4f}, {b["ci95_high"]:+.4f}]'])
    human=pd.read_csv(ROOT/'results'/'error_review.csv').fillna('')
    reviewed=int((human.review_status=='human_reviewed').sum())
    macro={name:float(grouped[grouped.model==name].mean_f1.mean()) for name in MODEL_NAMES}
    factual='; '.join(f'{SHORT[n]} {macro[n]:.4f}' for n in MODEL_NAMES)
    boot_text='\n'.join(f'- {DISPLAY[n]}: seed42 DistilBERT minus SVM = {bootstrap[n]["delta_macro_f1"]:+.4f}, paired 95% interval [{bootstrap[n]["ci95_low"]:+.4f}, {bootstrap[n]["ci95_high"]:+.4f}].' for n in NAMES)
    clear=[DISPLAY[n] for n in NAMES if bootstrap[n]['ci95_low']>0]
    inconclusive=[DISPLAY[n] for n in NAMES if bootstrap[n]['ci95_low']<=0<=bootstrap[n]['ci95_high']]
    transfer_bert=matrices['distilbert']
    report=f'''# Cross-Dataset Sentiment Classification

## Abstract

We compare TF-IDF Logistic Regression, TF-IDF Linear SVM and fine-tuned DistilBERT on three English review datasets under fixed labeled-data budgets. Each dataset has a separate source model. We select hyperparameters and checkpoints using source validation labels, then freeze all final artifacts before evaluating within-dataset and cross-dataset test performance. The equal-weight mean of the three within-dataset Macro-F1 values is: {factual}. These figures describe this sampled protocol, not full-dataset benchmark replication. Accuracy, transfer matrices and CPU latency provide additional evidence about practical trade-offs.

## 1. Problem and experimental questions

The application helps a reader classify English movie or business reviews as positive or negative. We ask whether contextual pretraining improves classification over strong sparse baselines, how directly transferred models behave on another review corpus, and what inference and training costs accompany any gains. We do not propose a new architecture. DistilBERT has external pretraining, so this comparison does not isolate architecture alone.

## 2. Data and leakage controls

{md_table(['Dataset','Train','Validation','Test','Test truncated at 256 tokens'],data_rows)}
All sampled splits are class-balanced. IMDb and Yelp validation examples come from their official training split. Rotten Tomatoes retains its official validation split. All test examples come from official test splits. Random sampling uses seed {CONFIG['sample_seed']}. Each run uses exactly the same frozen sample IDs.

We normalize HTML and whitespace, preserve negation and punctuation, and hash case-folded normalized text. Test examples take precedence over official validation and training rows during global exact deduplication. Conflicting duplicate labels are removed. The audit processes {audit['original_rows']:,} original rows and retains {audit['eligible_rows']:,} eligible unique rows before sampling. The final manifest contains no repeated normalized text hashes or sample IDs. A deterministic containment probe found {audit['containment_probe']['matching_count']} exact substring matches in {audit['containment_probe']['probe_count']} sampled short snippets of at least 50 characters. This is a limited probe, not proof that semantic near-duplicates are absent.

IMDb contains long movie reviews, Rotten Tomatoes contains short movie-review snippets, and Yelp contains business/service reviews. They differ in genre, length, label construction and difficulty. IMDb to Rotten Tomatoes is a change within the movie-review domain. Only comparisons involving Yelp include a broader domain change. Dataset versions and asset hashes appear in `assets.lock.json`; individual sample IDs and split hashes appear in `data/manifest.csv` and `data/audit.json`.

![Test truncation](figures/truncation.png)

## 3. Methods and model selection

The two sparse models use a TF-IDF pipeline fitted only on source training text. Both search C in {{0.1, 1, 10}} and word n-gram ranges {{(1,1), (1,2)}} with at most 30,000 features, min_df=2 and sublinear term frequency. They each receive six configurations per dataset. The first configuration in fixed order breaks exact validation ties.

DistilBERT starts from the generic `distilbert/distilbert-base-uncased` checkpoint, not a sentiment-fine-tuned model. We add a randomly initialized two-class head and fine-tune the encoder using AdamW, effective batch size {CONFIG['batch_size']}, weight decay {CONFIG['weight_decay']}, 10% warm-up and maximum input length {CONFIG['max_length']}. For each source, seed42 compares learning rates 2e-5 and 5e-5 for three epochs, retaining the best validation Macro-F1 epoch. The selected learning rate then trains seeds43 and44. Dynamic padding rounds to a multiple of eight. Training uses mixed precision on the available GPU; evaluation uses full precision. Seeds are fixed and deterministic algorithms are requested in warning mode. This run did not configure the CuBLAS deterministic-workspace environment variable, and PyTorch emitted reproducibility warnings retained in `run.log`. Exact GPU reruns are therefore not guaranteed, even on the same setup; the three-seed variation must not be read as a bitwise reproducibility guarantee.

{md_table(['Source','Model','TF-IDF n-gram','C','DistilBERT LR / selected epoch'],parameter_rows)}
There are 36 sparse candidate fits and 12 DistilBERT training runs, of which nine DistilBERT checkpoints enter the final results. The fifteen final model artifacts and their SHA256 values are recorded in `results/protocol.freeze.json`. The optional training-size experiment was excluded before testing. Neither target labels nor test metrics select learning rates, epochs or demonstration checkpoints. The demonstration uses seed42 consistently.

![Training and validation curves](figures/training_curves.png)

## 4. Within-dataset test results

{md_table(['Dataset','Model','Macro-F1','Accuracy','Training runs'],result_rows)}
![Within-dataset Macro-F1](figures/within_dataset.png)

DistilBERT entries give the mean and sample standard deviation over three training seeds on a fixed test sample. Traditional classifiers have a fixed fit and no artificial three-run error bar. Training standard deviation does not measure test-sample uncertainty. For a paired comparison, we preselect SVM and DistilBERT seed42 and resample the same test IDs within each class 1,000 times.

{md_table(['Dataset','DistilBERT seed42 minus SVM','95% paired bootstrap interval'],bootstrap_rows)}
The intervals condition on these two trained models and use the percentile method. They do not integrate training randomness, correct for multiple comparisons or establish a causal effect of architecture.

In the preselected seed42 comparison, the interval lies above zero for {', '.join(clear)}. It spans zero for {', '.join(inconclusive)}. Higher point estimates therefore provide stronger evidence on the short movie-review corpus than on the other two sampled corpora; they do not justify a universal claim that the Transformer wins reliably in every setting.

![Confusion matrices](figures/confusion_matrices.png)

## 5. Cross-dataset transfer

![Source-target matrices](figures/transfer.png)

Each row is the training source and each column the test target. We apply the saved source tokenizer/vectorizer and model directly, with no target-label adaptation or calibration. The diagonal repeats the within-dataset experiment; each off-diagonal cell tests direct transfer. DistilBERT cells average seeds42/43/44. Full per-seed metrics, class reports and confusion matrices are in `results/evaluation.json`; all {len(predictions):,} individual predictions are in `results/predictions.csv`.

Cross-dataset differences combine vocabulary, sentiment expression, document length and corpus difficulty. A source-diagonal minus target-cell difference is descriptive and must not be attributed purely to domain shift. No target-training-data adaptation experiment is included.

For example, the IMDb-trained DistilBERT averages {transfer_bert[0][2]:.4f} on Yelp but {transfer_bert[0][1]:.4f} on Rotten Tomatoes. Sharing a movie-review domain does not guarantee higher performance in this protocol. This observation is consistent with several explanations, including corpus difficulty and text style, which this experiment does not isolate causally.

## 6. Training and deployment costs

Training GPU: {env['hardware']['gpu']}. CPU: {env['hardware']['processor']}. Inference measurements use the same CPU and {CONFIG['cpu_threads']} threads for every model. All nine representative source-model combinations process the same fixed 200 mixed-source test texts at batch size1, with 20 warm-up examples and five repetitions. Timing includes cleaning, tokenization/vectorization and model inference, but excludes loading and downloading. The median and p95 are empirical timings, not serving-load throughput guarantees.

{md_table(['Source','Model','Selected fit (s)','LR/C search (s)','Model (MiB)','CPU median (ms)','CPU p95 (ms)'],cost_rows)}
For sparse models, selected fit time includes TF-IDF fitting; the search time includes all six candidates and their validation predictions. For DistilBERT, selected fit is seed42 training, validation and best-checkpoint saving; search sums the two seed42 learning-rate runs. Extra seeds43/44 are additional research cost and remain recorded separately in model metadata. Tokenization and I/O boundaries differ between these training implementations, so the fit columns are operational measurements rather than hardware-normalized efficiency claims. Both training device and preprocessing times are recorded in metadata. GPU peak allocated memory measures live PyTorch tensors, not total GPU or process memory. This implementation does not measure peak CPU RAM, which is a deviation from the original plan.

![CPU inference cost](figures/cpu_latency.png)

## 7. Error analysis and review status

The reproducible error queue contains {len(human)} samples, at most ten negative and ten positive examples per dataset, drawn from cases where at least one same-source model errs. DistilBERT uses seed42 for this queue. `results/error_review.csv` retains the full text, gold label and all model predictions. It deliberately leaves human categories, explanations, reviewers and signatures unfilled. Currently {reviewed} rows have the status `human_reviewed`. The queue is stratified for inspection and cannot estimate population error-category frequencies.

An assistant-generated qualitative draft may be supplied separately, clearly identified as a draft. Team members must verify the labels and explanations and record genuine reviewer names. They must not claim that this draft represents two independent human reviewers. Test-error inspection follows model freezing and does not feed another round of model selection.

## 8. Limitations and reproducibility

The three test subsets have 1,000 examples each and do not represent all review settings. We balance the classes artificially, so deployment priors can differ. Binary labels force neutral or mixed opinions into a positive/negative decision. Long reviews lose tokens beyond the first256 for DistilBERT, whereas TF-IDF receives the full cleaned text. The truncation audit makes this resource choice explicit. Near-duplicates, common movies/authors and unknown pretraining overlap remain possible.

Only three models and a small tuning budget are included. Unequal external pretraining and different search spaces limit architectural conclusions. Frozen protocol files, raw per-example outputs and environment versions allow the reported numbers to be independently checked. Source data and pretrained assets remain external downloads in the compact submission archive. Model weights and prepared text remain in the local project folder.

Reproduce in order: `python download_assets.py`, `python pipeline.py all`, `python make_report.py`, `python -m unittest -v test_project`. Already completed stages reuse recorded artifacts. A changed configuration requires a fresh project copy, not overwritten test history. Run `python -m streamlit run app.py` for the demonstration.

## 9. Submission status

Code, model artifacts, results and report form the computational deliverable. A slide deck and a timed narration script can support the required presentation. The course forbids generated voice. Actual team members must record the presentation, verify personal contributions, sign the reflection and use the course's exact GAI declaration template. Peer evaluation is an individual action on NTULearn. This report does not fabricate those actions or claim the project has been submitted.

## References

1. Maas et al. (2011). Learning Word Vectors for Sentiment Analysis. [ACL paper](https://aclanthology.org/P11-1015/). [IMDb dataset card](https://huggingface.co/datasets/stanfordnlp/imdb).
2. Pang and Lee (2005). Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. [Original data release](https://www.cs.cornell.edu/people/pabo/movie-review-data/). [Versioned-format dataset card](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).
3. Zhang, Zhao and LeCun (2015). Character-level Convolutional Networks for Text Classification. [Paper](https://arxiv.org/abs/1509.01626). [Yelp Polarity dataset card](https://huggingface.co/datasets/fancyzhx/yelp_polarity).
4. Sanh et al. (2019/2020). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. [Paper](https://arxiv.org/abs/1910.01108). [Base model card](https://huggingface.co/distilbert/distilbert-base-uncased).
5. Hugging Face. [Text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification).
6. scikit-learn. [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) documentation.
'''
    REPORT.mkdir(exist_ok=True)
    (REPORT/'experiment-report.md').write_text(report,'utf-8')
    slide_data={'datasets':[{'name':DISPLAY[n],**{s:audit['datasets'][n][s]['n'] for s in ['train','validation','test']}} for n in NAMES],
        'within':grouped.fillna(0).to_dict('records'),'transfer':matrices,'costs':training_summary,
        'bootstrap':bootstrap,'macro_average':macro,'hardware':env['hardware'],
        'data_audit':{'raw_rows':audit['original_rows'],'eligible_rows':audit['eligible_rows'],'selected_rows':24000},
        'truncation_percent':vals,'human_reviewed':reviewed,'error_queue_size':len(human)}
    write_json(REPORT/'slide-data.json',slide_data)
    script=f'''# 真人录制讲稿与演示顺序（目标 9 分钟）

状态：根据真实实验自动填入数字的讲解草稿。实际成员需理解、核对并用自己的声音录制。请勿使用 AI 配音。

## 0:00–0:40 项目问题
我们的项目比较英文评论情感分类中的传统方法和微调 DistilBERT。用户可以输入电影或商家评论，查看三个模型的判断。我们重点检查同集准确性、跨数据集迁移及计算成本，不提出新架构。

## 0:40–1:30 数据与公平比较
我们使用 IMDb、Rotten Tomatoes 和 Yelp，分别代表电影长评、电影短评以及商家评论。每集使用6,000条训练、1,000条验证、1,000条测试，类别均衡。我们保留官方测试来源，做全局文本去重并冻结相同的样本清单。两个电影语料不应被说成两个完全不同的领域。

## 1:30–2:40 模型与调参
两个传统模型共享 TF-IDF 特征方案，分别使用逻辑回归和线性 SVM。我们搜索两种 n-gram 和三个 C。DistilBERT 从基础预训练权重开始，输入最多256个 token。我们在验证集比较两个学习率，选最佳 epoch，再固定学习率运行三个随机种子。测试集始终不参与这些选择。

## 2:40–3:50 同集结果
展示 within_dataset 图。三个数据集等权平均 Macro-F1 分别是：{factual}。逐个解释每集的真实结果，不把传统模型的确定性结果冒充三个独立训练。DistilBERT 误差条表示三个训练种子的标准差，不是测试集置信区间。
配对分析可选讲其中一项：
{boot_text}

## 3:50–4:50 跨数据集迁移
展示 transfer 图。行是训练来源，列是测试目标。对角线是同集测试，其他格子是模型直接换到另一批评论。挑选图中一个实际明显的变化，报出两个格子的数值并说明比较方向。迁移变化也可能来自长度和文体，不能全部归因于领域。

## 4:50–5:40 成本与错误
展示 cpu_latency 图。所有模型使用同一 CPU、四线程、相同200条评论、单条推理，每条重复五次。先报延迟再报效果，说明权衡。错误分析使用单独抽取的最多60例；成员需在录制前核对案例。不要把自动生成的错误队列说成人工复核已完成。

## 5:40–6:40 Python 代码讲解
打开 pipeline.py：展示 prepare 的哈希去重、baselines 的 Pipeline、train_transformer 的验证集 checkpoint 选择，以及 freeze 在 evaluate 之前运行。打开 config.json 指出统一样本规模和固定参数。打开 results/protocol.freeze.json 与 metrics.csv 展示可追溯证据。

## 6:40–8:15 应用演示
启动 Streamlit。选择 IMDb，输入默认英文评论并比较三个模型；说明 SVM 的 margin 不是概率。换一条转折或混合情感评论。切换训练来源，观察判断是否变化。打开实验结果页，展示同集表和迁移矩阵。输入长评论时解释截断提示。避免演示过程中训练模型或下载文件。

## 8:15–9:00 结论与局限
按真实图表概括最有证据的两点。说明二分类无法表达中性、长文本截断以及预训练资源不对等的限制。最后展示参考文献，并由成员如实说明 GAI 辅助内容。

## 录制前核对
- 视频总时长不超过10分钟，所有配音来自成员真人。
- 读懂数字和比较方向，不照读尚未核对的案例或贡献。
- 不展示私人路径、无关桌面或登录信息。
- 签名反思和 GAI 表另外准备，按课程模板提交。
'''
    (REPORT/'narration-script-zh.md').write_text(script,'utf-8')
    gains=[]
    for source in NAMES:
        subset=grouped[grouped.source==source].set_index('model')
        strongest=max(float(subset.loc[n,'mean_f1']) for n in MODEL_NAMES[:2])
        bert=float(subset.loc['distilbert','mean_f1'])
        gains.append(f'- {DISPLAY[source]}：DistilBERT {bert:.4f}，该集较强传统模型 {strongest:.4f}，差值 {(bert-strongest)*100:+.2f} 个百分点。')
    chinese=f'''# 项目结果速览

这是已执行实验的结果摘要，不是预测成绩。完整方法、每类指标、逐种子记录与限制见 `experiment-report.md`。

## 已完成的计算实验

三数据集各6,000训练／1,000验证／1,000测试，共24,000条固定样本。36次传统候选拟合、12次 DistilBERT 训练，保留15个最终模型。冻结后完成45组来源—目标—种子评估，共45,000条预测记录，以及同一CPU上的推理计时。

## 同集测试

三个数据集等权平均 Macro-F1：{factual}。DistilBERT 的每集数值是三个训练种子的平均值，传统模型是固定配置的单次结果。

{chr(10).join(gains)}

![同集结果](figures/within_dataset.png)

## 迁移与成本

![迁移矩阵](figures/transfer.png)

训练来源按行、测试目标按列。模型直接迁移，不使用目标标签调整参数。两个电影语料的差异不应全称为领域差异，长度与文体也有影响。

![同CPU推理成本](figures/cpu_latency.png)

计时使用同一CPU、四线程、相同200条评论、batch1、每条五次。包含清理和特征转换，不含模型加载。实际延迟和训练成本详见完整报告。

## 使用与提交

在项目目录运行 `python -m streamlit run app.py`，或双击 `start_app.cmd`。应用可输入英文评论、选择训练来源、比较三模型并查看实际实验结果。演示固定使用 seed42。

现有人工复核队列共{len(human)}条，当前{reviewed}条标为人工复核完成；未代填复核人。真人录制视频、核实个人贡献并签名、填写老师的正式GAI表、完成个人互评仍需真实成员执行。

## 解释边界

这里只比较抽样后的英文正负二分类，不是原论文全量复现。DistilBERT 使用额外预训练，且长文本截断；IMDb 测试截断率 {vals[0]:.1f}%，Rotten Tomatoes {vals[1]:.1f}%，Yelp {vals[2]:.1f}%。三种子标准差不是测试置信区间，模型概率未经额外校准。GPU 确定性设置有已记录的 CuBLAS 限制，不能承诺逐位重现。
'''
    (REPORT/'results-summary-zh.md').write_text(chinese,'utf-8')
    print('Generated report, six figure families (PNG/PDF), slide data and narration script from actual results.')

if __name__=='__main__': main()

"""English GUI: binary/multiclass tasks and multiple evaluation metrics."""
import time
import numpy as np
import pandas as pd
import streamlit as st
from extension import ROOT,CFG,SOURCES,MODELS,LABELS,load_model as base_load_model,predict,directory as base_directory,metric_set
import hard_benchmark as hard
from pipeline import clean_text,read_json
from models_extra import words
st.set_page_config(page_title='Text Classification Lab',layout='wide')
st.title('Text Classification Lab')
st.caption('Traditional NLP features, sequential deep learning and a pretrained Transformer')
DATASETS={'imdb':'IMDb — long movie reviews','rotten_tomatoes':'Rotten Tomatoes — short movie reviews','yelp':'Yelp — business reviews','ag_news':'AG News — four news topics'}
DATASETS.update({'mnli':'MultiNLI — sentence-pair inference','mnli_matched':'MultiNLI — matched genres','mnli_mismatched':'MultiNLI — mismatched genres'})
LABELS={**LABELS,'mnli':hard.LABELS,'mnli_matched':hard.LABELS,'mnli_mismatched':hard.LABELS}
MODEL_LABELS={'logistic_regression':'TF-IDF + Logistic Regression','linear_svm':'TF-IDF + Linear SVM','word2vec_svm':'Word2Vec mean + Linear SVM','bilstm':'Word2Vec + BiLSTM','distilbert':'DistilBERT'}
METRIC_LABELS={'accuracy':'Accuracy','macro_precision':'Macro Precision','macro_recall':'Macro Recall','macro_f1':'Macro F1','weighted_precision':'Weighted Precision','weighted_recall':'Weighted Recall','weighted_f1':'Weighted F1','micro_precision':'Micro Precision','micro_recall':'Micro Recall','micro_f1':'Micro F1','mcc':'MCC'}
@st.cache_resource(show_spinner=False)
def cached_model(source,name):
 import torch
 torch.set_num_threads(4)
 return hard.load_model(name,42,'cpu') if source=='mnli' else base_load_model(source,name,42,'cpu')
def directory(source,name,seed=None):
 return hard.directory(name,seed) if source=='mnli' else base_directory(source,name,seed)
@st.cache_data(ttl=5)
def results_data():
 p=ROOT/'results-v2'/'evaluation.json'
 if p.exists():
  result=read_json(p)
  new=ROOT/'results-hard'/'evaluation.json'
  if new.exists():result['rows']=result['rows']+read_json(new)['rows']
  return result
 old=read_json(ROOT/'results'/'evaluation.json')['rows'];pred=pd.read_csv(ROOT/'results'/'predictions.csv');rows=[]
 for r in old:
  subset=pred[(pred.source==r['source'])&(pred.target==r['target'])&(pred.model==r['model'])]
  if r['seed'] is not None:subset=subset[subset.seed==r['seed']]
  rows.append({k:r[k] for k in ['source','target','model','seed','test_n']}|metric_set(subset.label,subset.prediction,r['target']))
 return {'rows':rows}
task=st.radio('Classification task',['Binary sentiment','Multiclass news','Natural language inference'],horizontal=True)
allowed=['mnli'] if task=='Natural language inference' else ['ag_news'] if task=='Multiclass news' else SOURCES[:3]
demo,results,robustness,methods=st.tabs(['Try a prediction','Evaluation & comparison','Imbalance robustness','Methods & limitations'])
with demo:
 source=st.selectbox('Training dataset',allowed,format_func=lambda n:DATASETS[n],key='demo_'+task)
 default='The team won the championship after a dramatic final match.' if source=='ag_news' else 'The story starts slowly, but the ending makes the whole film worthwhile.'
 premise=''
 if source=='mnli':
  premise=st.text_area('Premise',value='A woman is reading a book in a quiet library.',height=100,max_chars=10000,key='premise_mnli')
  text=st.text_area('Hypothesis',value='Someone is reading indoors.',height=100,max_chars=10000,key='hypothesis_mnli')
  st.caption('Entailment: follows from the premise. Contradiction: incompatible with it. Neutral: neither established nor contradicted.')
 else:text=st.text_area('Enter English text',value=default,height=140,max_chars=10000,key='text_'+task)
 st.caption('Classes: '+', '.join(LABELS[source])+'.')
 if source in SOURCES[:3]:st.caption('Neutral or mixed sentiment still receives a binary label.')
 if st.button('Compare all models',type='primary'):
  cleaned=clean_text(text)
  if not cleaned or (source=='mnli' and not clean_text(premise)):st.warning('Please enter some English text first. For inference, both the premise and hypothesis are required.')
  else:
   output=[]
   for name in MODELS:
    if not (directory(source,name,42 if name in ['bilstm','distilbert'] else None)/'metadata.json').exists():
     st.info(MODEL_LABELS[name]+': training is not complete yet.');continue
    with st.spinner('Running '+MODEL_LABELS[name]+'...'):
     model,tok=cached_model(source,name);truncated=False
     if name=='distilbert':truncated=len((tok(clean_text(premise),cleaned,truncation=False,verbose=False) if source=='mnli' else tok(cleaned,truncation=False,verbose=False))['input_ids'])>CFG['max_length']
     if name=='bilstm':truncated=max(len(words(cleaned)),len(words(premise)))>CFG['max_length']
     start=time.perf_counter()
     prediction,score,kind=hard.predict(model,tok,[{'premise':clean_text(premise),'hypothesis':cleaned}],name) if source=='mnli' else predict(model,tok,[cleaned],name)
     ms=(time.perf_counter()-start)*1000
    label=int(prediction[0]);values=np.asarray(score)
    if kind=='probability':value=f'{float(values[0,label]):.1%} predicted-class probability'
    elif values.ndim==1:value=f'{float(values[0]):+.3f} positive-class margin'
    else:value=f'{float(values[0,label]):+.3f} predicted-class margin'
    output.append({'Model':MODEL_LABELS[name],'Prediction':LABELS[source][label],'Score':value,'CPU latency (ms)':round(ms,2),'Input truncated':truncated})
   if output:st.dataframe(pd.DataFrame(output),hide_index=True,use_container_width=True)
   if any(x['Input truncated'] for x in output):st.warning('At least one input was truncated. BiLSTM reads up to 256 words per sentence; DistilBERT uses a combined 256-subword budget for a sentence pair.')
   st.caption('SVM margins are decision scores, not probabilities. Probability outputs are uncalibrated. Loading is excluded from latency.')
with results:
 data=results_data()
 eval_options=['mnli_matched','mnli_mismatched'] if task=='Natural language inference' else allowed
 chosen=st.selectbox('Evaluation dataset',eval_options,format_func=lambda n:DATASETS[n],key='eval_'+task)
 average=st.selectbox('Precision / Recall / F1 averaging',['macro','weighted','micro'],format_func=lambda n:{'macro':'Macro — equal class weight','weighted':'Weighted — class support','micro':'Micro — pooled decisions'}[n])
 rows=[r for r in data['rows'] if r['source']==('mnli' if chosen.startswith('mnli_') else chosen) and r['target']==chosen]
 if not rows:st.info('Evaluation for this task is still running. No estimated scores are shown.')
 else:
  f=pd.DataFrame(rows);columns=list(dict.fromkeys(['accuracy',average+'_precision',average+'_recall',average+'_f1','macro_f1','weighted_f1','mcc']))
  summary=f.groupby('model')[columns].mean().reindex([n for n in MODELS if n in f.model.unique()])
  summary['Training runs']=f.groupby('model').size();summary['Macro F1 training SD']=f.groupby('model').macro_f1.std()
  display=summary.rename(index=MODEL_LABELS,columns=METRIC_LABELS);display.index.name='Model'
  st.subheader('Overall test metrics')
  st.dataframe(display.style.format({c:'{:.4f}' for c in display.columns if c!='Training runs'},na_rep='—'),use_container_width=True)
  st.caption('Deep models average seeds 42/43/44. Traditional models use a fixed fit. Selection uses validation Macro F1; training SD is not a test confidence interval.')
  st.subheader('Which model is better?')
  for metric in ['macro_f1',average+'_precision',average+'_recall']:
   best=summary[metric].idxmax();st.write(f'Highest {METRIC_LABELS[metric]}: {MODEL_LABELS[best]} ({summary.loc[best,metric]:.4f}).')
  st.caption('Precision penalizes false positives; Recall penalizes missed examples; F1 balances both. Point estimates alone do not establish a reliable advantage.')
  if chosen in SOURCES[:3]:
   b=read_json(ROOT/'results'/'bootstrap.json')[chosen]
   st.caption(f'Original comparison, DistilBERT seed42 minus TF-IDF SVM: Delta Macro F1 {b["delta_macro_f1"]:+.4f}, 95% interval [{b["ci95_low"]:+.4f}, {b["ci95_high"]:+.4f}]. This does not test every added model.')
  if chosen.startswith('mnli_'):
   st.caption('Uniform guessing has expected Accuracy of 1/3. These are reserved samples from official validation pools, not official hidden-test leaderboard scores. Matched and mismatched evaluate different genre conditions.')
   diagnostic=next((r for r in rows if r['model']=='hypothesis_only_svm'),None)
   if diagnostic:st.write(f'Hypothesis-only diagnostic (fixed TF-IDF SVM): Accuracy {diagnostic["accuracy"]:.4f}, Macro F1 {diagnostic["macro_f1"]:.4f}. This checks how far surface cues can go without reading the premise.')
   path=ROOT/'results-hard'/'bootstrap.json'
   if path.exists():
    b=read_json(path)[chosen.removeprefix('mnli_')]
    st.caption(f'Premise-cluster bootstrap, DistilBERT seed42 minus pair SVM: Delta Macro F1 {b["delta_macro_f1"]:+.4f}, 95% interval [{b["ci95_low"]:+.4f}, {b["ci95_high"]:+.4f}]. Conditional on two trained models.')
  selected=st.selectbox('Inspect per-class results',[n for n in MODELS if n in f.model.unique()],format_func=lambda n:MODEL_LABELS[n])
  selected_rows=[r for r in rows if r['model']==selected];per_class=[]
  for label in LABELS[chosen]:
   vals=[r['classification_report'][label] for r in selected_rows]
   per_class.append({'Class':label,'Precision':np.mean([v['precision'] for v in vals]),'Recall':np.mean([v['recall'] for v in vals]),'F1':np.mean([v['f1-score'] for v in vals]),'Support':int(vals[0]['support'])})
  table=pd.DataFrame(per_class).set_index('Class');st.subheader('Per-class Precision, Recall and F1')
  st.dataframe(table.style.format({'Precision':'{:.4f}','Recall':'{:.4f}','F1':'{:.4f}'}),use_container_width=True)
  st.bar_chart(table[['Precision','Recall','F1']])
  st.subheader('Confusion matrix')
  representative=next((r for r in selected_rows if r['seed']==42),selected_rows[0]);cm=np.asarray(representative['confusion_matrix'])
  normalized=st.checkbox('Normalize each true-class row')
  if normalized:cm=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
  matrix=pd.DataFrame(cm,index=LABELS[chosen],columns=LABELS[chosen]);matrix.index.name='True class';matrix.columns.name='Predicted class'
  st.dataframe(matrix.style.format('{:.1%}' if normalized else '{:.0f}'),use_container_width=True)
  st.caption('Rows: true classes. Columns: predicted classes. Deep-model matrices use seed42, not averaged integer counts.')
  st.write('Macro weights each class equally; Weighted uses true class counts. The planned test sets are balanced, so Macro and Weighted scores coincide. Micro Precision, Recall and F1 equal Accuracy for single-label classification and are not independent evidence.')
  if chosen=='ag_news':st.write('News topics and sentiment are different tasks. Their scores are not a controlled measurement of the difficulty of binary versus multiclass classification.')
 if task=='Binary sentiment':
  with st.expander('Original cross-dataset sentiment transfer'):
   name=st.selectbox('Transfer model',['logistic_regression','linear_svm','distilbert'],format_func=lambda n:MODEL_LABELS[n])
   transfer=pd.DataFrame([r for r in data['rows'] if r['source'] in SOURCES[:3] and r['target'] in SOURCES[:3] and r['model']==name])
   st.dataframe(transfer.pivot_table(index='source',columns='target',values=average+'_f1').rename(index=DATASETS,columns=DATASETS).style.format('{:.4f}'),use_container_width=True)
   st.caption('Rows: training source. Columns: test target. Added Word2Vec/BiLSTM models have within-dataset evaluations only. News labels are incompatible with sentiment, so no cross-task transfer score is computed.')
 bench=ROOT/('results-hard' if chosen.startswith('mnli_') else 'results-v2')/'benchmark.json'
 if bench.exists():
  with st.expander('CPU cost on a common workload'):
   cost=pd.DataFrame([r for r in read_json(bench)['rows'] if r['source']==('mnli' if chosen.startswith('mnli_') else chosen)])
   if not cost.empty:st.dataframe(cost[['model','median_ms','p95_ms']].replace({'model':MODEL_LABELS}).rename(columns={'model':'Model','median_ms':'Median (ms)','p95_ms':'p95 (ms)'}),hide_index=True,use_container_width=True)
   st.caption('Same 200 matched sentence pairs, batch1, four threads, five repeats; loading excluded. This workload differs from the review benchmark.' if chosen.startswith('mnli_') else 'Same 200 review texts, batch1, four threads, five repeats. News-trained models use this common review workload too, not a news-serving benchmark. Legacy timings are reused.')
with robustness:
 st.subheader('Controlled training imbalance: AG News')
 st.write('World is the majority class. Each training condition has 6,000 unique texts. Validation and test stay fixed and balanced at 1,000 examples each, with 250 per class.')
 ratios={'balanced':[1500,1500,1500,1500],'mild':[4200,600,600,600],'moderate':[5100,300,300,300],'severe':[5640,120,120,120]}
 counts=pd.DataFrame.from_dict(ratios,orient='index',columns=LABELS['ag_news']);counts.index.name='Training condition'
 st.dataframe(counts,use_container_width=True)
 st.caption('Balanced checkpoints and predictions are reused. New subsets use fixed nested class orders. Hyperparameters remain those selected on balanced validation. This extension is exploratory after the original test inspection.')
 path=ROOT/'results-imbalance'/'evaluation.json'
 if not path.exists():st.info('The controlled robustness experiment is running. Results will appear when all final checkpoints have been evaluated.')
 else:
  records=read_json(path)['rows'];rf=pd.DataFrame(records);plain=rf[~rf.weighted]
  condition=st.selectbox('Training imbalance',list(ratios),format_func=lambda x:x.title()+' — '+str(int(100*ratios[x][0]/6000))+'% World')
  cols=['accuracy','macro_precision','macro_recall','macro_f1','mcc','minority_recall','minority_f1','worst_class_recall']
  means=plain.groupby(['condition','model'])[cols].mean()
  current=means.loc[condition].reindex(MODELS).copy();baseline=means.loc['balanced'].reindex(MODELS)
  current['Macro F1 decline (pp)']=100*(baseline.macro_f1-current.macro_f1)
  current['Minority recall decline (pp)']=100*(baseline.minority_recall-current.minority_recall)
  current['Macro F1 training SD']=plain[plain.condition==condition].groupby('model').macro_f1.std()
  current=current[['macro_f1','Macro F1 decline (pp)','minority_recall','Minority recall decline (pp)','accuracy','macro_precision','macro_recall','mcc','minority_f1','worst_class_recall','Macro F1 training SD']]
  labels={**METRIC_LABELS,'minority_recall':'Minority Recall','minority_f1':'Minority F1','worst_class_recall':'Worst-class Recall','Macro F1 decline (pp)':'F1 decline (pp)','Minority recall decline (pp)':'Recall decline (pp)'}
  st.subheader('Fixed balanced-test metrics and decline')
  st.dataframe(current.rename(index=MODEL_LABELS,columns=labels).style.format('{:.4f}',na_rep='—'),use_container_width=True)
  st.caption('Minority metrics average Sports, Business and Sci/Tech, including the balanced reference. Positive decline = own balanced score minus current score in percentage points. Deep results average three seeds; fixed traditional fits have no training SD.')
  view=st.selectbox('Deterioration metric',['macro_f1','minority_recall','minority_f1','worst_class_recall'],format_func=lambda x:labels[x])
  chart=plain.pivot_table(index='condition',columns='model',values=view).reindex(list(ratios)).rename(columns=MODEL_LABELS)
  st.line_chart(chart,x_label='Training condition: 1:1 / 7:1 / 17:1 / 47:1',y_label=labels[view]+' (0–1)')
  name=st.selectbox('Robustness model',MODELS,format_func=lambda x:MODEL_LABELS[x])
  chosen_rows=[r for r in records if r['condition']==condition and r['model']==name and not r['weighted']]
  per=[]
  for label in LABELS['ag_news']:
   vals=[r['classification_report'][label] for r in chosen_rows]
   per.append({'Class':label,'Precision':np.mean([v['precision'] for v in vals]),'Recall':np.mean([v['recall'] for v in vals]),'F1':np.mean([v['f1-score'] for v in vals]),'Test support':250})
  st.dataframe(pd.DataFrame(per).set_index('Class').style.format({'Precision':'{:.4f}','Recall':'{:.4f}','F1':'{:.4f}'}),use_container_width=True)
  r=next((r for r in chosen_rows if r['seed']==42),chosen_rows[0]);cm=pd.DataFrame(r['confusion_matrix'],index=LABELS['ag_news'],columns=LABELS['ag_news'])
  cm.index.name='True class';cm.columns.name='Predicted class';st.dataframe(cm,use_container_width=True)
  st.caption('Confusion counts use the fixed fit or seed42. Every true class has 250 test examples.')
  st.subheader('Class-weight mitigation at severe imbalance')
  weighted=rf[rf.weighted].groupby('model')[cols].mean();unweighted=means.loc['severe']
  weighted['Macro F1 gain (pp)']=100*(weighted.macro_f1-unweighted.macro_f1)
  weighted['Minority recall gain (pp)']=100*(weighted.minority_recall-unweighted.minority_recall)
  weighted=weighted[['macro_f1','Macro F1 gain (pp)','minority_recall','Minority recall gain (pp)','accuracy','macro_precision','macro_recall','mcc','minority_f1','worst_class_recall']]
  st.dataframe(weighted.rename(index=MODEL_LABELS,columns=labels).style.format('{:.4f}'),use_container_width=True)
  st.caption('SVM uses balanced class weights. DistilBERT retrains from the generic pretrained model with inverse-frequency weighted cross-entropy. Same severe data and training budgets. Positive gain = weighted minus unweighted severe score.')
  with st.expander('Changed-prior Accuracy simulation'):
   sim=plain[plain.condition==condition].groupby('model')[['accuracy','simulated_prior_weighted_accuracy','minority_recall']].mean()
   st.dataframe(sim.rename(index=MODEL_LABELS,columns={'accuracy':'Observed balanced-test Accuracy','simulated_prior_weighted_accuracy':'Simulated prior-weighted Accuracy','minority_recall':'Observed minority Recall'}).style.format('{:.4f}'),use_container_width=True)
   st.write('The simulation reweights the same test predictions to the selected training prior, assuming unchanged class-conditional text distributions. It is separate from the observed balanced-test metrics. At 94/2/2/2, an always-World rule has 94% simulated Accuracy, 25% balanced-test Accuracy, Macro F1 0.10 and zero minority Recall.')
  with st.expander('Scope and uncertainty'):
   bp=ROOT/'results-imbalance'/'bootstrap.json'
   if bp.exists():
    b=read_json(bp);st.dataframe(pd.DataFrame(b['rows']).replace({'model':MODEL_LABELS}),hide_index=True,use_container_width=True);st.caption(b['scope']+' '+b['contrast'])
   st.write('One corpus, one majority class and one subset draw limit generalization. Fixed total N controls sample count, while minority scarcity and changed training priors remain combined. Deep training SD does not include subset-selection uncertainty. On the balanced test, Accuracy equals balanced Accuracy and weighted F1 equals Macro F1.')
   st.write('Prediction demos elsewhere use the balanced checkpoints. Earlier CPU timing measurements remain the latency evidence; no new latency experiment is implied by the robustness curves.')
 st.markdown('[Balanced Accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) · [Balanced class weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)')
with methods:
 st.subheader('Course alignment')
 st.write('Traditional NLP: TF-IDF + SVM and Word2Vec mean features + SVM. Sequential deep learning: Word2Vec-initialized BiLSTM. Transformer: fine-tuned DistilBERT. TF-IDF Logistic Regression is an additional baseline.')
 st.write('The original four corpora have 6,000 training, 1,000 validation and 1,000 test examples each. MultiNLI adds sentence-pair semantics: 6,000 training, 999 source-validation and 999 examples in each reserved matched/mismatched evaluation pool.')
 st.write('MultiNLI models see both sentences. Sparse and mean-vector models concatenate premise, hypothesis, absolute difference and elementwise product features. A shared BiLSTM uses the same interaction structure; DistilBERT jointly encodes the pair. Normalized premises never cross partitions.')
 st.subheader('Selection and data isolation')
 st.write('Word2Vec, vocabulary and TF-IDF fit source training text only. Validation Macro F1 chooses learning rates and epochs. Deep models use three seeds; saved hashes fix the protocol.')
 st.write('V2 follows the original sentiment test inspection. Added binary comparisons are exploratory extensions; AG News has a newly reserved official-test sample. No test score selects a checkpoint.')
 st.subheader('Limitations')
 st.write('The course list offers possible approaches, rather than requiring every algorithm. CRF is a structured sequence-labeling method and is not needed for these document-level labels.')
 st.write('BiLSTM uses at most 256 word tokens; DistilBERT uses at most 256 subword tokens. The units differ. Word2Vec is trained locally; DistilBERT has external pretraining. These results do not isolate architecture alone.')
 st.write('Balanced sampling, corpus style, truncation and near-duplicates limit generalization. Member details and signatures remain outside this application.')
 st.subheader('Sources')
 st.markdown('[MultiNLI](https://aclanthology.org/N18-1101/) · [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) · [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) · [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) · [Yelp](https://huggingface.co/datasets/fancyzhx/yelp_polarity) · [DistilBERT](https://arxiv.org/abs/1910.01108)')

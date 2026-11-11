"""Export observed selected-checkpoint validation results; no training or Q use."""
import csv
import hashlib
import json
from pathlib import Path

root=Path('research-v3');folder=root/'reports/overnight'
base=json.loads((root/'training/overnight/run-plan.json').read_text(encoding='utf-8'))
extra=json.loads((root/'training/overnight/extension-run-plan.json').read_text(encoding='utf-8'))
snapshot=json.loads((folder/'latest-cluster-results.json').read_text(encoding='utf-8'))
design=json.loads((root/'matrix/design_manifest.json').read_text(encoding='utf-8'))
conditions={row['condition_id']:row for row in design['conditions']}
selected={row['entry']['run_id']:row for row in base['runs']+extra['runs']}
assert len(selected)==201
observed={row['run_id']:row for row in snapshot['completed']}
records=[]
for run_id,row in selected.items():
    entry=row['entry'];condition=conditions[entry['condition_id']]
    if row['completed_checkpoint_reuse']:
        path=root/'runs'/run_id/'status.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['status_sha256']
        status=json.loads(path.read_text(encoding='utf-8'))
        metrics=status.get('validation',{})
    elif run_id in observed:
        metrics=observed[run_id]
    else:
        continue
    metric='macro_ap' if entry['dataset_id']=='go_emotions' else 'macro_f1'
    records.append({'run_id':run_id,'dataset':entry['dataset_id'],'model':entry['model_id'],
        'n':row['training_rows'],'data_seed':entry['data_seed'],'training_seed':entry['training_seed'],
        'condition_id':entry['condition_id'],'condition_experiment':condition['experiment_id'],
        'aliases':','.join(sorted({alias['experiment_id'] for alias in entry['semantic_aliases']})),
        'metric':metric,'value':metrics.get(metric),'requested_ratio':condition['metadata'].get('requested_ratio'),
        'arm':entry['run']['hparams'].get('arm'),
        'evidence_role':'exploratory_V_selected_checkpoint_not_independent_test'})
assert len(records)==67+len(snapshot['completed'])
with (folder/'current-validation-results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,7.5))
titles={'yelp':'Sentiment: Yelp','go_emotions':'Emotion: GoEmotions','banking77':'Intent: BANKING77','yahoo_answers_topics':'Topic: Yahoo'}
colors={'tfidf_svm':'#2962a3','bilstm':'#d28b26','distilbert':'#24966f'}
names={'tfidf_svm':'TF-IDF SVM','bilstm':'BiLSTM','distilbert':'DistilBERT'}
points=[]
for ax,(ds,title) in zip(axes.flat,titles.items()):
    for model,color in colors.items():
        rows=sorted((row for row in records if row['dataset']==ds and row['model']==model
            and row['data_seed']==2026 and row['training_seed'] in (None,42)
            and 'E1' in row['aliases'].split(',') and row['value'] is not None),key=lambda x:x['n'])
        assert len(rows)==len({row['n'] for row in rows})
        ax.plot([row['n'] for row in rows],[row['value'] for row in rows],marker='o',color=color,label=names[model])
        points.extend(rows)
    ax.set_xscale('log');ax.set_title(title);ax.grid(alpha=.2)
    ax.set_xlabel('Training examples (log scale)');ax.set_ylabel('Macro-AP' if ds=='go_emotions' else 'Macro-F1')
    ax.set_ylim(bottom=0);ax.legend(fontsize=8)
fig.suptitle('Observed learning curves: exploratory validation\nFixed primary data/training seeds; missing conditions omitted',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92))
fig.savefig(folder/'current-learning-curves.png',dpi=200)
fig.savefig(folder/'current-learning-curves.pdf')
plt.close(fig)

with (folder/'current-imbalance-comparisons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['dataset','model','training_n','balanced_macro_f1','skew10_macro_f1','difference_skew_minus_balanced','evidence_role'])
    for ds in ('yelp','banking77','yahoo_answers_topics'):
        for model in colors:
            rows={row['requested_ratio']:row for row in records if row['dataset']==ds and row['model']==model
                and row['condition_experiment']=='E3' and row['data_seed']==2026 and row['training_seed'] in (None,42)}
            if 1 in rows and 10 in rows:
                a,b=rows[1],rows[10];assert a['n']==b['n']
                writer.writerow([ds,model,a['n'],a['value'],b['value'],b['value']-a['value'],'exploratory_V_not_pure_prior_intervention'])
print(json.dumps({'selected_requests':201,'completed_selected':len(records),'old_reused':67,'new_complete':len(snapshot['completed']),'learning_curve_points':len(points),'formal_Q_predictions_generated':False}))

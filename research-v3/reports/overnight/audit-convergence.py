"""Read saved trajectories and SVM solver records; never fit or score a model."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
sys.path.insert(0,str(ROOT.parent))
with (FOLDER/'charts-v1/all-completed-records.csv').open(encoding='utf-8-sig',newline='') as f:
    records=list(csv.DictReader(f))
selected=[r for r in records if r['experiment']=='E1' and r['n'] in ('1000','6000')
          and r['model'] in ('bilstm','distilbert') and r['data_seed']=='2026' and r['training_seed']=='42']
assert len(selected)==16
audit=[];histories={};sources={}
for row in selected:
    directory=ROOT/'runs'/row['run_id']
    item={k:row[k] for k in ('dataset','model','n','run_id')}
    item['history_available_locally']=(directory/'history.json').exists()
    if item['history_available_locally']:
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        history=json.loads((directory/'history.json').read_text(encoding='utf-8'))['epochs']
        assert len(history)==status['epochs_completed']
        metric='macro_ap' if row['dataset']=='go_emotions' else 'macro_f1'
        values=[r['validation'][metric] for r in history]
        assert abs(max(values)-status['validation'][metric])<1e-12
        item.update(epochs=status['epochs_completed'],selected_epoch=status['selected_epoch'],
            optimizer_updates=status['optimizer_updates'],first_train_loss=history[0]['train_loss'],
            last_train_loss=history[-1]['train_loss'],last_validation=values[-1],
            penultimate_validation=values[-2],best_validation=max(values),
            best_at_budget_end=status['selected_epoch']==status['epochs_completed'],
            assessment='budget_end_flag_not_proof_of_undertraining' if status['selected_epoch']==status['epochs_completed']
                       else 'earlier_validation_peak_not_proof_of_full_optimization')
        histories[row['dataset'],row['model'],row['n']]=history
        for name in ('status.json','history.json'):
            p=directory/name;sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    audit.append(item)
with (FOLDER/'convergence-audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=list(dict.fromkeys(k for row in audit for k in row))
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(audit)
solvers=[]
for row in records:
    if row['model']!='tfidf_svm' or row['n']!='6000' or row['data_seed']!='2026' or row['experiment']!='E1':continue
    directory=ROOT/'runs'/row['run_id']
    status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    bundle=joblib.load(directory/'bundle.joblib')
    model=bundle['model']
    estimators=model.estimators if model.task in ('multilabel','multi_label') else [model.estimator]
    estimators=[e for e in estimators if e is not None]
    iterations=[int(np.asarray(e.n_iter_).max()) for e in estimators]
    assert all(n<int(e.get_params()['max_iter']) for n,e in zip(iterations,estimators))
    solvers.append({'dataset':row['dataset'],'run_id':row['run_id'],'fitted_estimators':len(estimators),
        'maximum_observed_solver_iterations':max(iterations),'max_iter':estimators[0].get_params()['max_iter'],
        'tol':estimators[0].get_params()['tol'],
        'interpretation':'solver stopped below iteration cap; not a neural epoch count'})
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,ds in enumerate(('yelp','go_emotions')):
    for j,model in enumerate(('bilstm','distilbert')):
        ax=axes[i,j];history=histories[ds,model,'6000']
        metric='macro_ap' if ds=='go_emotions' else 'macro_f1'
        values=np.array([r['validation'][metric] for r in history]);epochs=np.arange(1,len(history)+1)
        best=int(values.argmax())
        color='#2872A8' if model=='bilstm' else '#168B78'
        ax.plot(epochs,values,marker='o',linewidth=2.5,color=color)
        ax.scatter(epochs[best],values[best],s=130,facecolors='none',edgecolors='#CB592C',linewidth=2,zorder=5)
        ax.axvline(len(history),color='#9AA6B2',linestyle=':',linewidth=1.5)
        status='最优仍在训练上限' if best==len(history)-1 else '最优早于训练上限'
        ax.set_title(f"{ds}: {'BiLSTM' if model=='bilstm' else 'DistilBERT'}\n{status}（第{best+1}轮）",loc='left')
        ax.set_xticks(epochs)
        ax.set_xlabel('实际训练轮次 epoch')
        ax.set_ylabel('V '+('Macro-AP' if ds=='go_emotions' else 'Macro-F1'))
        ax.grid(alpha=.15)
        ax.text(.98,.06,f'最好 {values[best]:.4f}\n最后 {values[-1]:.4f}',
                transform=ax.transAxes,ha='right',fontsize=11)
fig.suptitle('训练上限是否限制了结果？先看实际验证曲线\n各图均为6,000条训练；橙圈标出最优轮次，虚线标出预算终点。',fontsize=17,linespacing=1.6)
fig.supxlabel('预算终点仍最好是风险标记，不等于已证明欠训练；更早达到峰值也不等于已找到最优训练配方。',fontsize=9)
figure=FOLDER/'charts-research/05-训练预算与验证曲线.png'
fig.savefig(figure,dpi=160,bbox_inches='tight');plt.close(fig)
summary={'primary_neural_conditions':len(audit),'histories_available_locally':sum(r['history_available_locally'] for r in audit),
    'best_at_budget_end':sum(r.get('best_at_budget_end',False) for r in audit),
    'missing_histories_are_not_classified':True,'neural_conditions':audit,'svm_solvers':solvers,
    'source_sha256':sources,'new_training_started':False,
    'conclusion':'Fixed-recipe observations; sufficient optimization has not been established.'}
(FOLDER/'convergence-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('neural_conditions','source_sha256')},ensure_ascii=False))

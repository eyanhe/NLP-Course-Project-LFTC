"""Build a local, analysis-ready snapshot; never run models or access Q."""
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];ROOT=PROJECT/'research-v3'
sys.path.insert(0,str(PROJECT))
from research_v3.storage import read_json,hash_file
from research_v3.config import canonical_hash
from research_v3.partitions import clean_text
import pyarrow.parquet as pq

snapshot=read_json(HERE/'download-manifest.json')
history=read_json(ROOT/'training/cluster-migration/verified-checkpoints.json')
main=read_json(ROOT/'training/fixed-main-run-manifest.json')
aux=read_json(ROOT/'reports/overnight/supplement-1-4/manifest.json')
prepared=read_json(ROOT/'data/prepared_manifest.json')
design=read_json(ROOT/'matrix/design_manifest.json')
conditions={r['condition_id']:r for r in design['conditions']}
original={r['run_id']:r for r in main['runs']}
additional={r['entry']['run_id']:r for r in aux['fits']}
verified_history={r['run_id']:r for r in history['formal_checkpoints']}
all_ids=set(verified_history)|set(snapshot['run_states'])|set(additional)
index=[];conflicts=[];copied=[]

def copy_resource(path,relative):
    target=HERE/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    if hash_file(path)!=hash_file(target):raise ValueError('Copied resource changed')
    copied.append(dict(source=path.relative_to(PROJECT).as_posix(),file=relative,sha256=hash_file(target),bytes=target.stat().st_size))

for rid in sorted(all_ids):
    remote=HERE/'cluster-evidence/runs'/rid/'status.json';local=ROOT/'runs'/rid/'status.json'
    if rid in verified_history:
        if hash_file(local)!=verified_history[rid]['status_sha256']:raise ValueError('Historical status changed')
    if remote.exists() and local.exists():
        a=read_json(remote);b=read_json(local)
        if a.get('status')==b.get('status')=='complete' and hash_file(remote)!=hash_file(local):
            conflicts.append(rid)
    source=remote if remote.exists() else local
    if not source.exists():continue
    status=read_json(source);run=status.get('run',{});cid=run.get('condition_id');condition=conditions.get(cid,{})
    extra=additional.get(rid);entry=extra['entry'] if extra else original.get(rid,{})
    dataset=extra['dataset_id'] if extra else entry.get('dataset_id',condition.get('dataset_id'))
    scope=extra['supplement_id'] if extra else 'original_fixed_manifest' if rid in original else 'historical_other'
    destination=HERE/'cluster-evidence/runs'/rid
    if source==local:
        copy_resource(local,f'cluster-evidence/runs/{rid}/status.json')
        if (local.parent/'history.json').exists():
            expected=status.get('files',{}).get('history.json')
            if expected and hash_file(local.parent/'history.json')!=expected:raise ValueError('Historical history bytes changed')
            copy_resource(local.parent/'history.json',f'cluster-evidence/runs/{rid}/history.json')
    validation=status.get('validation') or {}
    index.append(dict(run_id=rid,scope=scope,status=status.get('status'),reload_verified=status.get('reload_verified',False),
        dataset_id=dataset,model_id=run.get('model_id',entry.get('model_id')),condition_id=cid,
        experiment_id=condition.get('experiment_id'),training_rows=status.get('training_unique_rows',condition.get('metadata',{}).get('rows')),
        data_seed=condition.get('data_seed',entry.get('data_seed')),training_seed=run.get('training_seed'),
        current_four_task_scope=dataset in {'yelp','go_emotions','banking77','yahoo_answers_topics'},
        epochs=status.get('epochs_completed'),selected_epoch=status.get('selected_epoch'),optimizer_updates=status.get('optimizer_updates'),
        macro_f1=validation.get('macro_f1'),macro_ap=validation.get('macro_ap'),accuracy=validation.get('accuracy'),
        validation_rows=validation.get('rows'),total_seconds=status.get('total_seconds'),
        hparams=run.get('hparams'),status_file=f'cluster-evidence/runs/{rid}/status.json',
        history_file=f'cluster-evidence/runs/{rid}/history.json' if (destination/'history.json').exists() else None,
        metrics_are_final=status.get('status')=='complete' and status.get('reload_verified') is True))
if conflicts:raise ValueError(f'Conflicting complete status versions: {conflicts}')
(HERE/'run-index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in index),encoding='utf-8')

frame=read_json(ROOT/'reports/overnight/failure-frame-plan.json')
frozen=read_json(HERE/'cluster-evidence/reports/overnight/supplement-1-4/S4-ai-reviewed/inputs-frozen.json')
s4_manifest=read_json(HERE/'cluster-evidence/reports/overnight/supplement-1-4/S4-ai-reviewed/result-manifest.json')
s4_result=HERE/'cluster-evidence/reports/overnight/supplement-1-4/S4-ai-reviewed/results.json'
assert hash_file(s4_result)==s4_manifest['results_sha256']
assert canonical_hash({k:v for k,v in frozen.items() if k!='freeze_hash'})==frozen['freeze_hash']==s4_manifest['freeze_hash']
case_rows=[]
for ds,f in frame['frames'].items():
    meta=prepared['datasets'][ds];ids=set(f['sample_ids']);s4ids=set()
    if ds=='banking77':s4ids={r['pair']['sample_id'] for r in frozen['pairs']};ids|=s4ids
    assert hash_file(ROOT/f['source_file'])==f['source_sha256']
    actual={r['sample_id']:r for r in pq.read_table(ROOT/f['source_file'],columns=['sample_id','group_id','labels','text'],
                                                filters=[('sample_id','in',sorted(ids))]).to_pylist()}
    visible=pq.read_table(ROOT/meta['inputs']['file'],columns=['sample_id','text','text_hash','content_tokens','full_tokens'],
                          filters=[('sample_id','in',sorted(ids))]).to_pylist()
    assert {r['sample_id'] for r in visible}==ids==set(actual)
    for row in sorted(visible,key=lambda r:r['sample_id']):
        sid=row['sample_id'];original_row=actual[sid]
        assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_hash']
        case_rows.append(dict(dataset_id=ds,sample_id=sid,group_id=original_row['group_id'],source_partition='discovery',
            task=meta['task'],label_ids=original_row['labels'],label_names=[meta['label_names'][n] for n in original_row['labels']],
            common_visible_text=row['text'],source_full_text_normalized=clean_text(original_row['text']),
            input_text_hash=row['text_hash'],content_tokens=row['content_tokens'],full_tokens=row['full_tokens'],
            in_S2_frame=sid in f['sample_ids'],in_S4_base=sid in s4ids))
(HERE/'discovery-texts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in case_rows),encoding='utf-8')

for record in aux['trajectory_sources']:
    rid=record['entry']['run_id'];path=HERE/f'cluster-evidence/reports/overnight/supplement-1-4/S2-predictions/{rid}.json'
    saved=read_json(path);assert saved['run_id']==rid
    assert len(saved['rows'])==len(frame['frames'][record['dataset_id']]['sample_ids'])
    assert saved['status_sha256']==hash_file(HERE/f'cluster-evidence/runs/{rid}/status.json')

for source,relative in [
    (ROOT/'data/prepared_manifest.json','protocol/prepared_manifest.json'),
    (ROOT/'matrix/design_manifest.json','protocol/design_manifest.json'),
    (ROOT/'training/fixed-main-run-manifest.json','protocol/fixed-main-run-manifest.json'),
    (ROOT/'training/overnight/run-plan.json','protocol/overnight-run-plan.json'),
    (ROOT/'training/overnight/extension-run-plan.json','protocol/extension-run-plan.json'),
    (ROOT/'reports/overnight/supplement-1-4/manifest.json','protocol/supplement-manifest.json'),
    (ROOT/'reports/overnight/results-analysis-v1-zh.md','reference/existing-analysis-zh.md'),
    (PROJECT/'docs/superpowers/specs/2026-11-07-text-classification-research-design.md','protocol/original-research-design.md'),
    (PROJECT/'docs/superpowers/plans/2026-11-08-ai-reviewed-m1.md','protocol/AI-reviewed-M1-amendment.md'),
    (ROOT/'reports/overnight/supplement-1-4/human-review/manifest.json','protocol/review-source-manifest.json'),
    (PROJECT/'outputs/supplement14-human-review-20261108/build/submitted-review-audit.json','protocol/submitted-review-audit.json'),
    (PROJECT/'outputs/supplement14-human-review-20261108/build/completed-review-readonly.json','protocol/submitted-form-values.json')]:
    copy_resource(source,relative)

aux_counts=Counter(r['status'] for r in index if r['scope'] in {'S1','S3'})
utc=snapshot['snapshot_at_utc'];sg=datetime.fromisoformat(utc).astimezone(timezone(timedelta(hours=8))).isoformat()
counts=dict(unique_run_records=len(index),completed_unique_run_records=sum(r['metrics_are_final'] for r in index),
    supplemental_training_status=dict(aux_counts),S2_prediction_checkpoints=24,S4_AI_reviewed_pairs=37,
    discovery_text_records=len(case_rows),complete_status_conflicts=conflicts)
summary=dict(snapshot_at_utc=utc,snapshot_at_singapore=sg,counts=counts,
    remote_snapshot_manifest='download-manifest.json',model_weights_fully_downloaded=False,
    raw_training_and_test_corpora_in_this_bundle=False,joined_discovery_texts_included=True,
    original_local_corpora_location='research-v3/data and research-v3/inputs',
    Q_predictions_generated=False,review_source='User-accepted Gemini-assisted review, not two independent humans',
    live_results_are_provisional=bool(aux_counts.get('running') or aux_counts.get('verifying')),
    snapshot_is_immutable_complete_study=False)
(HERE/'handoff-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

readme=f'''# 给另一个AI的本地分析入口

数据快照时间（新加坡）：{sg}。这是可以直接分析的结果快照，不是集群全量镜像。

研究方向：**文本分类模型在不同数据条件下的能力边界与失败机制研究**。
请基于原始指标、同一样本预测与文本独立完成定量、定性分析。既有结论仅作参考。请直接陈述实验范围内得到的比较结论，注明具体模型、任务与条件。

## 当前材料

- `run-index.jsonl`：每个唯一训练检查点一行，含范围、数据集、模型、规模、种子、超参数、指标及原状态/曲线路径。共{len(index)}条，{sum(r['metrics_are_final'] for r in index)}条完成并通过模型重载核验；这不是同样数量的独立实验。
- `cluster-evidence/runs/<run_id>/status.json`：原始指标，包括逐类统计和资源记录；`history.json`为逐轮损失、V指标、更新数。历史本机结果也已按同结构并入。
- `cluster-evidence/reports/overnight/supplement-1-4/S2-predictions/`：四任务三模型×1k/6k的24组完整D预测；同级`S2-summary.json`有12组错误轨迹。
- `discovery-texts.jsonl`：{len(case_rows)}条相关D样本的完整文本、实际可见文本、金标和组ID，用`dataset_id + sample_id`与预测连接，无需另加载大语料。
- `cluster-evidence/reports/overnight/supplement-1-4/S4-ai-reviewed/`：固定37对原文/线索改写/匹配控制、审核选择依据、SVM及DistilBERT逐对预测和结果清单。
- `cluster-evidence/reports/overnight/discovery-cases/`：此前80条SVM失败候选与线性贡献、4,604条跨条件预测。它与S2、S4部分不同，不能将行数直接相加。
- `protocol/`：原设计、原矩阵、一晚范围、补充实验、AI审核修订、准备清单、标签映射和填写表原始值。
- `reference/existing-analysis-zh.md`：当前解释稿；请用原始文件核实数值和推论。
- `download-manifest.json`：远端实际下载文件、大小、SHA、未取回的大型模型/序列化特征清单；`live-cluster-state.txt`为当时队列。

## 范围与计数

主线四任务：Yelp、GoEmotions、BANKING77、Yahoo。IMDB/AG News/Rotten Tomatoes属于原有历史覆盖，不等于本轮新增。
原固定矩阵2859条；一晚选中201条（67复用、134新增上限）；独立CPU补充S1/S3合计24条。
本快照CPU补充完成状态：{dict(aux_counts)}。`running`的指标与曲线为进行中记录，不当最终成绩；后续仍会产生新文件。
S2与S4是现有检查点推理，不额外计为新训练。

## 分析要点

1. 单标签主要比较Macro-F1，多标签主要比较Macro-AP；V用于调参与选epoch，D用于诊断，请保持用途区分。
2. 比较同任务、同评估集和数据条件；E1规模对照与E3均衡基线不能混用。原始配方和补充优化配方分开。
3. S1预算变动包含相应的线性学习率调度；固定数据/学习率的5/10轮比较不等于固定调度下纯epoch效应。
4. S3是BANKING77固定表示2×2：完整/尾部删减支持×均匀/偏斜类别贡献。冻结表示分类头与端到端DistilBERT分别分析。
5. S4审核来源为Gemini两份表格版本；用户接受其判断。A表前3条保留用户人工批注，不能称两位独立真人。37对在产生编辑预测前固定。
6. S4新增错误率以各模型原文答对样本为分母。比较模型改写效果时可用共同37条上的准确率变化；不同模型原文正确子集不同。
7. BANKING77未预留S2诊断框仅23条；S4另37条是AI审核可改写子集。明确结论针对的数据范围。
8. E3/M3等条件若未提供完整逐样本预测，只从已有指标、逐类统计作分析；不要将16条训练重载probe当评估集。

这份包不含模型权重、运行环境、缓存和全部训练/测试原始语料，也没有正式Q预测。若只做现有结果的定量/定性分析，可直接开始；若要对新文本推理，需要另取对应模型权重。
本次任务是分析现有材料，继续使用现有训练结果，无需重新提交集群实验。连接/凭据脚本没有收进分析包。
'''
(HERE/'README.md').write_text(readme,encoding='utf-8')

files=[]
for path in sorted(HERE.rglob('*')):
    if not path.is_file() or path.suffix in {'.zip','.part'} or path.name in {'download-evidence.py','build-handoff.py','file-index.json'}:continue
    files.append(dict(path=path.relative_to(HERE).as_posix(),bytes=path.stat().st_size,sha256=hash_file(path)))
(HERE/'file-index.json').write_text(json.dumps(dict(files=files),ensure_ascii=False,indent=2),encoding='utf-8')
archive=HERE/'analysis-ready.zip';temporary=HERE/'analysis-ready.zip.part'
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for record in files:z.write(HERE/record['path'],arcname=record['path'])
    z.write(HERE/'file-index.json',arcname='file-index.json')
temporary.replace(archive)
with zipfile.ZipFile(archive) as z:
    for record in files:
        assert hashlib.sha256(z.read(record['path'])).hexdigest()==record['sha256']
print(json.dumps(dict(**counts,snapshot_at_singapore=sg,archive_bytes=archive.stat().st_size,
                     archive_sha256=hash_file(archive),verified_archive_members=len(files)+1),ensure_ascii=False))

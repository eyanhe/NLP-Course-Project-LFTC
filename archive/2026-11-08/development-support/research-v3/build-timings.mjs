import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const out=path.resolve('outputs/training-inference-timings-20261108');
const support=path.join(out,'support');await fs.mkdir(support,{recursive:true});
const data=JSON.parse(await fs.readFile('research-v3/reports/overnight/convergence-v1/training-inference-timings.json','utf8'));
const wb=Workbook.create();const main=wb.worksheets.add('训练与推理耗时');const raw=wb.worksheets.add('计时明细');
const fields=[
 ['case_id','记录ID'],['dataset_id','数据集'],['model_id','模型'],['training_rows','训练样本数'],['role','用途'],
 ['parameter','学习率或C'],['training_seed','训练种子'],['status','训练状态'],
 ['fit_seconds_last_segment','拟合末段（秒）'],['feature_seconds_last_segment','特征末段（秒）'],
 ['pipeline_seconds_last_segment','流程末段（秒）'],['prior_interrupted_wall_seconds','中断前重建（秒）'],
 ['training_pipeline_wall_seconds','累计训练流程（秒）'],['training_time_source','累计计时来源'],
 ['epochs_completed','完成轮数'],['training_cpu_threads','训练线程'],['training_device','训练设备'],['training_node_last_segment','最近训练节点'],
 ['verification_bundle_seconds','校验及bundle（秒）'],['model_load_seconds','模型加载（秒）'],['predict_seconds','批预测（秒）'],
 ['inference_cleanup_seconds','推理清理（秒）'],['authenticated_inference_seconds','推理端到端（秒）'],
 ['test_rows','测试样本数'],['rows_per_second','吞吐（条/秒）'],['milliseconds_per_row','每条（毫秒）'],['inference_status','推理状态'],['notes','说明']];
const n=data.rows.length;assert.equal(n,30);const rawEnd=3+n,mainEnd=8+n;
raw.showGridLines=false;main.showGridLines=false;main.tabColor='#17365D';
raw.getRange(`A1:AB${rawEnd}`).format.font={name:'Arial',size:10,color:'#263746'};
raw.getRange(`A1:AB${rawEnd}`).format.verticalAlignment='center';
raw.getRange('A1').values=[['来源：集群训练日志、恢复记录、Slurm作业计时与正式测试计时。I—K为日志字段，L—M为恢复对账结果。']];
raw.getRange('A2').values=[['历史日志未逐项记录缓存冷热，所列为实际执行耗时；不包含预训练模型的预训练成本。']];
raw.getRange(`A3:AB${rawEnd}`).values=[fields.map(x=>x[1]),...data.rows.map(r=>fields.map(([key])=>r[key]??null))];
raw.tables.add(`A3:AB${rawEnd}`,true,'TimingDetails').showFilterButton=true;
raw.freezePanes.freezeRows(3);raw.freezePanes.freezeColumns(3);
const rawWidths=[400,160,115,100,155,110,85,125,130,130,135,140,150,270,95,95,95,160,150,145,130,135,150,110,140,135,170,520];
rawWidths.forEach((w,i)=>raw.getRangeByIndexes(0,i,rawEnd,1).format.columnWidthPx=w);
raw.getRange(`A4:AB${rawEnd}`).format.rowHeight=30;raw.getRange(`AB4:AB${rawEnd}`).format.wrapText=true;
for(const c of ['I','J','K','L','M','S','T','U','V','W','Y','Z'])raw.getRange(`${c}4:${c}${rawEnd}`).setNumberFormat('0.000');
for(let i=0;i<n;i++)raw.getRange(`F${i+4}`).setNumberFormat(data.rows[i].model_id==='tfidf_svm'?'0.0':'0.00000');
main.getRange(`A1:N${mainEnd}`).format.font={name:'Arial',size:10,color:'#263746'};
main.getRange(`A1:N${mainEnd}`).format.verticalAlignment='center';
main.getRange('A2').values=[['训练与推理耗时']];main.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#17365D'};
main.getRange('A2:N2').format.rowHeight=28;main.getRange('A2:N2').format.borders={bottom:{style:'thin',color:'#B4C1CF'}};
const asOf=new Date(Date.parse(data.recorded_at_utc)+8*3600000).toISOString().slice(0,16).replace('T',' ');
main.getRange('A3').values=[[`数据截至 ${asOf}（SGT）。本表含26个本轮请求及4个历史参考。`]];
main.getRange('A4').values=[['本轮完成记录']];main.getRange('B4').formulas=[[`=COUNTIFS('计时明细'!H4:H${rawEnd},"complete",'计时明细'!E4:E${rawEnd},"<>历史固定预算参考")`]];
main.getRange('D4').values=[['已测正式推理']];main.getRange('E4').formulas=[[`=COUNTIFS('计时明细'!AA4:AA${rawEnd},"measured")`]];
main.getRange('A5').values=[['训练流程含特征处理、训练、逐轮验证、模型保存与重载。续跑累计值按已有日志重建并标注。']];
main.getRange('A6').values=[['推理取首次实际批预测，含文本处理。校验和加载另计。空白表示尚未执行或无法完整核实。']];
main.getRange('A3:N6').format.rowHeight=22;main.getRange('A3:N6').format.wrapText=false;
for(const range of ['A3:N3','A5:N6'])main.getRange(range).format.font={name:'Arial',size:10,color:'#617286'};
const headers=['数据集','模型','训练样本','用途','学习率/C','种子','训练状态','完成轮数','训练流程（分）','批预测（秒）','测试样本','每条（毫秒）','模型加载（秒）','说明'];
main.getRange('A8:N8').values=[headers];
const states={complete:'已完成',running:'进行中',verifying:'重载校验中',not_started:'待启动',awaiting_resume:'等待续跑',failed:'失败',incomplete:'未完成'};
const names={tfidf_svm:'TF-IDF + SVM',bilstm:'BiLSTM',distilbert:'DistilBERT'};
for(let i=0;i<n;i++){
 const rr=i+4,r=i+9,row=data.rows[i],ref=c=>`'计时明细'!${c}${rr}`;
 main.getRange(`A${r}:N${r}`).values=[[row.dataset_id,names[row.model_id],row.training_rows,row.role,null,null,states[row.status]??row.status,null,null,null,null,null,null,row.notes]];
 const numeric=(target,source,divisor=null)=>main.getRange(`${target}${r}`).formulas=[[`=IF(ISNUMBER(${ref(source)}),${ref(source)}${divisor?'/'+divisor:''},"")`]];
 main.getRange(`E${r}`).formulas=[[`=IF(ISNUMBER(${ref('F')}),${ref('F')},"待选")`]];
 main.getRange(`F${r}`).formulas=[[`=IF(ISNUMBER(${ref('G')}),${ref('G')},"固定")`]];
 numeric('H','O');numeric('I','M',60);numeric('J','U');numeric('K','X');numeric('L','Z');numeric('M','T');
 main.getRange(`A${r}:N${r}`).format.fill=i%2?'#F1F4F8':'#FFFFFF';
}
main.tables.add(`A8:N${mainEnd}`,true,'TrainingInferenceTimes').showFilterButton=true;
const widths=[150,125,90,150,100,65,110,90,145,135,100,140,145,520];
widths.forEach((w,i)=>main.getRangeByIndexes(0,i,mainEnd,1).format.columnWidthPx=w);
main.getRange(`A9:N${mainEnd}`).format.rowHeight=30;main.getRange(`N9:N${mainEnd}`).format.wrapText=true;
for(let i=0;i<n;i++)main.getRange(`E${i+9}`).setNumberFormat(data.rows[i].model_id==='tfidf_svm'?'0.0':'0.00000');
for(const column of ['C','E'])main.getRange(`${column}9:${column}${mainEnd}`).format.borders={right:{style:'thin',color:'#DDE4EA'}};
main.getRange(`F9:F${mainEnd}`).format.horizontalAlignment='center';
main.getRange(`I9:I${mainEnd}`).setNumberFormat('0.0000');
main.getRange(`J9:J${mainEnd}`).setNumberFormat('0.000');main.getRange(`L9:M${mainEnd}`).setNumberFormat('0.000');
for(const [s,range] of [[main,'A8:N8'],[raw,'A3:AB3']]){
 s.getRange(range).format={fill:'#17365D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:34};
 s.getRange(range).format.borders={insideVertical:{style:'thin',color:'#FFFFFF'}};
}
main.getRange(`G9:G${mainEnd}`).conditionalFormats.add('containsText',{text:'进行中',format:{fill:'#FFF2CC'}});
main.freezePanes.freezeRows(8);main.freezePanes.freezeColumns(2);
wb.recalculate();
for(let i=0;i<n;i++){
 const r=i+9,v=main.getRange(`I${r}:M${r}`).values[0],row=data.rows[i];
 if(row.training_pipeline_wall_seconds===null)assert.ok(v[0]===null||v[0]==='');
 else assert.ok(Math.abs(v[0]-row.training_pipeline_wall_seconds/60)<1e-8);
 if(row.predict_seconds===null)assert.ok(v[1]===null||v[1]==='');
 else assert.ok(Math.abs(v[1]-row.predict_seconds)<1e-8);
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1200});
console.log(JSON.stringify({records:n,formulaErrors:errors.ndjson}));
const summary=await wb.inspect({kind:'table',range:'训练与推理耗时!A8:M13',include:'values,formulas',tableMaxRows:6,tableMaxCols:13,maxChars:1800});
await fs.writeFile(path.join(support,'inspection.ndjson'),summary.ndjson);
for(const [sheet,range,file] of [['训练与推理耗时','A2:M14','overview.png'],['计时明细','I3:R12','timing-detail.png'],['计时明细','S3:AB12','inference-detail.png']]){
 const blob=await wb.render({sheetName:sheet,range,scale:1.3,format:'png'});
 await fs.writeFile(path.join(support,file),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'训练与推理耗时记录.xlsx'));
try{await fs.rename(path.join(out,'训练与推理耗时记录.xlsx.inspect.ndjson'),path.join(support,'训练与推理耗时记录.xlsx.inspect.ndjson'));}catch(error){if(error.code!=='ENOENT')throw error;}
await fs.writeFile(path.join(support,'source-timings.json'),JSON.stringify(data,null,2));
console.log(JSON.stringify({saved:path.join(out,'训练与推理耗时记录.xlsx'),records:n,measuredInference:data.rows.filter(r=>r.predict_seconds!==null).length}));

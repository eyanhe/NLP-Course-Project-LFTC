import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const data=JSON.parse(await fs.readFile('research-v3/reports/overnight/analysis-v2/analysis-data.json','utf8'));
const out=path.resolve('outputs/research-analysis-20261108');await fs.mkdir(out,{recursive:true});
const support=path.join(out,'support');await fs.mkdir(support,{recursive:true});
const wb=Workbook.create(),summary=wb.worksheets.add('模型Q比较'),effects=wb.worksheets.add('对照差值'),raw=wb.worksheets.add('42模型Q明细'),vd=wb.worksheets.add('V与D证据');
const names={tfidf_svm:'SVM',bilstm:'BiLSTM',distilbert:'DistilBERT',fixed_tfidf_svm:'固定TF-IDF + SVM',frozen_generic_distilbert_head:'冻结编码器 + 分类头'};
const ds={banking77:'BANKING77',go_emotions:'GoEmotions'};
const translate=s=>s.replaceAll('frozen_generic_distilbert_head','冻结编码器+分类头').replaceAll('fixed_tfidf_svm','固定TFIDF+SVM').replaceAll('distilbert','DistilBERT').replaceAll('bilstm','BiLSTM').replaceAll('go_emotions','GoEmotions').replaceAll('banking77','BANKING77').replaceAll('tail_reduced','标注删减').replaceAll('uniform','贡献均匀').replaceAll('skew','贡献偏斜').replaceAll('full','完整支持');
for(const s of [summary,effects,raw,vd]){s.showGridLines=false;s.getRange('A1:V140').format.font={name:'Arial',size:10,color:'#263746'};s.getRange('A1:V140').format.verticalAlignment='center';}
function header(s,range,labels){s.getRange(range).values=[labels];s.getRange(range).format={fill:'#17365D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:34};}
function widths(s,values,rows=140){values.forEach((w,i)=>s.getRangeByIndexes(0,i,rows,1).format.columnWidthPx=w);}
function title(s,value,note){s.getRange('A2').values=[[value]];s.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#17365D'};s.getRange('A2:V2').format.rowHeight=34;s.getRange('A3').values=[[note]];s.getRange('A3:V3').format.rowHeight=26;}
title(raw,'42个模型的正式Q结果','每行一个固定检查点；各任务主指标独立报告。0是真实测值，空白表示不适用。逐样本预测已重算核对。');
const fields=[['stage','实验组'],['dataset','数据集'],['model','模型'],['role','用途'],['train_rows','训练样本'],['seed','训练种子'],['lr','学习率'],['C','C值'],['budget','预算上限'],['metric','主指标'],['score','Q主指标值'],['macro_f1','Q Macro-F1'],['macro_ap','Q Macro-AP'],['tail_f1','Q尾类F1'],['tail_recall','Q尾类召回'],['head_f1','Q头类F1'],['test_rows','Q样本数'],['predict_seconds','批预测秒'],['load_seconds','模型加载秒'],['milliseconds_per_row','每条毫秒'],['run_id','唯一模型ID'],['source','来源结果文件']];
header(raw,'A5:V5',fields.map(x=>x[1]));
const roleNames={primary:'主比较',fixed_budget_reference:'历史预算参考',supplemental_Q:'S1/S3固定对照'};
const qIndex=new Map();data.q_records.forEach((r,i)=>{const row=i+6;qIndex.set(r.run_id,row);raw.getRange(`A${row}:V${row}`).values=[fields.map(([key])=>key==='model'?(names[r[key]]??r[key]):key==='dataset'?(ds[r[key]]??r[key]):key==='role'?(roleNames[r[key]]??r[key]):r[key]??null)];});
raw.tables.add('A5:V47',true,'SavedQResults').showFilterButton=true;
raw.getRange('A6:V47').format.rowHeight=27;raw.getRange('G6:G47').setNumberFormat('0.00000');raw.getRange('H6:H47').setNumberFormat('0.0');raw.getRange('K6:P47').setNumberFormat('0.0000');raw.getRange('O6:O47').setNumberFormat('0.0%');raw.getRange('R6:T47').setNumberFormat('0.000');
raw.getRange('I6:I47').format.borders={right:{style:'thin',color:'#DDE4EA'}};
widths(raw,[150,130,190,160,90,80,95,70,80,110,110,110,110,110,110,110,95,105,110,100,500,850]);raw.freezePanes.freezeRows(5);raw.freezePanes.freezeColumns(3);
title(vd,'V与D证据','本表保留证据用途：M2与长尾对照为V，错误轨迹和37对词汇干预为D。它们不并入42个Q模型。');
header(vd,'A5:D5',['M2四臂','配置数','V F1均值','配置SD']);
header(vd,'A13:E13',['M2分支','数据抽样种子','训练种子','V Macro-F1','唯一模型ID']);
const mIndex=new Map();data.m2.forEach((r,i)=>{const row=i+14;mIndex.set(r.run_id,row);vd.getRange(`A${row}:E${row}`).values=[[r.arm,r.data_seed,r.training_seed,r.macro_f1,r.run_id]];});
data.m2_summary.forEach((r,i)=>{const row=i+6,refs=data.m2.filter(v=>v.arm===r.arm).map(v=>`D${mIndex.get(v.run_id)}`);vd.getRange(`A${row}:D${row}`).values=[[r.arm,r.n,null,null]];vd.getRange(`C${row}`).formulas=[[`=AVERAGE(${refs.join(',')})`]];vd.getRange(`D${row}`).formulas=[[`=STDEV(${refs.join(',')})`]];});
header(vd,'A52:G52',['任务','模型','训练样本','均衡V F1','倾斜V F1','降幅百分点','来源模型ID']);
data.imbalance.forEach((r,i)=>{const row=i+53;vd.getRange(`A${row}:G${row}`).values=[[r.dataset,names[r.model],r.n,r.balanced,r.skewed,null,r.run_ids.join(' / ')]];vd.getRange(`F${row}`).formulas=[[`=(D${row}-E${row})*100`]];});
header(vd,'A64:G64',['D任务','模型','D样本数','持续错误','被纠正','新增错误','始终正确']);
data.trajectories.forEach((r,i)=>vd.getRange(`A${i+65}:G${i+65}`).values=[[r.dataset,names[r.model],r.n,r.persistent_error,r.corrected,r.new_error,r.always_correct]]);
header(vd,'A79:F79',['37对AI审核样本','样本数','原文准确率','线索替换准确率','控制准确率','线索相对控制降幅']);
data.s4.forEach((r,i)=>{const row=i+80;vd.getRange(`A${row}:F${row}`).values=[[names[r.model],r.n,r.original,r.cue,r.control,null]];vd.getRange(`F${row}`).formulas=[[`=(E${row}-D${row})*100`]];});
vd.getRange('A84').values=[['来源：analysis-v2/analysis-data.json逐源SHA；M2来自原训练状态，D来自S2-summary与S4冻结逐对结果。']];
widths(vd,[260,170,130,140,520,130,1050]);vd.getRange('A6:G84').format.rowHeight=26;
for(const r of ['C6:D9','D14:D49','D53:E61','C80:E81'])vd.getRange(r).setNumberFormat('0.0000');vd.getRange('F53:F61').setNumberFormat('0.00');vd.getRange('F80:F81').setNumberFormat('0.00');vd.freezePanes.freezeRows(5);

title(summary,'能力边界与失败机制：模型Q比较','42份Q预测已重算。神经模型3种子，SVM单次；不同任务主指标分别报告。');
header(summary,'A5:H5',['数据条件','模型','主指标','种子数','Q均值','种子SD','Q最低','Q最高']);
data.primary.forEach((r,i)=>{const row=i+6,refs=r.run_ids.map(id=>`'42模型Q明细'!K${qIndex.get(id)}`);summary.getRange(`A${row}:H${row}`).values=[[ds[r.dataset]+(r.dataset==='banking77'?' 1k':' 6k'),names[r.model],r.metric,r.n,null,null,null,null]];summary.getRange(`E${row}`).formulas=[[`=AVERAGE(${refs.join(',')})`]];if(r.n>1)summary.getRange(`F${row}`).formulas=[[`=STDEV(${refs.join(',')})`]];summary.getRange(`G${row}`).formulas=[[`=MIN(${refs.join(',')})`]];summary.getRange(`H${row}`).formulas=[[`=MAX(${refs.join(',')})`]];});
summary.getRange('A13').values=[['SD为训练种子波动，不是测试置信区间；Macro-F1与Macro-AP分别解释。']];
title(effects,'机制与配方：对照差值','条件前后固定匹配；差值由原始结果引用计算，单位为百分点。不同证据分支保留V/Q用途。');
header(effects,'A5:G5',['证据分支','固定条件与改变项','指标','条件前','条件后','差值百分点','解释范围']);
const cols={macro_f1:'L',macro_ap:'M',tail_f1:'N',tail_recall:'O'};
data.contrasts.forEach((r,i)=>{const row=i+6;effects.getRange(`A${row}:G${row}`).values=[[r.section,translate(r.context),r.metric,null,null,null,r.meaning]];
let a,b;if(r.section==='M2预训练与适配'){a=r.run_ids.slice(0,9).map(id=>`'V与D证据'!D${mIndex.get(id)}`);b=r.run_ids.slice(9).map(id=>`'V与D证据'!D${mIndex.get(id)}`);}else if(r.section==='Q模型比较'){a=[`'42模型Q明细'!K${qIndex.get(r.run_ids[0])}`];b=r.run_ids.slice(1).map(id=>`'42模型Q明细'!K${qIndex.get(id)}`);}else{a=[`'42模型Q明细'!${cols[r.metric]}${qIndex.get(r.run_ids[0])}`];b=[`'42模型Q明细'!${cols[r.metric]}${qIndex.get(r.run_ids[1])}`];}
effects.getRange(`D${row}`).formulas=[[`=AVERAGE(${a.join(',')})`]];effects.getRange(`E${row}`).formulas=[[`=AVERAGE(${b.join(',')})`]];effects.getRange(`F${row}`).formulas=[[`=(E${row}-D${row})*100`]];});
effects.tables.add(`A5:G${5+data.contrasts.length}`,true,'ResearchContrasts').showFilterButton=true;
summary.getRange('E6:H11').setNumberFormat('0.0000');effects.getRange('D6:E49').setNumberFormat('0.0000');effects.getRange('F6:F49').setNumberFormat('+0.00;-0.00;0.00');summary.getRange('A6:H11').format.rowHeight=30;effects.getRange('A6:G49').format.rowHeight=30;
widths(summary,[180,150,130,90,110,110,110,110]);widths(effects,[175,560,140,100,105,105,580]);effects.freezePanes.freezeRows(5);
effects.getRange('F6:F49').format.borders={right:{style:'thin',color:'#DDE4EA'}};
wb.recalculate();
for(let i=0;i<data.primary.length;i++){const r=data.primary[i],row=i+6;assert.ok(Math.abs(summary.getRange(`E${row}`).values[0][0]-r.mean)<1e-12);if(r.sd!==null)assert.ok(Math.abs(summary.getRange(`F${row}`).values[0][0]-r.sd)<1e-12);}
for(let i=0;i<data.contrasts.length;i++)assert.ok(Math.abs(effects.getRange(`F${i+6}`).values[0][0]-data.contrasts[i].difference_pp)<1e-10);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:15},maxChars:500});console.log(errors.ndjson);
for(const [sheet,range,name] of [['模型Q比较','A2:H13','overview'],['对照差值','A5:G12','contrasts'],['42模型Q明细','A5:M12','q-detail'],['V与D证据','A2:D9','vd-detail']]){const blob=await wb.render({sheetName:sheet,range,scale:1.2,format:'png'});await fs.writeFile(path.join(support,name+'.png'),new Uint8Array(await blob.arrayBuffer()));}
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'研究问题定量分析.xlsx'));
try{await fs.rename(path.join(out,'研究问题定量分析.xlsx.inspect.ndjson'),path.join(support,'研究问题定量分析.xlsx.inspect.ndjson'));}catch(error){if(error.code!=='ENOENT')throw error;}
console.log(JSON.stringify({Qrows:42,contrasts:data.contrasts.length,formulasVerified:true,saved:out}));

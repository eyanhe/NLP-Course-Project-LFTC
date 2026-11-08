import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const data=JSON.parse(await fs.readFile('research-v3/reports/overnight/convergence-v1/experiment-inventory-current.json','utf8'));
const out=path.resolve('outputs/experiment-inventory-20261108'); await fs.mkdir(out,{recursive:true});
const support=path.join(out,'support');await fs.mkdir(support,{recursive:true});
const wb=Workbook.create(),main=wb.worksheets.add('实验总览'),raw=wb.worksheets.add('逐模型记录');
const names={tfidf_lr:'TF-IDF + LR',tfidf_svm:'TF-IDF + SVM',w2v_svm:'Word2Vec + SVM',bilstm:'BiLSTM',distilbert:'DistilBERT',modernbert:'ModernBERT',m2_distilbert:'DistilBERT机制对照',fixed_tfidf_svm:'固定TF-IDF + SVM',frozen_generic_distilbert_head:'冻结DistilBERT + 分类头'};
const states={complete:'完成',interrupted:'中断',running:'运行中'};
const rows=data.rows.sort((a,b)=>a.category.localeCompare(b.category)||String(a.dataset).localeCompare(String(b.dataset))||String(a.model).localeCompare(String(b.model))||a.run_id.localeCompare(b.run_id));
const columns=[['category','训练阶段'],['dataset','数据集'],['model','模型'],['status','训练状态'],['train_rows','训练样本'],['epochs','完成轮数'],['seed','随机种子'],['lr','学习率'],['C','C值'],['arm','机制分支'],['experiments','原实验编号'],['validation','有V指标'],['validation_rows','V样本数'],['V_macro_f1','V Macro-F1'],['V_macro_ap','V Macro-AP'],['D_S2_prediction','S2探索D预测'],['formal_Q','正式Q推理'],['Q_rows','Q样本数'],['Q_role','Q用途'],['notes','说明'],['run_id','唯一训练ID'],['status_source','训练证据路径'],['Q_prediction_source','正式预测路径'],['status_sha256','状态文件SHA256']];
const end=rows.length+3;
for(const sheet of [main,raw]){sheet.showGridLines=false;sheet.getRange('A1:X460').format.font={name:'Arial',size:10,color:'#263746'};sheet.getRange('A1:X460').format.verticalAlignment='center';}
raw.getRange('A1').values=[['每行一个run_id。完成与中断分开；验证指标来自实际保存的模型状态，正式推理以已下载预测文件为准。']];
raw.getRange('A2').values=[['D只列已核对的S2探索预测；“未列入”不表示从未执行其他探索推理。E0为工程用途。训练ID与证据路径用于追溯。']];
raw.getRange(`A3:X${end}`).values=[columns.map(x=>x[1]),...rows.map(r=>columns.map(([k])=>k==='model'?(names[r[k]]??r[k]):k==='status'?(states[r[k]]??r[k]):['validation','formal_Q'].includes(k)?(r[k]?'是':'否'):k==='D_S2_prediction'?(r[k]?'完成':'未列入'):r[k]??null))];
raw.tables.add(`A3:X${end}`,true,'ExperimentRecords').showFilterButton=true;
raw.getRange(`A4:X${end}`).format.rowHeight=28;
const rawWidths=[180,145,180,70,90,85,80,105,70,200,105,80,85,105,105,110,90,90,150,400,480,660,660,500];
rawWidths.forEach((w,i)=>raw.getRangeByIndexes(0,i,end,1).format.columnWidthPx=w);
raw.getRange(`H4:H${end}`).setNumberFormat('0.00000');raw.getRange(`I4:I${end}`).setNumberFormat('0.0');raw.getRange(`N4:O${end}`).setNumberFormat('0.0000');
raw.freezePanes.freezeRows(3);raw.freezePanes.freezeColumns(3);
main.getRange('A2').values=[['训练、验证与正式推理：累计实验清单']];main.getRange('A2:F2').merge();main.getRange('A2:F2').format={font:{name:'Arial',size:17,bold:true,color:'#17365D'},rowHeight:38};
main.getRange('A3:F3').merge();main.getRange('A3').values=[[`核对时间 ${new Date(Date.parse(data.checked_at_utc)+8*3600000).toISOString().slice(0,16).replace('T',' ')} SGT；累计42个Q模型，S1/S3最后于06:58完成。`]];
main.getRange('A4:F4').merge();main.getRange('A4').values=[['累计完成428条，当前研究232条；另181条早期选参及15条工程检查。2条中断不计完成。复用及续跑按同一ID计一次。']];
main.getRange('A6:F6').values=[['训练阶段','完成模型记录','有V指标','S2探索D预测','正式Q推理','范围说明']];
const descriptions=['7数据集、7模型/机制变体；原E1—E5及M2、M5的已完成部分','BANKING77 1k和GoEmotions 6k；两神经模型，学习率×预算2×2','BANKING77固定表示×独立标注支持×类别贡献2×2','BANKING77 1k和GoEmotions 6k；20次神经训练及4次新SVM拟合','早期共217候选，其中36条已纳入原主研究，本行只计额外181条','小样本真实训练与重载检查，用于工程验证，不纳入研究结论'];
data.totals.forEach((r,i)=>{const rr=7+i;main.getRange(`A${rr}:F${rr}`).values=[[r.category,null,null,null,null,descriptions[i]]];main.getRange(`B${rr}`).formulas=[[`=COUNTIFS('逐模型记录'!A4:A${end},A${rr},'逐模型记录'!D4:D${end},"完成")`]];for(const [dest,col,term] of [['C','L','是'],['D','P','完成'],['E','Q','是']])main.getRange(`${dest}${rr}`).formulas=[[`=COUNTIFS('逐模型记录'!A4:A${end},A${rr},'逐模型记录'!D4:D${end},"完成",'逐模型记录'!${col}4:${col}${end},"${term}")`]];});
main.getRange('A13').values=[['当前研究小计']];main.getRange('A14').values=[['全部完成记录']];for(const col of ['B','C','D','E']){main.getRange(`${col}13`).formulas=[[`=SUM(${col}7:${col}10)`]];main.getRange(`${col}14`).formulas=[[`=SUM(${col}7:${col}12)`]];}
main.getRange('A16:H16').merge();main.getRange('A16').values=[['最近这轮26条：24条新训练 + 2条旧SVM复用。每个神经模型含3档学习率seed42及最优配方seed43/44。']];
main.getRange('A18:H18').values=[['数据集','模型','训练样本','记录数','V验证','正式Q','最优学习率/C','Q范围']];
data.latest_round.forEach((r,i)=>{const rr=19+i;main.getRange(`A${rr}:H${rr}`).values=[[r.dataset,names[r.model],r.train_rows,r.records,r.V,r.Q,r.best_lr_or_C,r.model==='tfidf_svm'?'最优C一次':'最优配方3个种子']];main.getRange(`G${rr}`).setNumberFormat(r.model==='tfidf_svm'?'0.0':'0.00000');});
const notes=[['A26','正式Q共42个模型：上表14个主比较、4个历史参考、S1的16个与S3的8个。未选学习率/C候选只在V上选参。'],['A27','V验证已经包含真实推理。D用于探索错误轨迹；Q是独立正式测试，三者分别记录。'],['A28','v3首批固定预算6条strict_no_val分支按设计未使用V；不是漏做。184条中其余178条有V指标。'],['A29','S2：四任务24个原模型的D预测、12组1k→6k轨迹已完成。S4：37对Gemini审核编辑，两个模型三臂推理已完成，无新训练。'],['A30','Q尚未覆盖全模型/全数据条件；S1、S3已有正式Q。原2859项主清单完成184项，其余2675项未完成，补充实验另计。']];
for(const [cell,value] of notes){const row=Number(cell.slice(1));main.getRange(`A${row}:H${row}`).merge();main.getRange(cell).values=[[value]];main.getRange(`A${row}:H${row}`).format={wrapText:true,rowHeight:34,font:{name:'Arial',size:10,color:'#52657A'}};}
const widths=[200,180,105,100,90,610,135,180];widths.forEach((w,i)=>main.getRangeByIndexes(0,i,31,1).format.columnWidthPx=w);
main.getRange('A7:F14').format.rowHeight=42;main.getRange('F7:F12').format.wrapText=true;
main.getRange('E7:E14').format.borders={right:{style:'thin',color:'#DDE4EA'}};
main.getRange('A19:H24').format.rowHeight=30;
for(const [s,range] of [[main,'A6:F6'],[main,'A18:H18'],[raw,'A3:X3']])s.getRange(range).format={fill:'#17365D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:34,horizontalAlignment:'center'};
for(const rr of [13,14])main.getRange(`A${rr}:F${rr}`).format={fill:rr===13?'#DCEAF7':'#E9EDF2',font:{name:'Arial',size:10,bold:true,color:'#17365D'}};
main.freezePanes.freezeRows(6);
wb.recalculate();assert.equal(main.getRange('B13').values[0][0],232);assert.equal(main.getRange('C13').values[0][0],226);assert.equal(main.getRange('E14').values[0][0],42);assert.equal(main.getRange('B14').values[0][0],428);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},maxChars:500});
console.log(errors.ndjson);
const blob=await wb.render({sheetName:'实验总览',range:'A2:F14',scale:1.25,format:'png'});await fs.writeFile(path.join(support,'overview.png'),new Uint8Array(await blob.arrayBuffer()));
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'累计训练验证推理清单.xlsx'));
try{await fs.rename(path.join(out,'累计训练验证推理清单.xlsx.inspect.ndjson'),path.join(support,'累计训练验证推理清单.xlsx.inspect.ndjson'));}catch(error){if(error.code!=='ENOENT')throw error;}
console.log(JSON.stringify({rows:rows.length,complete:428,research:232,Q:42,saved:out}));

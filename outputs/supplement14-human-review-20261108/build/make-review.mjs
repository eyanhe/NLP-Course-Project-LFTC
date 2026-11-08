import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const project=process.cwd();
const outputDir=path.resolve('outputs/supplement14-human-review-20261108');
const sourceDir=path.resolve('research-v3/reports/overnight/supplement-1-4/human-review');
const criteria=[
  '原标签与原文相符，且意图可明确判断',
  'A 保持原文含义、标签和关键槽位',
  'B 保持原文含义、标签和关键槽位',
  'A 自然通顺，未新增歧义',
  'B 自然通顺，未新增歧义',
  'A 替换的是内容词',
  'B 替换的是内容词',
];
const headers=['案例编号','数据集原标签','原文','文本 A','文本 B','原标签正确','A 含义保持','B 含义保持','A 自然通顺','B 自然通顺','A 改内容词','B 改内容词','备注／修改建议','本表行状态','样本 ID'];
const navy='#26374D',ink='#243247',amber='#FFF2CC',muted='#66758A';
const audit=[];
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');

function parseMarkdown(md){
  const chunks=md.split(/^## /m).slice(1);
  return chunks.map(chunk=>{
    const identity=chunk.match(/^(M1-\d+) · ([^\r\n]+)/);
    assert(identity,'Case identity missing');
    const field=label=>{
      const line=chunk.split(/\r?\n/).find(x=>x.startsWith(label+'：'));
      assert(line,`Missing ${label}`);return line.slice(label.length+1);
    };
    const judgements=criteria.map(criterion=>{
      const line=chunk.split(/\r?\n/).find(x=>x.startsWith('| '+criterion+' |'));
      assert(line,`Missing criterion ${criterion}`);
      const value=line.split('|')[2].trim();
      assert(['待填','是','否','不确定'].includes(value),`Unexpected human entry: ${value}`);
      return value==='待填'?null:value;
    });
    const notes=field('备注/拒绝原因/修改建议');
    return {caseId:identity[1],sampleId:identity[2],label:field('原标签'),original:field('原文'),
      A:field('文本 A'),B:field('文本 B'),judgements,notes:notes==='待填'?null:notes};
  });
}

await fs.mkdir(outputDir,{recursive:true});
for(const reviewer of ['A','B']){
  const raw=await fs.readFile(path.join(sourceDir,`人审${reviewer}.md`));
  const cases=parseMarkdown(raw.toString('utf8'));
  assert.equal(cases.length,56);assert.equal(new Set(cases.map(c=>c.sampleId)).size,56);
  const wb=Workbook.create();const sheet=wb.worksheets.add('逐条人审');
  sheet.showGridLines=false;sheet.tabColor=navy;
  const start=10,end=start+cases.length-1;
  const grid=sheet.getRange(`A1:O${end}`);
  grid.format.font={name:'Arial',size:11,color:ink};grid.format.verticalAlignment='center';
  sheet.getRange('A1:O1').format.rowHeight=8;
  sheet.getRange('A2').values=[[`M1 独立人审表 ${reviewer}`]];
  sheet.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:navy};
  sheet.getRange('A2:O2').format.rowHeight=30;
  sheet.getRange('A2:O2').format.borders={bottom:{style:'thin',color:'#BAC6D5'}};
  sheet.getRange('A3').values=[['复核者']];sheet.getRange('B3').values=[[null]];
  sheet.getRange('C3').values=[['复核日期']];sheet.getRange('D3').values=[[null]];sheet.getRange('D3').setNumberFormat('yyyy-mm-dd');
  sheet.getRange('F3').values=[['看过预测？']];sheet.getRange('G3').values=[[null]];
  sheet.getRange('G3').dataValidation={rule:{type:'list',values:['未见过','见过','不确定']}};
  sheet.getRange('I3').values=[['复核信息']];
  sheet.getRange('J3').formulas=[['=IF(B3="","待填姓名",IF(D3="","待填日期",IF(G3="","待确认盲法",IF(G3="未见过","已填","盲法未满足"))))']];
  for(const cell of ['B3','D3','G3'])sheet.getRange(cell).format.fill=amber;
  sheet.getRange('A3:O3').format.rowHeight=28;
  sheet.getRange('A4').values=[['草稿对数']];sheet.getRange('B4').values=[[cases.length]];
  sheet.getRange('C4').values=[['判断填齐']];
  sheet.getRange('D4').formulas=[[`=COUNTIFS(N${start}:N${end},"本对同意")+COUNTIFS(N${start}:N${end},"不同意")+COUNTIFS(N${start}:N${end},"需讨论")`]];
  sheet.getRange('F4').values=[['本表同意']];sheet.getRange('G4').formulas=[[`=COUNTIFS(N${start}:N${end},"本对同意")`]];
  sheet.getRange('I4').values=[['尚待填齐']];sheet.getRange('J4').formulas=[['=B4-D4']];
  sheet.getRange('A4:O4').format.rowHeight=24;
  for(const cell of ['B4','D4','G4','J4'])sheet.getRange(cell).format.horizontalAlignment='center';
  sheet.getRange('A5').values=[['只填浅黄色：每项选择“是／否／不确定”，备注可写拒绝原因或修改建议。']];
  sheet.getRange('A6').values=[['独立填写，不看另一份表或模型预测。不要改原文及A/B草稿；本表同意不等于双人审核通过。']];
  sheet.getRange('A7').values=[['来源：预注册 BANKING77 D 试点的固定候选顺序。A/B顺序沿用原表，未揭示干预臂。']];
  sheet.getRange('A5:O7').format.rowHeight=22;
  sheet.getRange('A5:O7').format.font={name:'Arial',size:11,color:muted};
  sheet.getRange('A5:O7').format.wrapText=false;
  sheet.getRange('A8:O8').format.rowHeight=6;

  const rows=cases.map(c=>[c.caseId,c.label,c.original,c.A,c.B,...c.judgements,c.notes,null,c.sampleId]);
  sheet.getRange(`A9:O${end}`).values=[headers,...rows];
  const table=sheet.tables.add(`A9:O${end}`,true,'M1Review');
  table.showFilterButton=true;
  const widths=[84,194,300,300,300,92,92,92,92,92,92,92,240,110,215];
  widths.forEach((width,col)=>sheet.getRangeByIndexes(0,col,end,1).format.columnWidthPx=width);
  sheet.getRange(`A10:O${end}`).format.wrapText=true;
  sheet.getRange(`B10:E${end}`).format.verticalAlignment='top';
  sheet.getRange(`M10:M${end}`).format.verticalAlignment='top';
  for(let i=0;i<cases.length;i++){
    const row=start+i,c=cases[i];
    const maxLines=Math.max(...[c.label,c.original,c.A,c.B].map((text,j)=>Math.ceil(text.length/(j===0?24:43))));
    sheet.getRange(`A${row}:O${row}`).format.rowHeight=Math.max(62,maxLines*15+15);
    sheet.getRange(`A${row}:O${row}`).format.fill=i%2?'#F1F4F8':'#FFFFFF';
    sheet.getRange(`N${row}`).formulas=[[`=IF(COUNTA(F${row}:L${row})<7,"待填写",IF(COUNTIFS(F${row}:L${row},"否")>0,"不同意",IF(COUNTIFS(F${row}:L${row},"不确定")>0,"需讨论","本对同意")))`]];
  }
  sheet.getRange(`F10:M${end}`).format.fill=amber;
  sheet.getRange(`F10:L${end}`).dataValidation={rule:{type:'list',values:['是','否','不确定']}};
  sheet.getRange(`F10:L${end}`).format.horizontalAlignment='center';
  sheet.getRange(`N10:N${end}`).format.horizontalAlignment='center';
  sheet.getRange(`O10:O${end}`).format.font={name:'Arial',size:11,color:muted};
  sheet.getRange('A9:O9').format={fill:navy,font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},
    horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:38};
  sheet.getRange('A9:O9').format.borders={insideVertical:{style:'thin',color:'#FFFFFF'}};
  sheet.freezePanes.freezeRows(9);sheet.freezePanes.freezeColumns(2);
  for(const [text,color] of [['不同意','#FCE4E4'],['需讨论','#FFE4B5']]){
    sheet.getRange(`N10:N${end}`).conditionalFormats.add('containsText',{text,format:{fill:color,font:{color:'#8A3333'}}});
  }
  // Verify that partial entries cannot count as complete and metadata still shows missing prerequisites.
  sheet.getRange('F10').values=[['是']];wb.recalculate();
  assert.equal(sheet.getRange('N10').values[0][0],'待填写');
  assert.equal(sheet.getRange('D4').values[0][0],0);
  sheet.getRange('F10:L10').values=[Array(7).fill('是')];wb.recalculate();
  assert.equal(sheet.getRange('N10').values[0][0],'本对同意');
  assert.equal(sheet.getRange('D4').values[0][0],1);assert.equal(sheet.getRange('G4').values[0][0],1);
  assert.equal(sheet.getRange('J3').values[0][0],'待填姓名');
  sheet.getRange('G10').values=[['不确定']];wb.recalculate();assert.equal(sheet.getRange('N10').values[0][0],'需讨论');
  sheet.getRange('G10').values=[['否']];wb.recalculate();assert.equal(sheet.getRange('N10').values[0][0],'不同意');
  sheet.getRange('F10:L10').values=[cases[0].judgements];
  wb.recalculate();
  // Preserve all original source text, ordering, notes and any actual existing decisions.
  for(let i=0;i<cases.length;i++){
    const values=sheet.getRange(`A${start+i}:M${start+i}`).values[0];
    assert.deepEqual(values,rows[i].slice(0,13));
  }
  const inspect=await wb.inspect({kind:'table',range:'逐条人审!A3:J4',include:'values,formulas',tableMaxRows:2,tableMaxCols:10,maxChars:1400});
  console.log(JSON.stringify({reviewer,summary:inspect.ndjson}));
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000});
  console.log(JSON.stringify({reviewer,errors:errors.ndjson}));
  const blob=await wb.render({sheetName:'逐条人审',range:'A2:N12',scale:1,format:'png'});
  await fs.writeFile(path.join(outputDir,`build/preview-${reviewer}.png`),new Uint8Array(await blob.arrayBuffer()));
  const file=path.join(outputDir,`M1独立人审-${reviewer}.xlsx`);
  await (await SpreadsheetFile.exportXlsx(wb)).save(file);
  const result=await fs.readFile(file);
  audit.push({reviewer,file:path.relative(project,file),cases:cases.length,source_sha256:sha(raw),xlsx_sha256:sha(result),bytes:result.length,
    human_judgements_copied:cases.flatMap(c=>c.judgements).filter(v=>v!==null).length,source_order_and_text_checked:true,workflow_input_changes_checked:true});
  console.log(JSON.stringify(audit.at(-1)));
}
await fs.writeFile(path.join(outputDir,'build/verification.json'),JSON.stringify(audit,null,2));

import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const dir=path.resolve('outputs/supplement14-human-review-20261108');
const reports=[];
for(const reviewer of ['A','B']){
  const file=path.join(dir,`M1独立人审-${reviewer}.xlsx`);
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const sheet=wb.worksheets.getItem('逐条人审');
  const metadata=sheet.getRange('A3:J4').values;
  const rows=sheet.getRange('A10:O65').values;
  assert.equal(rows.length,56);
  console.log(JSON.stringify({reviewer,metadata}));
  reports.push({reviewer,file,metadata,rows});
}
const source=path.resolve('research-v3/reports/overnight/supplement-1-4/human-review');
const pairs=(await fs.readFile(path.join(source,'AI-draft-pairs.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);
const mapping=(await fs.readFile(path.join(source,'arm-key-do-not-open-before-review.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);
const pairById=new Map(pairs.map(p=>[p.sample_id,p]));const mapByCase=new Map(mapping.map(m=>[m.case_id,m]));
const verdicts={};
for(const report of reports){
  const seen=new Set();const issues=[];const counts={complete:0,all_yes:0,has_no:0,uncertain_without_no:0,incomplete:0};const records=[];
  for(const row of report.rows){
    const [caseId,label,original,A,B,...rest]=row;const decisions=row.slice(5,12);const note=row[12];
    const m=mapByCase.get(caseId);assert(m,`Unknown case ${caseId}`);assert(!seen.has(caseId));seen.add(caseId);
    const p=pairById.get(m.sample_id);const changed=[];
    if(row[14]!==m.sample_id)changed.push('sample_id');
    if(original!==p.original_text)changed.push('original');
    if(A!==p[m.A_role+'_text'])changed.push('A_draft');
    if(B!==p[m.B_role+'_text'])changed.push('B_draft');
    if(changed.length)issues.push({caseId,changed});
    const complete=decisions.every(v=>['是','否','不确定'].includes(v));
    const kind=!complete?'incomplete':decisions.includes('否')?'has_no':decisions.includes('不确定')?'uncertain_without_no':'all_yes';
    counts[kind]++;if(complete)counts.complete++;
    records.push({caseId,sampleId:m.sample_id,decisions,note,kind,source_changes:changed});
  }
  verdicts[report.reviewer]=new Map(records.map(r=>[r.caseId,r]));
  report.counts=counts;report.issues=issues;report.records=records;
  console.log(JSON.stringify({reviewer:report.reviewer,counts,issues,not_unanimous_yes:records.filter(r=>r.kind!=='all_yes')}));
}
const joint={both_all_yes:[],both_not_all_yes:[],only_A_all_yes:[],only_B_all_yes:[],cell_disagreements:[]};
for(const [caseId,a] of verdicts.A){
  const b=verdicts.B.get(caseId);assert(b);
  const ay=a.kind==='all_yes',by=b.kind==='all_yes';
  joint[ay&&by?'both_all_yes':ay?'only_A_all_yes':by?'only_B_all_yes':'both_not_all_yes'].push(caseId);
  for(let i=0;i<7;i++)if(a.decisions[i]!==b.decisions[i])joint.cell_disagreements.push({caseId,criterion_index:i,A:a.decisions[i],B:b.decisions[i],A_note:a.note,B_note:b.note});
}
console.log(JSON.stringify({joint}));
await fs.writeFile(path.join(dir,'build/completed-review-readonly.json'),JSON.stringify({reports,joint},null,2));

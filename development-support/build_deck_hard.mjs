import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');process.env.RUNTIME_NODE=path.join(runtime,'node/bin/node.exe');process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
const skill='C:/Users/LUZIJ/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.12130/skills/presentations';
const workspace='C:/Users/LUZIJ/Documents/Codex/2026-11-06/ni';const root=path.join(workspace,'outputs/sentiment-project');const tmp=path.join(workspace,'work/deck-hard');
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const d=JSON.parse(await fs.readFile(path.join(root,'report/slide-data-hard.json'),'utf8'));
const old=JSON.parse(await fs.readFile(path.join(root,'report/slide-data-v2.json'),'utf8'));
const models=['logistic_regression','linear_svm','word2vec_svm','bilstm','distilbert'];
const short={logistic_regression:'Pair TF-IDF + LR',linear_svm:'Pair TF-IDF + SVM',word2vec_svm:'Pair Word2Vec + SVM',bilstm:'Shared BiLSTM',distilbert:'DistilBERT'};
const p=Presentation.create({slideSize:{width:1280,height:720}});const bg='#f7f6f0',ink='#173c35',body='#253a34',muted='#5a6962',font='Arial';
function text(s,t,x,y,w,h,size=28,bold=false,color=body){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={fontFamily:font,fontSize:size,bold,color};return a;}
function slide(title,notes){const s=p.slides.add();s.background.fill=bg;text(s,title,64,44,1150,90,42,true,ink);text(s,String(p.slides.items.length).padStart(2,'0'),1170,675,60,25,16,false,muted);s.speakerNotes.text=notes;return s;}
function footer(s,t){text(s,t,76,610,1110,55,20,false,muted);}
function lines(s,items,start=170,step=102,size=29){items.forEach((t,i)=>text(s,t,76,start+i*step,1110,step-12,size));}
function table(s,values,y=175,h=370,widths=[420,240,240,240]){const t=s.tables.add({rows:values.length,columns:values[0].length,left:70,top:y,width:1140,height:h,values,columnWidths:widths});t.borders.assign({fill:'#d8ddd7',width:1});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const a=t.getCell(r,c);a.fill=r===0?ink:bg;a.text.style={fontFamily:font,fontSize:23,bold:r===0,color:r===0?'#ffffff':body};}return t;}
const own='Source: results-hard/evaluation.json, results-hard/metrics.csv, report/slide-data-hard.json. Deep values average seeds 42/43/44, traditional models use fixed fits.';
const row=(split,m)=>d.summary.find(r=>r.target==='mnli_'+split&&r.model===m);
{
const s=slide('MultiNLI sentence-pair inference','Williams, Nangia and Bowman (2018), A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. https://aclanthology.org/N18-1101/ . Dataset https://huggingface.co/datasets/nyu-mll/multi_nli . This supplement adds a semantic task to the V2 topic/sentiment project.');
text(s,'Premise',76,175,1080,50,30,true,ink);text(s,'A woman is reading a book in a library.',76,235,1080,60,31);
text(s,'Hypothesis',76,345,1080,50,30,true,ink);text(s,'Someone is reading indoors.',76,405,1080,60,31);
footer(s,'Entailment / Neutral / Contradiction. Illustrative pair, not a scored benchmark sample.');
}
{
const s=slide('Why this is a harder comparison','Source: original MultiNLI paper https://aclanthology.org/N18-1101/ and the observed frozen results. Difficulty is semantic rather than just a larger class count. Do not claim that model gains prove formal logical reasoning.');
lines(s,['The label describes a relation between two sentences.',
'Lexical overlap alone does not determine entailment or contradiction.',
'Matched and mismatched genres expose different generalization conditions.',
'A hypothesis-only baseline tests predictive cues without the premise.']);
}
{
const s=slide('Fixed-budget protocol and premise grouping','Source: config-hard.json, data-hard/audit.json, results-hard/protocol.freeze.json. Official validation_matched/mismatched are reserved evaluation pools, not official hidden test leaderboard data. All normalized premise groups are disjoint across the four partitions. Final checkpoints frozen before final evaluation.');
table(s,[['Partition','Pairs','Source'],['Training','6,000','Official training pool'],['Model selection','999','Official training pool'],['Matched evaluation','999','Official validation_matched'],['Mismatched evaluation','999','Official validation_mismatched']],170,350,[380,180,580]);
footer(s,'333 examples per evaluation class. Shared normalized premises never cross partitions.');
}
{
const s=slide('Pair-aware traditional and deep models','Sources: pair_models.py, hard_benchmark.py. Traditional pair features concatenate premise, hypothesis, absolute difference and product. BiLSTM shares an encoder and uses the same interaction form. DistilBERT jointly tokenizes the two sentences. Local word vectors and generic external pretraining are unequal resources.');
table(s,[['Family','Sentence representation','Interaction'],['TF-IDF + LR / SVM','Training-only sparse vectors','P, H, absolute difference, product'],['Word2Vec + SVM','Mean of 100D training vectors','Same paired feature structure'],['Shared BiLSTM','Trainable word-vector sequences','Two states plus interactions'],['DistilBERT','Joint contextual pair encoding','New three-class head']],170,350,[360,380,400]);
footer(s,'BiLSTM: up to 256 words per sentence. DistilBERT: 256 subwords for the combined pair.');
}
{
const s=slide('Matched and mismatched results',own+' Scores are a sampled fixed-budget course experiment, not a full-data leaderboard replication.');
table(s,[['Model','Matched F1','Mismatched F1','Matched Accuracy'],...models.map(m=>[short[m],row('matched',m).macro_f1.toFixed(4),row('mismatched',m).macro_f1.toFixed(4),row('matched',m).accuracy.toFixed(4)])],170,375,[420,240,240,240]);
footer(s,'Macro F1 and Accuracy are distinct. Full Precision, Recall, SD and per-class results accompany the report.');
}
{
const s=slide('Performance across tasks',own+' AG News values from report/slide-data-v2.json. Corpora differ in input structure, genre, class count and annotation; the chart does not isolate a single causal effect of task difficulty.');
const c=s.charts.add('bar',{position:{left:70,top:145,width:1140,height:430},title:'Mean Macro F1 under the course training budgets',titleTextStyle:{fontSize:24,fill:ink,bold:true},categories:['AG News','MultiNLI matched','MultiNLI mismatched'],series:models.map((m,i)=>({name:short[m].replace("Pair ","").replace("Shared ",""),values:[old.summary.find(r=>r.source==='ag_news'&&r.model===m).macro_f1,row('matched',m).macro_f1,row('mismatched',m).macro_f1].map(v=>Number(v.toFixed(6))),fill:['#bdc5c0','#899a91','#596f66','#4e7788','#1b766e'][i]})),hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18,fill:body}},barOptions:{direction:'column',grouping:'clustered'},xAxis:{textStyle:{fontSize:20,fill:body}},yAxis:{min:0,max:1,title:{text:'Macro F1 (0–1)',textStyle:{fontSize:20,fill:body}},textStyle:{fontSize:20,fill:body},majorGridlines:{fill:'#d8ddd7',width:1}},chartFill:'none',plotAreaFill:'none'});applyPresentationChartFont(c,{fontFamily:font});
footer(s,'The empirical performance range is lower on MultiNLI. Task differences confound causal claims.');
}
{
const s=slide('Uncertainty and premise-free cues','Source: results-hard/bootstrap.json and hypothesis-only diagnostic in evaluation.json. Preselected SVM versus DistilBERT seed42. Bootstrap resamples premise groups 1,000 times, not individual correlated hypotheses. Conditional on two trained models, without multiple-comparison correction. Annotation-artifact motivation: https://aclanthology.org/N18-2017/ .');
table(s,[['Condition','Seed42 Delta F1','95% cluster interval'],...['matched','mismatched'].map(split=>{const b=d.bootstrap[split];return [split,b.delta_macro_f1.toFixed(4),`[${b.ci95_low.toFixed(4)}, ${b.ci95_high.toFixed(4)}]`];})],165,200,[350,340,450]);
text(s,'Hypothesis-only diagnostic',76,410,1100,50,31,true,ink);
text(s,d.diagnostics.map(r=>`${r[0]}: Accuracy ${r[1]}, Macro F1 ${r[2]}`).join('\n'),76,472,1110,100,28);
footer(s,'Premise-free predictive cues limit claims that an NLI score measures only sentence reasoning.');
}
{
const s=slide('Confusion patterns on matched pairs',own+' Confusion matrices use fixed SVM and DistilBERT seed42, each true class support 333. Rows=true, columns=predicted. E=Entailment, N=Neutral, C=Contradiction.');
for(let i=0;i<2;i++){
 const name=['linear_svm','distilbert'][i],x=70+590*i;text(s,short[name],x,165,540,55,30,true,ink);
 const matrix=d.confusions.matched[name],values=[['Class','E','N','C'],...matrix.map((v,j)=>[['E','N','C'][j],...v.map(String)])];
 const t=s.tables.add({rows:4,columns:4,left:x,top:250,width:550,height:280,columnWidths:[160,130,130,130],values});t.borders.assign({fill:'#d8ddd7',width:1});
 for(let r=0;r<4;r++)for(let c=0;c<4;c++){const a=t.getCell(r,c);a.fill=r===0?ink:bg;a.text.style={fontFamily:font,fontSize:25,color:r===0?'#ffffff':body,bold:r===0};}
}
footer(s,'E=Entailment, N=Neutral, C=Contradiction. Per-class Precision/Recall/F1 are in the GUI and report.');
}
{
const s=slide('Interpretation and English demonstration',own+' GUI adds separate premise/hypothesis inputs and matched/mismatched selector. Do not tune to final test gaps. Actual narration must use real members. This supplement can replace part of the V2 presentation while keeping the full course video under 10 minutes.');
lines(s,['A harder benchmark can expose a task-dependent representation gap.',
'A larger score gap is evidence for this budget, not universal superiority.',
'Cross-genre performance need not decrease on every sampled condition.',
'English GUI: both sentences, all five models, metrics and error matrices.']);
}
{
const s=slide('Sources and scope','https://aclanthology.org/N18-1101/\nhttps://huggingface.co/datasets/nyu-mll/multi_nli\nhttps://aclanthology.org/N18-2017/\nTask provenance: assets.lock.json, data-hard/audit.json and config-hard.json. Local result artifacts remain separate from official leaderboard results.');
lines(s,['Williams, Nangia and Bowman (2018): MultiNLI',
'Gururangan et al. (2018): annotation artifacts in NLI',
'Source data and versions: NYU MultiNLI dataset card and local asset lock',
'Scope: a predeclared 6,000-pair course experiment, not a leaderboard claim'],170,105,28);
}
await fs.mkdir(tmp,{recursive:true});const candidate=path.join(tmp,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(root,'report/hard-benchmark-supplement-r2.pptx');
const result=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[3,4,5,7,8].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[3,4,5,7,8],requiredNativeChartOwnerSlides:[6],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation-r2.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));for(let i=0;i<final.slides.items.length;i++){const img=await final.export({slide:final.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));}
console.log(JSON.stringify({finalPath,slides:final.slides.items.length,validated:result.finalSha256}));

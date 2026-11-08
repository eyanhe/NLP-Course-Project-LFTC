import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');process.env.RUNTIME_NODE=path.join(runtime,'node/bin/node.exe');process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
const skill='C:/Users/LUZIJ/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.12130/skills/presentations';
const workspace='C:/Users/LUZIJ/Documents/Codex/2026-11-06/ni',root=path.join(workspace,'outputs/sentiment-project'),tmp=path.join(workspace,'work/deck-final');
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const d=JSON.parse(await fs.readFile(path.join(root,'report/slide-data-final.json'),'utf8'));
const v2=JSON.parse(await fs.readFile(path.join(root,'report/slide-data-v2.json'),'utf8'));
const cost=JSON.parse(await fs.readFile(path.join(root,'results-v2/benchmark.json'),'utf8')).rows.filter(r=>r.source==='imdb');
const models=['logistic_regression','linear_svm','word2vec_svm','bilstm','distilbert'];
const short={logistic_regression:'TF-IDF + LR',linear_svm:'TF-IDF + SVM',word2vec_svm:'Word2Vec + SVM',bilstm:'Word2Vec + BiLSTM',distilbert:'DistilBERT'};
const p=Presentation.create({slideSize:{width:1280,height:720}}),bg='#f7f6f0',ink='#173c35',body='#253a34',muted='#5a6962',font='Arial';
function text(s,t,x,y,w,h,size=28,bold=false,color=body){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={fontFamily:font,fontSize:size,bold,color};return a;}
function slide(title,notes){const s=p.slides.add();s.background.fill=bg;text(s,title,64,44,1150,90,42,true,ink);text(s,String(p.slides.items.length).padStart(2,'0'),1170,675,60,25,16,false,muted);s.speakerNotes.text=notes;return s;}
function footer(s,t){text(s,t,76,610,1110,55,20,false,muted);}
function lines(s,items,start=170,step=102,size=29){items.forEach((t,i)=>text(s,t,76,start+i*step,1110,step-12,size));}
function table(s,values,y=175,h=370,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:70,top:y,width:1140,height:h,values,columnWidths:widths});t.borders.assign({fill:'#d8ddd7',width:1});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const a=t.getCell(r,c);a.fill=r===0?ink:bg;a.text.style={fontFamily:font,fontSize:23,bold:r===0,color:r===0?'#ffffff':body};}return t;}
const note='Local experiment sources: report/final-report.md, report/imbalance-report.md, results-v2, results-hard and results-imbalance. Deep results mean seeds42/43/44, traditional methods fixed fits. All scores are empirical course-budget results.';
{
const s=slide('Traditional NLP and deep learning',note+' Project compares conventional classification, MultiNLI and fixed-total training imbalance. Human contributions and voice recording are outside this artifact.');
text(s,'Text classification under changing task demands',76,205,1100,100,38,true,ink);
text(s,'Conventional datasets\nSentence-pair inference\nClass-imbalance robustness',76,355,1100,180,32);
footer(s,'Five model families. Predictive quality, inference cost and minority-class performance.');
}
{
const s=slide('Experimental scope and controls',note+' Source: config.json, config-v2.json, config-hard.json and config-imbalance.json. Test exposure and exploratory extensions are disclosed. All training features use their own training subset.');
table(s,[['Experiment','Labels and budget','Main question'],['IMDb / RT / Yelp','Binary, 6,000 training texts','Quality and inference cost'],['AG News','Four topics, 6,000 training texts','Multiclass quality'],['MultiNLI','Three relations, 6,000 pairs','Complex sentence semantics'],['AG News imbalance','Four fixed-N training ratios','Minority-class deterioration']],165,365,[320,400,420]);
footer(s,'TF-IDF LR/SVM, Word2Vec SVM, BiLSTM and DistilBERT. Validation selects checkpoints.');
}
{
const s=slide('Conventional classification: Macro F1',note+' Source: results-v2/evaluation.json. Macro F1 across four independent tasks. All final tests balanced. Similarity of scores is task-dependent, not a universal result.');
const tasks=['imdb','rotten_tomatoes','yelp','ag_news'];
table(s,[['Model','IMDb','RT','Yelp','AG News'],...models.map(m=>[short[m],...tasks.map(t=>v2.summary.find(r=>r.source===t&&r.model===m).macro_f1.toFixed(4))])],165,375,[420,180,180,180,180]);
footer(s,'LR and DistilBERT differ by less than 1 F1 point on IMDb/Yelp. RT and news gaps are larger.');
}
{
const s=slide('CPU inference cost on a common workload',note+' Source: results-v2/benchmark.json. IMDb-trained models, same 200 mixed review texts, batch1, four CPU threads, five repeats. Includes cleaning/tokenization/features/forward, excludes loading. Legacy results reused. Hardware RTX4090 laptop host, CPU execution only. News workloads differ.');
table(s,[['IMDb-trained model','Median ms/text','p95 ms/text'],...models.map(m=>{const r=cost.find(r=>r.model===m);return [short[m],r.median_ms.toFixed(2),r.p95_ms.toFixed(2)];})],165,375,[540,300,300]);
footer(s,'Same 200 review texts, batch1, four CPU threads, five repeats. Model loading excluded.');
}
{
const s=slide('MultiNLI sentence-pair inference',note+' Sources: https://aclanthology.org/N18-1101/ and hard-benchmark-report.md. Pair TF-IDF and Word2Vec concatenate P,H,abs(P-H),P*H. Shared BiLSTM uses pair interactions. DistilBERT jointly encodes the pair. Matched/mismatched samples are reserved official validation pools, not hidden test leaderboard scores. Premise groups never cross partitions. Hypothesis-only SVM Accuracy .4404/.4555 indicates predictive annotation cues.');
table(s,[['Model','Matched Macro F1','Mismatched Macro F1'],...models.map(m=>[short[m],d.hard[0][m].toFixed(4),d.hard[1][m].toFixed(4)])],165,375,[540,300,300]);
footer(s,'DistilBERT exceeds pair SVM by about 16 F1 points. BiLSTM shows no comparable gain.');
}
{
const s=slide('Controlled training imbalance',note+' Source: config-imbalance.json and data-imbalance/audit.json. World majority only, one fixed subset draw. Minority subsets nested, World prefixes expand, official training pool only, official test hashes excluded. Balanced reference checkpoints reused. New ratios and weighted mitigation predeclared before new runs. Existing balanced test previously inspected, so extension is exploratory.');
table(s,[['Training condition','World','Each other class','Ratio'],...Object.entries(d.counts).map(([c,n])=>[c,String(n[0]),String(n[1]),`${n[0]/n[1]}:1`])],165,330,[390,250,280,220]);
text(s,'Every training condition: 6,000 unique texts\nFixed validation and test: 1,000 each, 250 per class',76,525,1110,80,27,true,ink);
}
{
const s=slide('Macro F1 under training imbalance',note+' Source: results-imbalance/comparison.csv. Balanced test remains the same 1000 texts,250 per class. Deep means of three seeds. Full report shows training SD; this chart displays means only. Hyperparameters frozen from balanced-validation selection.');
const conditions=['balanced','mild','moderate','severe'];
const chart=s.charts.add('line',{position:{left:70,top:155,width:1140,height:420},title:'Fixed balanced-test Macro F1',titleTextStyle:{fontSize:24,fill:ink,bold:true},categories:['1:1','7:1','17:1','47:1'],series:models.map((m,i)=>({name:short[m],values:conditions.map(c=>Number(d.summary.find(r=>r.condition===c&&r.model===m).macro_f1.toFixed(6))),fill:['#a2afa8','#668477','#45605c','#56849a','#157e75'][i]})),hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18,fill:body}},xAxis:{title:{text:'Training ratio: World : each minority class',textStyle:{fontSize:19,fill:body}},textStyle:{fontSize:21,fill:body}},yAxis:{min:0,max:1,title:{text:'Macro F1 (0–1)',textStyle:{fontSize:20,fill:body}},textStyle:{fontSize:20,fill:body},majorGridlines:{fill:'#d8ddd7',width:1}},chartFill:'none',plotAreaFill:'none'});applyPresentationChartFont(chart,{fontFamily:font});
footer(s,'All curves use the same test. Deep values average three seeds. Training total stays at 6,000.');
}
{
const s=slide('Severe imbalance and minority recall',note+' Source: results-imbalance/comparison.csv. Delta is own balanced score minus severe score, percentage points. Minority recall averages Sports, Business and Sci/Tech equally. Negative deterioration is possible. Deep scores mean three seeds. Absolute severe performance and deterioration answer different questions.');
table(s,[['Model','Macro F1','F1 drop, pp','Minority R','R drop, pp'],...models.map(m=>{const r=d.severe.find(r=>r.model===m);return [short[m],r.macro_f1.toFixed(4),r.macro_f1_drop_pp.toFixed(2),r.minority_recall.toFixed(4),r.minority_recall_drop_pp.toFixed(2)];})],165,375,[420,180,180,180,180]);
footer(s,'Positive drop = balanced minus severe. Minority recall averages the three non-World classes.');
}
{
const s=slide('Class weights at severe imbalance',note+' Source: results-imbalance/mitigation.csv. SVM class_weight=balanced. DistilBERT starts again from generic pretrained checkpoint with weights N/(4*n_c), same learning rate/epochs/validation selection, three seeds. No test-based weight search. Two representative methods only.');
table(s,[['Model','Macro F1','F1 gain, pp','Minority R','R gain, pp'],...['linear_svm','distilbert'].map(m=>{const r=d.mitigation.find(r=>r.model===m);return [short[m],r.macro_f1.toFixed(4),r.macro_f1_gain_pp.toFixed(2),r.minority_recall.toFixed(4),r.minority_recall_gain_pp.toFixed(2)];})],190,255,[420,180,180,180,180]);
text(s,'Inverse-frequency class weights\nSame severe data and training budgets',76,495,1100,90,30,true,ink);
footer(s,'Positive gain = weighted minus unweighted severe performance. No test-based weight search.');
}
{
const s=slide('Why Accuracy needs context',note+' Mathematical always-World diagnostic, not a trained model or new observed deployment dataset. Balanced test metrics exact. Prior-weighted Accuracy assumes label shift with unchanged class-conditional distributions. Source definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html .');
text(s,'An always-World prediction rule',76,170,1100,65,34,true,ink);
table(s,[['Evaluation view','Accuracy','Macro F1','Minority Recall'],['Observed balanced test','25%','0.10','0%'],['Simulated 94/2/2/2 prior','94%','—','0%']],270,235,[460,180,280,220]);
footer(s,'The 94% value is a mathematical prior-weighting example. Main experimental scores stay balanced.');
}
{
const s=slide('Findings across the three experiments',note+' All conclusions limited to sampled tasks and budgets. DistilBERT external pretraining limits architecture-only attribution. One World-majority experiment does not establish universal family robustness. Pair grouping, bootstrap limitations and label shift simulation documented in detailed reports.');
const sWorst=[...d.severe].sort((a,b)=>b.macro_f1_drop_pp-a.macro_f1_drop_pp)[0];
const sBest=[...d.severe].sort((a,b)=>b.macro_f1-a.macro_f1)[0];
lines(s,['Conventional tasks: similar scores on IMDb/Yelp, with lower traditional inference latency.',
'MultiNLI: the tested pretrained Transformer shows a larger quality advantage.',
`Imbalance: ${short[sWorst.model]} has the largest F1 drop (${sWorst.macro_f1_drop_pp.toFixed(1)} pp).`,
`${short[sBest.model]} retains the highest severe-condition F1 (${sBest.macro_f1.toFixed(4)}).`],165,100,29);
footer(s,'Model choice depends on task semantics, latency constraints and minority-class requirements.');
}
{
const s=slide('English application and interpretation limits',note+' GUI: app.py four tabs. Detailed reports: final-report.md, classification-report-v2.md, hard-benchmark-report.md, imbalance-report.md. References: Williams et al.2018 https://aclanthology.org/N18-1101/ ; Gururangan et al.2018 https://aclanthology.org/N18-2017/ ; Sanh et al.2019 https://arxiv.org/abs/1910.01108 ; scikit-learn metrics https://scikit-learn.org/stable/modules/model_evaluation.html ; class weights https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html . No invented member contributions, signatures or video recording.');
lines(s,['Prediction and multi-metric comparisons for binary, news and NLI tasks.',
'Imbalance tab: decline curves, minority scores, confusion and weighted-loss results.',
'One corpus and majority class limit robustness generalization. Pretraining resources differ.',
'Sources: MultiNLI, DistilBERT and scikit-learn. Full citations and data versions in reports.'],165,100,28);
footer(s,'Three-seed SD measures training variation. Conditional bootstrap intervals address test sampling.');
}
await fs.mkdir(tmp,{recursive:true});const candidate=path.join(tmp,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(root,'report/final-presentation-r2.pptx');
const tableOwners=[2,3,4,5,6,8,9,10];
const result=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[7],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation-r2.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));for(let i=0;i<final.slides.items.length;i++){const img=await final.export({slide:final.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));}
console.log(JSON.stringify({finalPath,slides:final.slides.items.length,validated:result.finalSha256}));

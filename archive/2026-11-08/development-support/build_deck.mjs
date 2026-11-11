import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
process.env.RUNTIME_NODE=path.join(runtime,'node/bin/node.exe');
process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
const skill='C:/Users/LUZIJ/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.12130/skills/presentations';
const project='C:/Users/LUZIJ/Documents/Codex/2026-11-06/ni/outputs/sentiment-project';
const workspace='C:/Users/LUZIJ/Documents/Codex/2026-11-06/ni';
const tmp=path.join(workspace,'work/deck-build');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data=JSON.parse(await fs.readFile(path.join(project,'report/slide-data.json'),'utf8'));
const evaluation=JSON.parse(await fs.readFile(path.join(project,'results/evaluation.json'),'utf8'));
await fs.mkdir(tmp,{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Arial',ink='#173c35',body='#253a34',muted='#5a6962',bg='#f7f6f0';
const names=['imdb','rotten_tomatoes','yelp'];
const models=['logistic_regression','linear_svm','distilbert'];
const labels={imdb:'IMDb',rotten_tomatoes:'Rotten Tomatoes',yelp:'Yelp'};
const short={logistic_regression:'TF-IDF + LR',linear_svm:'TF-IDF + SVM',distilbert:'DistilBERT'};
const sources={imdb:'https://huggingface.co/datasets/stanfordnlp/imdb',rotten_tomatoes:'https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes',yelp:'https://huggingface.co/datasets/fancyzhx/yelp_polarity',model:'https://huggingface.co/distilbert/distilbert-base-uncased',paper:'https://arxiv.org/abs/1910.01108'};
function text(s,t,x,y,w,h,size=28,color=body,bold=false){
 const obj=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 obj.text=t;obj.text.style={fontFamily:font,fontSize:size,color,bold};return obj;
}
function slide(title,notes=''){
 const s=p.slides.add();s.background.fill=bg;
 text(s,title,64,44,1150,80,42,ink,true);
 s.speakerNotes.text=notes;
 text(s,String(p.slides.items.length).padStart(2,'0'),1175,674,48,25,16,muted);
 return s;
}
function table(s,values,x,y,w,h,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:'#d8ddd7',width:1,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?ink:bg;
  cell.text.style={fontFamily:font,fontSize:23,color:r===0?'#ffffff':body,bold:r===0};
 }
 return t;
}
function lines(s,items,start=174,step=98,size=29){
 items.forEach((t,i)=>text(s,t,76,start+i*step,1110,step-12,size));
}
function note(s,t){text(s,t,76,615,1110,52,20,muted);}
function chart(s,title,categories,series,pos,ytitle,max){
 const chartSeries=series.map(item=>({...item,values:item.values.map(v=>Number(v.toFixed(6)))}));
 const c=s.charts.add('bar',{position:pos,title,titleTextStyle:{fontSize:24,fill:ink,bold:true},categories,series:chartSeries,
  hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:22,fill:body}},
  barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
  xAxis:{title:{text:'Dataset',textStyle:{fontSize:20,fill:body}},textStyle:{fontSize:22,fill:body},majorGridlines:null},
  yAxis:{title:{text:ytitle,textStyle:{fontSize:20,fill:body}},min:0,...(max?{max}:{}),numberFormatCode:'0.00',textStyle:{fontSize:20,fill:body},majorGridlines:{fill:'#d8ddd7',width:1}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:20,fill:body}},chartFill:'none',plotAreaFill:'none'});
 applyPresentationChartFont(c,{fontFamily:font});return c;
}

{
const s=p.slides.add();s.background.fill=bg;
text(s,'Cross-Dataset\nSentiment Classification',72,145,1140,190,62,ink,true);
text(s,'TF-IDF baselines and fine-tuned DistilBERT',76,390,1110,60,32,body);
text(s,'Accuracy, transferability and computational cost',76,472,1110,55,27,muted);
s.speakerNotes.text='NLP group project. All quantitative claims come from the local frozen evaluation. Actual team names, contributions and narration require confirmation by the real members. Target narration length: approximately nine minutes.';
}
{
const s=slide('Problem and research questions','Use case: support readers who need a first-pass positive/negative label for English reviews. Scope: binary English sentiment classification, not new architecture or production deployment.');
lines(s,['English movie and business reviews receive positive or negative labels.',
'Does contextual pretraining improve over strong sparse baselines?',
'How well does a saved model transfer to another review corpus?',
'What training and inference costs accompany the measured gains?'],170,105,29);
}
{
const s=slide('Data and fixed sample budgets',Object.values(sources).slice(0,3).join('\n')+'\nSee data/audit.json and assets.lock.json. Training and validation labels serve model fitting/selection. Test examples retain official test provenance. Each sampled split is balanced.');
table(s,[['Dataset','Review type','Train','Validation','Test'],['IMDb','Movie, long','6,000','1,000','1,000'],['Rotten Tomatoes','Movie, short','6,000','1,000','1,000'],['Yelp','Business / service','6,000','1,000','1,000']],70,168,1140,286,[270,325,180,190,175]);
text(s,`${data.data_audit.raw_rows.toLocaleString()} original rows audited before sampling\n24,000 selected examples with globally unique normalized hashes`,76,492,1110,96,29);
note(s,'Two movie corpora share a domain. Length, style and label construction also differ.');
}
{
const s=slide('Representations and classifiers',sources.paper+'\n'+sources.model+'\nhttps://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\nTF-IDF uses training-only vocabulary and IDF. Logistic Regression estimates probabilities. LinearSVC produces margins. DistilBERT uses contextual self-attention and a two-class head. External pretraining prevents an architecture-only causal comparison.');
text(s,'Sparse text features',76,165,490,55,32,ink,true);
text(s,'TF-IDF word unigrams / bigrams\n\nLogistic Regression\nLinear SVM\n\nFull cleaned review as input',76,240,500,330,29);
text(s,'Contextual representation',684,165,510,55,32,ink,true);
text(s,'Six-layer DistilBERT encoder\n\nBidirectional self-attention\nTwo-class classification head\n\nFirst 256 tokens as input',684,240,510,330,29);
note(s,'Base pretrained weights plus a new sentiment head. No ready-made sentiment classifier.');
}
{
const s=slide('Validation selection before test evaluation','Sources: config.json, data/manifest.csv, per-model metadata.json, results/protocol.freeze.json. Thirty-six traditional candidate fits. Twelve DistilBERT training runs, including two learning-rate candidates per source and the two additional seeds for the selected LR. Fifteen final artifacts: six traditional plus nine Transformer. Optional learning curves excluded before testing.');
lines(s,['Source-only validation selects hyperparameters and checkpoints.',
'Sparse models: 2 n-gram choices × 3 regularization values per source.',
'DistilBERT: 2 learning rates, 3 epochs and seeds 42 / 43 / 44.',
'Frozen sample manifests and 15 final model files precede testing.'],168,105,29);
note(s,'TF-IDF fits source training text only. Direct transfer uses no target-label adaptation.');
}
{
const s=slide('Within-dataset performance','Source: results/metrics.csv, source==target. Values are Macro-F1 on 1,000 fixed test reviews per dataset. DistilBERT averages three training seeds. Traditional models have one fixed fit. The report contains per-seed SD and the full predictions.');
chart(s,'Macro-F1 on fixed test samples',names.map(n=>labels[n]),models.map((m,i)=>({name:short[m],values:names.map(n=>data.within.find(r=>r.source===n&&r.model===m).mean_f1),fill:['#aeb7b0','#66776b','#1b766e'][i]})),{left:70,top:140,width:1140,height:448},'Macro-F1 (0–1)',1.05);
note(s,'DistilBERT bars average three seeds. Per-seed variation appears in the report, not as a test confidence interval.');
}
{
const s=slide('Direct transfer across review corpora','Source: results/metrics.csv. Rows: training sources IMDb, Rotten Tomatoes (RT), Yelp. Columns: test targets in the same order. No target-data fine-tuning. Diagonal repeats within-dataset results. DistilBERT cells average three seeds. Corpus difficulty and text length confound pure domain-shift interpretations.');
for(let i=0;i<3;i++){
 const m=models[i],x=65+i*402;
 text(s,short[m],x,151,385,50,28,ink,true);
 const values=[['Train / Test','IMDb','RT','Yelp'],...names.map((n,r)=>[n==='rotten_tomatoes'?'RT':labels[n],...data.transfer[m][r].map(v=>v.toFixed(3))])];
 const t=table(s,values,x,220,382,285,[142,80,80,80]);
 for(let r=0;r<4;r++)for(let c=0;c<4;c++)t.getCell(r,c).text.style={fontFamily:font,fontSize:21,color:r===0?'#ffffff':body,bold:r===0};
}
note(s,'Rows = training source. Columns = test target. Values = Macro-F1. RT = Rotten Tomatoes.');
}
{
const s=slide('Paired comparison on the same test IDs','Source: results/bootstrap.json. One thousand paired, class-stratified bootstrap samples. DistilBERT seed42 and SVM selected in advance. Percentile intervals condition on the two trained models, exclude training uncertainty and do not correct for multiple comparisons.');
table(s,[['Dataset','DistilBERT minus SVM','95% paired interval'],...names.map(n=>[labels[n],`${data.bootstrap[n].delta_macro_f1>=0?'+':''}${data.bootstrap[n].delta_macro_f1.toFixed(4)}`,`[${data.bootstrap[n].ci95_low.toFixed(4)}, ${data.bootstrap[n].ci95_high.toFixed(4)}]`])],70,190,1140,270,[350,395,395]);
text(s,'Comparison uses DistilBERT seed42.\nThese intervals describe test-sample uncertainty for fixed models.',76,505,1100,100,28);
}
{
const s=slide('CPU inference cost',`Source: results/benchmark.json. All models process the same 200 mixed-source review texts with batch size1, four CPU threads, 20 warm-up examples and five repeats. This slide uses IMDb-trained checkpoints, DistilBERT seed42. Timing includes cleaning and feature conversion, excludes loading/download. Training GPU: ${data.hardware.gpu}.`);
const costs=models.map(m=>data.costs.find(r=>r.source==='imdb'&&r.model===m));
const c=chart(s,'IMDb-trained models on the same review set',models.map(m=>short[m]),[{name:'Median',values:costs.map(r=>r.median_ms),fill:'#1b766e'},{name:'p95',values:costs.map(r=>r.p95_ms),fill:'#9ca99e'}],{left:70,top:140,width:1140,height:448},'Latency (ms / review)');
c.xAxis={title:{text:'Model',textStyle:{fontSize:20,fill:body}}};
c.dataLabels={numberFormatCode:'0.00'};
note(s,'One CPU, 4 threads, batch=1. 200 reviews × 5 repeats. Training and model-size costs appear in the report.');
}
{
const s=slide('Classification errors and truncation','Source: results/evaluation.json for DistilBERT seed42 within-dataset confusion matrices, data/audit.json for truncation. False positives are negative labels predicted positive; false negatives are positive labels predicted negative. Error-review queue is a draft for genuine human checking.');
const rows=names.map(n=>{
const row=evaluation.rows.find(r=>r.source===n&&r.target===n&&r.model==='distilbert'&&r.seed===42);
return [labels[n],String(row.confusion_matrix[0][1]),String(row.confusion_matrix[1][0]),data.truncation_percent[names.indexOf(n)].toFixed(1)+'%'];});
table(s,[['Dataset','False positives','False negatives','Input truncated'],...rows],70,180,1140,260,[330,260,260,290]);
text(s,`${data.error_queue_size} error / disagreement cases form the review queue.\nHuman reviewers must verify categories and explanations.`,76,490,1110,100,28);
note(s,'Long-review truncation is measured, not inferred as the cause of every error.');
}
{
const s=slide('Python implementation and live demonstration','Demo script: report/narration-script-zh.md. Show actual Python code, dataset audit, frozen artifacts, predictions and application. The demonstration fixes DistilBERT seed42. All narration must come from real team members, not generated voice.');
lines(s,['pipeline.py: data isolation, source validation and checkpoint freezing',
'app.py: choose a training source and compare all three predictions',
'Results page: within-dataset scores, transfer matrices and CPU timing',
'Margin and probability outputs have different interpretations'],175,105,29);
}
{
const best=models.reduce((a,b)=>data.macro_average[a]>=data.macro_average[b]?a:b);
const s=slide('Conclusions and limits','Source: report/slide-data.json, results/metrics.csv and report/experiment-report.md. Equal-weight means are calculated across three within-dataset Macro-F1 values. They are not pooled F1 or full-benchmark performance. Corpus/label differences, external pretraining and truncated inputs limit generalization. No unsupported novelty or causal architecture claim.');
text(s,`${short[best]} has the highest mean Macro-F1\nin this fixed three-corpus protocol`,76,170,1110,125,38,ink,true);
text(s,models.map(m=>`${short[m]}: ${data.macro_average[m].toFixed(4)}`).join('\n'),76,330,530,170,29);
text(s,'Binary English labels only\nExternal pretraining differs\nLong inputs truncate\nNear-duplicates may remain',690,330,490,210,28);
note(s,'Results support this sampled experiment. Broader deployment requires new validation.');
}
{
const s=slide('References',Object.values(sources).join('\n')+'\nhttps://aclanthology.org/P11-1015/\nhttps://www.cs.cornell.edu/people/pabo/movie-review-data/\nhttps://arxiv.org/abs/1509.01626\nhttps://huggingface.co/docs/transformers/tasks/sequence_classification');
lines(s,['Maas et al. (2011). Learning Word Vectors for Sentiment Analysis.',
'Pang and Lee (2005). Seeing stars. Cornell movie-review data.',
'Zhang et al. (2015). Character-level CNNs for Text Classification.',
'Sanh et al. (2019/2020). DistilBERT.',
'Hugging Face model / dataset cards and scikit-learn documentation.'],160,88,27);
}

await fs.writeFile(path.join(tmp,'presentation.json'),JSON.stringify(p.toProto()));
for(let i=0;i<p.slides.items.length;i++){
 const img=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));
}
const candidate=path.join(tmp,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(project,'report/project-presentation.pptx');
const result=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:path.join(runtime,'python/python.exe'),
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,7,8,10].flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:[3,7,8,10],requiredNativeChartOwnerSlides:[6,9],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(tmp,'validation.json')});
console.log(JSON.stringify({slides:p.slides.items.length,finalPath,result}));

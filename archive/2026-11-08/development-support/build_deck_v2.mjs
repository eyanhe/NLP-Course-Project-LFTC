import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
process.env.RUNTIME_NODE=path.join(runtime,'node/bin/node.exe');
process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
const skill='C:/Users/LUZIJ/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.12130/skills/presentations';
const workspace='C:/Users/LUZIJ/Documents/Codex/2026-11-06/ni';
const root=path.join(workspace,'outputs/sentiment-project');const tmp=path.join(workspace,'work/deck-v2');
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const d=JSON.parse(await fs.readFile(path.join(root,'report/slide-data-v2.json'),'utf8'));
const costs=JSON.parse(await fs.readFile(path.join(root,'results-v2/benchmark.json'),'utf8')).rows;
const names=d.sources,models=d.models;
const short={logistic_regression:'TF-IDF + LR',linear_svm:'TF-IDF + SVM',word2vec_svm:'Word2Vec + SVM',bilstm:'Word2Vec + BiLSTM',distilbert:'DistilBERT'};
const titles={imdb:'IMDb',rotten_tomatoes:'Rotten Tomatoes',yelp:'Yelp',ag_news:'AG News'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#173c35',body='#253a34',muted='#5a6962',bg='#f7f6f0',font='Arial';
function text(s,t,x,y,w,h,size=28,bold=false,color=body){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={fontFamily:font,fontSize:size,bold,color};return a;}
function slide(title,notes){const s=p.slides.add();s.background.fill=bg;text(s,title,64,42,1150,90,42,true,ink);text(s,String(p.slides.items.length).padStart(2,'0'),1175,675,48,25,16,false,muted);s.speakerNotes.text=notes;return s;}
function footer(s,t){text(s,t,76,610,1110,58,20,false,muted);}
function lines(s,items,y=170,step=106,size=29){items.forEach((t,i)=>text(s,t,76,y+i*step,1110,step-12,size));}
function table(s,values,x=70,y=170,w=1140,h=380,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});t.borders.assign({fill:'#d8ddd7',width:1});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const a=t.getCell(r,c);a.fill=r===0?ink:bg;a.text.style={fontFamily:font,fontSize:23,bold:r===0,color:r===0?'#ffffff':body};}return t;}
function chart(s,title,categories,series,yTitle,max=undefined){const c=s.charts.add('bar',{position:{left:70,top:140,width:1140,height:440},title,titleTextStyle:{fontSize:24,fill:ink,bold:true},categories,series:series.map(r=>({...r,values:r.values.map(v=>Number(v.toFixed(6)))})),hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:19,fill:body}},barOptions:{direction:'column',grouping:'clustered'},xAxis:{textStyle:{fontSize:20,fill:body},majorGridlines:null},yAxis:{min:0,...(max?{max}:{}),title:{text:yTitle,textStyle:{fontSize:20,fill:body}},numberFormatCode:'0.00',textStyle:{fontSize:20,fill:body},majorGridlines:{fill:'#d8ddd7',width:1}},chartFill:'none',plotAreaFill:'none'});applyPresentationChartFont(c,{fontFamily:font});return c;}
const own='Sources: results-v2/evaluation.json, results-v2/metrics.csv and report/classification-report-v2.md. All displayed numbers are actual frozen evaluation results.';
const row=(source,model)=>d.summary.find(r=>r.source===source&&r.model===model);
const news=models.map(m=>row('ag_news',m));
{
const s=slide('Binary and Multiclass\nText Classification','Course NLP project V2. Actual members must record real narration and verify contributions. No generated voice. This deck supersedes the V1 presentation for the expanded scope.');
text(s,'Traditional NLP and deep learning',76,280,1110,80,38,true,ink);
text(s,'Four datasets, five methods and multiple evaluation metrics',76,396,1110,90,30);
text(s,'TF-IDF, Word2Vec, BiLSTM and DistilBERT',76,510,1110,65,28,false,muted);
}
{
const s=slide('Course alignment and research questions','The course examples list possible approaches, not a requirement to implement every listed algorithm. We include traditional features with SVM, a BiLSTM sequence model and a BERT-family Transformer. CRF is not used for these document-level labels.');
lines(s,['Traditional features with SVM versus sequential and Transformer models',
'Binary sentiment and four-class news categorization',
'Precision, Recall and F1 reveal different classification errors',
'Accuracy and inference cost provide additional practical context']);
}
{
const s=slide('Four datasets with fixed sample budgets','Sources: original data/audit.json and data-v2/audit.json; https://huggingface.co/datasets/fancyzhx/ag_news. AG News official 120000 train, 7600 test. New samples preserve official test provenance. All sampled partitions are class-balanced and hash-isolated.');
table(s,[['Dataset','Task','Classes','Train / Val / Test'],['IMDb','Sentiment','2','6000 / 1000 / 1000'],['Rotten Tomatoes','Sentiment','2','6000 / 1000 / 1000'],['Yelp','Sentiment','2','6000 / 1000 / 1000'],['AG News','News topic','4','6000 / 1000 / 1000']],70,170,1140,348,[290,270,150,430]);
footer(s,'AG News: World, Sports, Business, Sci/Tech. Each news test class has 250 examples.');
}
{
const s=slide('Representations and model families','Sources: extension.py, models_extra.py, config-v2.json. Word2Vec: https://radimrehurek.com/gensim/models/word2vec.html. DistilBERT: https://arxiv.org/abs/1910.01108. All local vocabulary/embeddings use training text only.');
table(s,[['Model','Representation','Classifier / encoder'],['TF-IDF + LR','Sparse word n-grams','Logistic Regression'],['TF-IDF + SVM','Sparse word n-grams','Linear SVM'],['Word2Vec + SVM','Mean of 100D word vectors','Scaled Linear SVM'],['Word2Vec + BiLSTM','Trainable word embeddings','64 units per direction'],['DistilBERT','Pretrained contextual tokens','Fine-tuned classification head']],70,160,1140,410,[320,390,430]);
footer(s,'Local Word2Vec and externally pretrained DistilBERT have different prior resources.');
}
{
const s=slide('Validation selection and protocol history','Source: config-v2.json and results-v2/protocol.freeze.json. V2 follows original sentiment test inspection. New binary comparisons are exploratory extensions. News tests were newly reserved. Word2Vec is fit once per source and reused for vector SVM and sequence initialization.');
lines(s,['Training-only text features and source validation select configurations.',
'BiLSTM compares two learning rates over five epochs.',
'DistilBERT compares two learning rates over three epochs.',
'Deep models use three seeds. V2 freezes 36 final model artifacts.']);
footer(s,'Added binary models extend an already inspected benchmark; AG News has newly reserved tests.');
}
{
const s=slide('Precision, Recall and F1','Source: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics. Precision=TP/(TP+FP). Recall=TP/(TP+FN). F1=2PR/(P+R). Micro equals Accuracy in single-label classification. Weighted and Macro coincide under the balanced supports used here.');
lines(s,['Precision: how often a predicted class label is correct',
'Recall: how much of a true class the model finds',
'F1: harmonic mean of Precision and Recall',
'Macro: equal class weight. Weighted: class support. Micro: pooled decisions.'],160,100,29);
footer(s,'Balanced tests make Macro = Weighted. Single-label Micro P/R/F1 = Accuracy. These are not independent evidence.');
}
{
const s=slide('Sentiment results across three corpora',own+' This slide averages each model metric equally over the three within-dataset sentiment corpora. It is not pooled-prediction F1. Deep metrics first average seeds within each corpus.');
const values=models.map(m=>[short[m],...['accuracy','macro_precision','macro_recall','macro_f1'].map(k=>(names.slice(0,3).reduce((a,n)=>a+row(n,m)[k],0)/3).toFixed(4))]);
table(s,[['Model','Accuracy','Macro P','Macro R','Macro F1'],...values],70,170,1140,370,[380,190,190,190,190]);
footer(s,'Equal-weight mean over IMDb, Rotten Tomatoes and Yelp. Full corpus-level results remain in the report.');
}
{
const s=slide('Four-class AG News results',own+' Deep-model values average three seeds. Traditional models use one fixed fit. No significance claim is inferred merely from the largest point estimate.');
table(s,[['Model','Accuracy','Macro P','Macro R','Macro F1'],...models.map(m=>[short[m],...['accuracy','macro_precision','macro_recall','macro_f1'].map(k=>row('ag_news',m)[k].toFixed(4))])],70,170,1140,370,[380,190,190,190,190]);
footer(s,'Class-balanced official-test sample: 1,000 examples. Deep models: seeds 42/43/44.');
}
{
const s=slide('Macro F1 across binary and multiclass tasks',own+' Do not attribute differences between corpora purely to their number of classes. Different domains, lengths and label definitions remain confounders.');
chart(s,'Within-dataset Macro F1',names.map(n=>titles[n]),models.map((m,i)=>({name:short[m],values:names.map(n=>row(n,m).macro_f1),fill:['#bdc5c0','#899a91','#596f66','#4e7788','#1b766e'][i]})),'Macro F1 (0–1)',1);
footer(s,'Each corpus has a different task distribution. Deep bars average three seeds.');
}
{
const s=slide('Four-class confusion matrices',own+' Matrices use seed42 for deep models and the fixed SVM fit. Row labels are true classes; columns are predicted classes. W=World, S=Sports, B=Business, T=Sci/Tech.');
for(let z=0;z<3;z++){
 const m=['linear_svm','bilstm','distilbert'][z],x=60+405*z;
 text(s,short[m],x,160,385,48,28,true,ink);
 const values=[['Class','W','S','B','T'],...d.news_confusions[m].map((v,i)=>[['W','S','B','T'][i],...v.map(String)])];
 const t=table(s,values,x,230,380,300,[100,70,70,70,70]);
 for(let i=0;i<5;i++)for(let j=0;j<5;j++)t.getCell(i,j).text.style={fontFamily:font,fontSize:21,color:i===0?'#ffffff':body,bold:i===0};
}
footer(s,'Rows: true class. Columns: predicted class. W=World, S=Sports, B=Business, T=Sci/Tech.');
}
{
const s=slide('AG News class-level F1',own+' Values average per-class F1 across seeds 42/43/44. See results-v2/class_metrics.csv and the GUI for per-class Precision, Recall and support.');
const per=JSON.parse(await fs.readFile(path.join(root,'results-v2/evaluation.json'),'utf8')).rows;
chart(s,'BiLSTM and DistilBERT by news category',d.labels,['bilstm','distilbert'].map((m,i)=>({name:short[m],values:d.labels.map(label=>{const r=per.filter(r=>r.source==='ag_news'&&r.target==='ag_news'&&r.model===m);return r.reduce((v,r)=>v+r.classification_report[label]['f1-score'],0)/r.length;}),fill:i?'#1b766e':'#4e7788'})),'Class F1 (0–1)',1);
footer(s,'Per-class Precision and Recall expose whether an error is a false positive or a miss.');
}
{
const s=slide('Inference cost and model choice','Source: results-v2/benchmark.json. Same 200 review texts, batch 1, four CPU threads, 20 warm-up texts and five repetitions. Measurements include cleaning and feature conversion, exclude loading. AG News-trained models also use the common review workload, not news-serving inputs. Legacy timings reused.');
chart(s,'AG News-trained models on a common review workload',models.map(m=>short[m]),[{name:'Median (ms)',values:models.map(m=>costs.find(r=>r.source==='ag_news'&&r.model===m).median_ms),fill:'#1b766e'},{name:'p95 (ms)',values:models.map(m=>costs.find(r=>r.source==='ag_news'&&r.model===m).p95_ms),fill:'#9ca99e'}],'Latency (ms / review)');
footer(s,'This common review workload does not measure news-serving latency. Loading is excluded.');
}
{
const s=slide('English application and reproducible evidence','Application source: app.py. Reports: classification-report-v2.md, results-summary-v2-zh.md. Demo seed42 fixed. Show the actual task switch, all five model predictions, the metric averaging selector, per-class P/R/F1, and count/normalized confusion matrices.');
lines(s,['Binary sentiment and Multiclass news task selector',
'Five model predictions, with probability and margin labels',
'Accuracy, P/R/F1 averages and per-class confusion matrices',
'Frozen sample hashes, model hashes and 66,000 stored predictions']);
}
{
const best=news.reduce((a,b)=>a.macro_f1>=b.macro_f1?a:b);
const s=slide('Conclusions and limitations',own+' Highest mean Macro F1 is descriptive. No new significance test covers every added model pair. Word2Vec and external pretraining differ. The original bootstrap evidence remains conditional on its predefined comparison.');
text(s,`${short[best.model]} has the highest AG News\nmean Macro F1 in this protocol (${best.macro_f1.toFixed(4)})`,76,160,1110,135,36,true,ink);
lines(s,['Model choice depends on false positives, missed classes and inference cost.',
'Different corpora do not isolate the effect of class count.',
'Truncation, training resources and prior test exposure limit conclusions.'],340,84,27);
}
{
const s=slide('Sources and references','https://huggingface.co/datasets/fancyzhx/ag_news\nhttps://radimrehurek.com/gensim/models/word2vec.html\nhttps://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html\nhttps://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics\nhttps://arxiv.org/abs/1910.01108\nOriginal sentiment sources and papers remain in experiment-report.md.');
lines(s,['AG News and the original sentiment dataset cards',
'Mikolov et al. Word2Vec and the Gensim implementation',
'PyTorch LSTM documentation and the bidirectional sequence model',
'Sanh et al. DistilBERT and scikit-learn classification metrics'],170,105,28);
}
await fs.mkdir(tmp,{recursive:true});
const candidate=path.join(tmp,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(root,'report/classification-presentation-v2.pptx');
const result=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[3,4,7,8,10].flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:[3,4,7,8,10],requiredNativeChartOwnerSlides:[9,11,12],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<final.slides.items.length;i++){const img=await final.export({slide:final.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));}
console.log(JSON.stringify({finalPath,slides:final.slides.items.length,validated:result.finalSha256}));

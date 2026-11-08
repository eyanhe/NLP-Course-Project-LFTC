import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
process.on('uncaughtException',error=>{process.stderr.write(error.name+': '+error.message+'\n');process.exit(1);});

const project=process.cwd();
const workspace=path.join(project,'outputs/research-ppt-20261108');
const build=path.join(workspace,'.build'),images=path.join(workspace,'images'),files=path.join(workspace,'files');
const finalOutput=path.join(files,'文本分类研究图表-演示版-r2.pptx');
for(const dir of [build,images,files])await fs.mkdir(dir,{recursive:true});
const skill='C:/Users/超级罪犯 顶风作案/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.22356/skills/presentations';
const runtime='C:/Users/超级罪犯 顶风作案/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE=path.join(runtime,'node/bin/node.exe');
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
process.env.RUNTIME_BIN_DIR=path.join(runtime,'bin');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
if(process.argv.includes('--render-final')){
  const final=await PresentationFile.importPptx(await FileBlob.load(finalOutput));
  for(let i=0;i<final.slides.items.length;i++){
    const png=await final.export({slide:final.slides.items[i],format:'png',scale:1});
    await fs.writeFile(path.join(images,`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
  }
  console.log(JSON.stringify({renderedFinalSlides:final.slides.items.length}));process.exit(0);
}
const atlasPath=path.join(project,'research-v3/reports/overnight/complete-atlas-v1/figure-data.json');
const atlas=JSON.parse(await fs.readFile(atlasPath,'utf8'));
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
for(const [relative,hash] of Object.entries(atlas.source_sha256))assert.equal(digest(await fs.readFile(path.join(project,relative))),hash,relative);
const get=n=>atlas.pages.find(r=>r.page===n).data;
const scale=get(1),s1=get(2),curves=get(3),primary=get(4).primary,refs=get(4).historical_Q,
      m2=get(5),imbalance=get(6),s3=get(7),trajectories=get(8),s4=get(9),cases=get(10),cost=get(11).records;
assert.equal(primary.length,6);assert.equal(s1.length,16);assert.equal(s3.length,8);
const SCALE=1.5,W=1920,H=1080,FONT='Microsoft YaHei';
const C={ink:'#182A43',muted:'#66758A',blue:'#376CF3',gold:'#D99932',green:'#139A82',
         red:'#D05B63',gray:'#B9C6D6',grid:'#E6EBF2',white:'#FFFFFF',light:'#F4F7FB'};
const models=['tfidf_svm','bilstm','distilbert'];
const names={tfidf_svm:'SVM',bilstm:'BiLSTM',distilbert:'DistilBERT'};
const mc={tfidf_svm:C.blue,bilstm:C.gold,distilbert:C.green};
const dsNames={banking77:'BANKING77',go_emotions:'GoEmotions',yelp:'Yelp',yahoo_answers_topics:'Yahoo'};
const p=Presentation.create({slideSize:{width:W,height:H}});
p.theme.defaultFont=FONT;
const records=[],chartOwners=[],tableOwners=[];

function text(slide,value,x,y,w,h,size=24,bold=false,color=C.ink){
  const sh=slide.shapes.add({geometry:'textbox',position:{left:x*SCALE,top:y*SCALE,width:w*SCALE,height:h*SCALE},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;sh.text.style={fontFamily:FONT,typeface:FONT,fontSize:size*SCALE,bold,color};sh.text.insets=0;return sh;
}
function rule(s,x,y,w,color=C.grid){return s.shapes.add({geometry:'line',position:{left:x*SCALE,top:y*SCALE,width:w*SCALE,height:0},line:{fill:color,width:SCALE}});}
function visualScale(value,key=''){
  if(typeof value==='number')return ['left','top','width','height','fontSize'].includes(key)?value*SCALE:value;
  if(Array.isArray(value))return value.map(v=>visualScale(v));
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,visualScale(v,k)]));
  return value;
}
function slide(title,scope,detail,topic){
  const s=p.slides.add();s.background.fill=C.white;
  const n=p.slides.items.length;
  text(s,title,64,38,1152,84,44,true);
  text(s,scope,66,128,1130,42,23,false,C.muted);
  rule(s,66,656,1148);
  text(s,topic,66,674,1050,24,16,false,C.muted);
  text(s,String(n).padStart(2,'0'),1160,673,50,26,18,false,C.muted);
  s.speakerNotes.text=detail+'\n\n数据来源：research-v3/reports/overnight/complete-atlas-v1/figure-data.json。原始分析：analysis-v2/analysis-data.json。图册来源SHA已核验。';
  records.push({slide:n,title,scope,topic,notes:detail,chartCount:0});return s;
}
function note(s,value,color=C.muted){text(s,value,66,602,1140,40,21,false,color);}
function headline(s,number,caption,x=970,y=245,color=C.green){
  text(s,number,x,y,250,90,58,true,color);text(s,caption,x,y+98,245,110,25,false,C.ink);
}
function addChart(s,type,categories,series,pos,opts={}){
  for(const set of series)assert(set.values.every(v=>v===null||Number.isFinite(v)),`Non-finite chart input: ${set.name}`);
  // Six decimals on the 0–100 display scale retain far more precision than the visible labels,
  // while keeping native chart literals exactly portable to Excel's numeric cells.
  series=series.map(set=>({...set,values:set.values.map(v=>v===null?null:Number(v.toFixed(6)))}));
  if(type==='bar')series=series.map(set=>({...set,line:{style:'solid',fill:'none',width:0},points:set.points?.map(point=>({...point,line:{style:'solid',fill:'none',width:0}}))}));
  const cfg={position:pos,categories,series,hasLegend:series.length>1,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:21,fill:C.muted}},
    chartFill:'none',plotAreaFill:'none',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    xAxis:{textStyle:{fontSize:23,fill:C.ink},line:{fill:C.grid,width:1},majorGridlines:null},
    yAxis:{min:0,max:100,majorUnit:25,numberFormatCode:'0',textStyle:{fontSize:20,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.grid,width:1}},
    dataLabels:{showValue:false,textStyle:{fontSize:23,fill:C.ink}},...opts};
  const chart=s.charts.add(type,visualScale(cfg));applyPresentationChartFont(chart,{fontFamily:FONT});
  const n=p.slides.items.length;if(!chartOwners.includes(n))chartOwners.push(n);
  records.at(-1).chartCount++;return chart;
}
const series=(name,values,color,extra={})=>({name,values,fill:color,line:{fill:color,width:3},marker:{symbol:'circle',size:7,fill:color},...extra});
function col(s,categories,sets,pos,opts={}){
  return addChart(s,'bar',categories,sets,pos,{barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:23,fill:C.ink,bold:true}},...opts});
}
function line(s,categories,sets,pos,opts={}){
  return addChart(s,'line',categories,sets,pos,{lineOptions:{smooth:false},...opts});
}
function panel(s,title,x,y=196,w=550){text(s,title,x,y,w,40,27,true);}
function twoScale(datasets,title,topic){
  const s=slide(title,'训练文本 1,000 → 6,000；原固定预算V比较',
    'E1。BiLSTM10轮、DistilBERT5轮。每个任务内固定评估条件；数据规模变化不等于精确能力临界点。\n'+JSON.stringify(scale.filter(r=>datasets.includes(r.dataset))),topic);
  datasets.forEach((ds,i)=>{
    const x=64+i*600;panel(s,dsNames[ds],x+8);
    const sets=models.map(m=>series(names[m],scale.filter(r=>r.dataset===ds&&r.model===m).sort((a,b)=>a.n-b.n).map(r=>r.score*100),mc[m]));
    line(s,['1,000条','6,000条'],sets,{left:x,top:242,width:550,height:340},
      {yAxis:{min:0,max:ds==='go_emotions'?50:100,majorUnit:ds==='go_emotions'?10:25,numberFormatCode:'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:ds==='go_emotions'?'V Macro-AP（0–100）':'V Macro-F1（0–100）',textStyle:{fontSize:20,fill:C.muted}}}});
  });
  note(s,'旧固定预算用于发现现象；优化后的泛化比较见正式Q结果');return s;
}
function sensitivity(model,title,topic){
  const s=slide(title,'S1独立Q；比较学习率与总训练预算',
    '全部固定检查点在读取Q前确定。单seed42。总预算变化也改变线性衰减计划，不能解释为纯epoch效应。\n'+JSON.stringify(s1.filter(r=>r.model===model)),topic);
  ['banking77','go_emotions'].forEach((ds,i)=>{
    const x=64+i*600;panel(s,dsNames[ds],x+8);
    const rows=s1.filter(r=>r.dataset===ds&&r.model===model),lrs=[...new Set(rows.map(r=>r.lr))].sort((a,b)=>a-b);
    const budgets=[...new Set(rows.map(r=>r.budget))].sort((a,b)=>a-b);
    const sets=lrs.map((lr,k)=>series(`lr ${lr}`,rows.filter(r=>r.lr===lr).sort((a,b)=>a.budget-b.budget).map(r=>100*r.score),[C.blue,C.green][k],
       {dataLabels:{showValue:true,position:k===0?'bottom':'top',numberFormatCode:'0.0',textStyle:{fontSize:22,fill:[C.blue,C.green][k],bold:true}}}));
    const max=model==='bilstm'?(ds==='banking77'?8:16):(ds==='banking77'?85:55);
    line(s,budgets.map(b=>`${b}轮`),sets,{left:x,top:242,width:550,height:340},
      {yAxis:{min:0,max,numberFormatCode:'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:ds==='go_emotions'?'Q Macro-AP（0–100）':'Q Macro-F1（0–100）',textStyle:{fontSize:20,fill:C.muted}}}});
  });
  note(s,'单seed42；预算变化包含学习率调度变化，各子图使用标注的指标范围');return s;
}
function learning(ds,title,topic){
  const s=slide(title,`${dsNames[ds]}；V选定配方的三个训练种子`,
    '逐轮验证指标与最终选中检查点不同于训练停止位置。12个最终神经模型都触发patience；这不证明全局最优。\n'+JSON.stringify(curves.filter(r=>r.dataset===ds)),topic);
  ['bilstm','distilbert'].forEach((m,i)=>{
    const x=64+i*600,rows=curves.filter(r=>r.dataset===ds&&r.model===m).sort((a,b)=>a.seed-b.seed);
    panel(s,names[m],x+8);
    const maxEpoch=Math.max(...rows.map(r=>r.epochs_completed));
    const sets=rows.map((r,k)=>series(`seed ${r.seed}`,r.history.map(h=>100*h.metric),[C.blue,C.gold,C.green][k],
      {xValues:r.history.map(h=>h.epoch),marker:{symbol:'none'}}));
    addChart(s,'scatter',[],sets,{left:x,top:242,width:550,height:310},
      {scatterOptions:{style:'line'},xAxis:{min:0,max:Math.ceil(maxEpoch/5)*5,majorUnit:maxEpoch>30?20:5,
         numberFormatCode:'0',textStyle:{fontSize:20,fill:C.muted},majorGridlines:null},
       yAxis:{min:0,max:ds==='banking77'?100:55,majorUnit:ds==='banking77'?25:10,numberFormatCode:'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:ds==='go_emotions'?'V Macro-AP（0–100）':'V Macro-F1（0–100）',textStyle:{fontSize:20,fill:C.muted}}},displayBlanksAs:'gap'});
    text(s,`共训练 ${Math.min(...rows.map(r=>r.epochs_completed))}–${maxEpoch} 轮`,x+22,554,520,32,24,true);
  });
  note(s,'横轴为训练轮次；均满足既定早停条件，未按Q逐轮选模型');return s;
}
function transition(datasets,title,topic){
  const keys=['always_correct','corrected','persistent_error','new_error'];
  const labels=['持续正确','由错变对','持续错误','新增错误'];
  const colors=[C.gray,C.green,C.red,C.gold];
  const s=slide(title,'S2探索D；相同文本上的1k → 6k模型转移',
    '旧固定预算检查点。持续错误不等于优化后不可解。GoEmotions按0阈值完整标签集合判断整条对错，不能跨任务将错误率排序。\n'+JSON.stringify(trajectories.filter(r=>datasets.includes(r.dataset))),topic);
  datasets.forEach((ds,i)=>{
    const x=64+i*600,rows=models.map(m=>trajectories.find(r=>r.dataset===ds&&r.model===m));
    panel(s,`${dsNames[ds]}  n=${rows[0].n}`,x+8);
    const sets=keys.map((k,j)=>series(labels[j],rows.map(r=>100*r[k]/r.n),colors[j],{
      dataLabels:{showValue:true,position:'center',numberFormatCode:'0"%"',textStyle:{fontSize:21,fill:j===1||j===2?C.white:C.ink}},
      dataLabelOverrides:rows.map((r,idx)=>({idx,showValue:100*r[k]/r.n>=8}))}));
    addChart(s,'bar',models.map(m=>names[m]),sets,{left:x,top:245,width:550,height:330},
      {barOptions:{direction:'bar',grouping:'stacked',overlap:100,gapWidth:80},
       xAxis:{visible:true,textStyle:{fontSize:23,fill:C.ink},line:{fill:'none',width:0}},yAxis:{min:0,max:100,majorUnit:25,numberFormatCode:'0"%"',textStyle:{fontSize:20,fill:C.muted},majorGridlines:null}});
  });
  note(s,datasets.includes('go_emotions')?'GoEmotions采用完整标签集合判定；各任务D样本与判断口径分别解释':'BANKING77仅23条D样本；展示错误转移，不推断总体机制发生率');return s;
}

// 01: handoff continuity, rendered as an editable argument rather than a dense matrix.
{
 const s=slide('模型比较留下了需要解释的问题','原项目的发现，形成当前研究的起点',
  '原HANDOFF与最终报告按原交接清单校验。原常规任务、MultiNLI、AG News长尾比较是继承证据，当前BANKING77机制不直接解释MultiNLI分差。旧CPU4batch1计时与新CPU8批预测不可比较加速比。','研究脉络');
 const cols=[66,470,874];
 ['原项目发现','当前追问','现有解释证据'].forEach((t,i)=>text(s,t,cols[i],220,330,48,32,true,[C.blue,C.gold,C.green][i]));
 const bodies=['任务之间，模型差距不同\n长尾训练，尾类明显退化\n原文高分，仍有具体错误',
   '数据和优化怎样改变优势？\n预训练和适配贡献多少？\n缺标注与贡献偏置如何区分？',
   '规模与优化Q对照\nM2初始化与适配消融\nS3支持分离与S4词汇干预'];
 bodies.forEach((t,i)=>text(s,t,cols[i],310,350,205,26,false,C.ink));
 note(s,'核心问题：文本分类模型在不同数据条件下，能力边界和失败机制如何变化');
}
// 02: optimized primary Q, with large independent panels.
{
 const s=slide('优化后，DistilBERT在两组Q条件下领先','神经模型为3个训练种子均值；SVM为确定性单次',
   'BANKING77训练1k、Q3079；GoEmotions训练6k、Q5401。配方由V选择。神经样本SD与逐种子数值：\n'+JSON.stringify(primary),'正式Q比较');
 ['banking77','go_emotions'].forEach((ds,i)=>{
   const x=64+i*600;panel(s,`${dsNames[ds]}  ${ds==='banking77'?'1k / F1':'6k / AP'}`,x+8);
   const order=['bilstm','tfidf_svm','distilbert'];
   const values=order.map(m=>primary.find(r=>r.dataset===ds&&r.model===m).mean*100);
   const set=series('Q主指标',values,C.blue,{points:order.map((m,idx)=>({idx,fill:mc[m]}))});
   addChart(s,'bar',order.map(m=>names[m]),[set],{left:x,top:250,width:550,height:325},
     {hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:65},
      xAxis:{visible:true,textStyle:{fontSize:24,fill:C.ink},line:{fill:'none',width:0}},yAxis:{min:0,max:100,majorUnit:25,majorGridlines:null,textStyle:{fontSize:20,fill:C.muted}},
      dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:29,bold:true,fill:C.ink}}});
 });
 note(s,'指标按0–100展示；两组条件均为DistilBERT > SVM > BiLSTM，SD见备注');
}
twoScale(['banking77','go_emotions'],'数据规模改变了模型之间的差距','规模对照');
sensitivity('distilbert','延长预算的收益取决于任务和学习率','S1优化敏感性');
learning('banking77','最终BANKING77模型已触发动态早停','训练曲线');
// 06: pretrained/adapted factors.
{
 const s=slide('预训练与任务适配共同形成优势','M2同架构四臂；BANKING77 1k，V Macro-F1（0–100）',
   '固定分类头初始化和编码器学习率2e-5，3数据抽样×3训练种子，每臂9配置。下述数值来自V，不是42Q的一部分。SD和配对配置：\n'+JSON.stringify(m2),'预训练与适配');
 const arms=['random_frozen','pretrained_frozen','random_unfrozen','pretrained_unfrozen'];
 const vals=arms.map(a=>m2.filter(r=>r.arm===a).reduce((sum,r)=>sum+r.macro_f1,0)/9*100);
 col(s,['随机初始化\n冻结编码器','预训练初始化\n冻结编码器','随机初始化\n解冻编码器','预训练初始化\n解冻编码器'],
   [series('V Macro-F1',vals,C.blue,{points:[C.gray,C.blue,C.gold,C.green].map((fill,idx)=>({idx,fill}))})],
   {left:64,top:238,width:860,height:350},{hasLegend:false,yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:20,fill:C.muted},majorGridlines:{fill:C.grid,width:1}}});
 headline(s,'+50.08','解冻时，预训练的V F1收益（点）',960,258);
 note(s,'每臂9配置；相同配对下，预训练和任务适配两个方向均为9/9提升');
}
// 07: task-dependent deterioration.
{
 const s=slide('长尾鲁棒性的排序会随任务变化','E3旧固定预算V；相对各模型自身均衡基线的F1降幅',
  'Yelp/Yahoo总量6000，BANKING77总量781。BANKING77 BiLSTM均衡F1仅.0132，小降幅不等于稳健。原固定预算包含优化因素。\n'+JSON.stringify(imbalance),'分布倾斜');
 col(s,['Yelp','BANKING77','Yahoo'],models.map(m=>series(names[m],['yelp','banking77','yahoo_answers_topics'].map(ds=>imbalance.find(r=>r.dataset===ds&&r.model===m).drop_pp),mc[m])),
   {left:66,top:225,width:1140,height:365},{yAxis:{min:0,max:30,majorUnit:10,title:{text:'Macro-F1降幅（点）',textStyle:{fontSize:21,fill:C.muted}},textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1}}});
 note(s,'BANKING77的BiLSTM基线很低；不能把小降幅解释成更强的绝对能力');
}
// 08: separate information support from contribution.
{
 const rows=s3.filter(r=>r.model==='fixed_tfidf_svm');
 const s=slide('均匀贡献不能补回缺失的尾类标注','S3固定TF-IDF + SVM；BANKING77独立Q',
  '支持完整781、删减555，总权重固定781，尾类事先定义，单训练seed。绝对架构分数不与冻结头比较。\n'+JSON.stringify(rows),'S3独立标注支持');
 col(s,['支持完整\n781条','尾部删减\n555条'],['uniform','skew'].map((risk,i)=>series(i===0?'贡献均匀':'贡献偏斜',['full','tail_reduced'].map(support=>100*rows.find(r=>r.support===support&&r.risk===risk).macro_f1),[C.blue,C.gray][i])),
  {left:64,top:240,width:855,height:345},{yAxis:{min:0,max:80,majorUnit:20,textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:'Q Macro-F1（0–100）',textStyle:{fontSize:20,fill:C.muted}}}});
 headline(s,'−20.10','贡献均匀时，删减标注导致的Q F1损失（点）',960,260,C.red);
 note(s,'尾类召回同时从65.6%降至10.4%；加权不能替代本组缺失的独立监督信息');
}
// 09: expose a hidden tail failure alongside the mean score.
{
 const rows=s3.filter(r=>r.model==='frozen_generic_distilbert_head');
 const s=slide('整体分数提高，尾类仍可能继续恶化','S3冻结编码器 + 分类头；两种支持条件下的贡献对照',
  '固定245次更新；尾类集合事先定义，各格一个seed。支持删减下均匀→偏斜整体F1 .04361→.10334，尾Recall .048→.004。\n'+JSON.stringify(rows),'S3局部失败');
 ['macro_f1','tail_recall'].forEach((key,i)=>{
   const x=64+i*600;panel(s,i===0?'整体Q F1（0–100）':'尾类Q召回（%）',x+8);
   col(s,['支持完整','尾部删减'],['uniform','skew'].map((risk,j)=>series(j===0?'贡献均匀':'贡献偏斜',['full','tail_reduced'].map(support=>100*rows.find(r=>r.support===support&&r.risk===risk)[key]),[C.blue,C.gold][j])),
     {left:x,top:245,width:550,height:330},{yAxis:{min:0,max:i===0?16:28,numberFormatCode:'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:22,fill:C.ink,bold:true}}});
 });
 note(s,'删减支持时：整体F1 4.36 → 10.33；尾类召回4.8% → 0.4%');
}
transition(['banking77','yahoo_answers_topics'],'增加数据既能纠错，也会产生新错误','S2错误轨迹');
// 11: controlled lexical perturbation, paired with equal-budget controls.
{
 const s=slide('原文高分不保证词汇表达稳定','S4配对干预；BANKING77两个固定6k模型，同37例',
  '37对为Gemini两个表格版本全同意后的AI辅助审核子集，不是双真人独立复核。不估计总体扰动发生率。原文、控制和线索准确率及逐例来源：\n'+JSON.stringify(s4),'S4词汇干预');
 col(s,['原文','控制替换','线索替换'],s4.map(r=>series(names[r.model],['original','control','cue'].map(k=>100*r[k]),mc[r.model])),
   {left:64,top:242,width:850,height:350},{yAxis:{min:0,max:110,majorUnit:25,numberFormatCode:'0"%"',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0"%"',textStyle:{fontSize:22,bold:true,fill:C.ink}}});
 headline(s,'24.32','SVM相对控制的损失（点）',950,248,C.blue);
 text(s,'DistilBERT损失 8.11点',950,470,265,80,25,true,C.green);
 note(s,'37对AI审核子集；原文表现与表达稳定性是两个需要分别观察的维度');
}
// 12: qualitative comparison is an editable evidence table, not a text screenshot.
{
 const s=slide('预训练模型相对稳定，也存在共同失败','两个真实配对案例：控制替换均保持正确',
  'M1-013：transaction→transfer使SVM转为pending_transfer，DistilBERT保持正确；did→made控制都正确。M1-019：refund→reimbursement两模型均错，showing→appearing控制都正确。案例只解释形式，不估计发生率。\n'+JSON.stringify(cases),'词汇失败案例');
 text(s,'转账表达',66,220,520,40,28,true);text(s,'退款表达',680,220,520,40,28,true);
 text(s,'transaction → transfer',66,282,550,50,31,true,C.blue);text(s,'refund → reimbursement',680,282,550,50,31,true,C.blue);
 text(s,'SVM：错误\nDistilBERT：正确',66,370,510,130,31,false);
 text(s,'SVM：错误\nDistilBERT：错误',680,370,510,130,31,false);
 text(s,'收款人未收到转账',66,526,520,40,24,false,C.muted);text(s,'退款未到账',680,526,520,40,24,false,C.muted);
 note(s,'M1-013、M1-019；来自37对AI审核子集，不能推广为全部错误的发生比例');
}
// 13: show exact time rather than visually suppressing small times on a linear axis.
{
 const median=values=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
 const s=slide('性能提升需要更高的训练与预测成本','后续优化轮最终模型；CPU 8线程，各模型中位数',
  '训练流程含特征/验证/保存，不含预训练与全网格搜索；BANKING77 DistilBERT seed43含续跑重建。预测不含加载/校验/清理，是批次均摊，不是在线单请求延迟。共享CPU非通用硬件基准。\n'+JSON.stringify(cost),'计算成本');
 const value=(ds,m,key)=>median(cost.filter(r=>r.dataset_id===ds&&r.model_id===m).map(r=>r[key]));
 const fmt=(v,d=1)=>v<.01?v.toFixed(4):v<1?v.toFixed(3):v.toFixed(d);
 const values=[['模型','Bank训练秒','Go训练秒','Bank预测ms/条','Go预测ms/条'],...models.map(m=>[names[m],fmt(value('banking77',m,'training_pipeline_wall_seconds')),fmt(value('go_emotions',m,'training_pipeline_wall_seconds')),fmt(value('banking77',m,'milliseconds_per_row'),2),fmt(value('go_emotions',m,'milliseconds_per_row'),2)])];
 const table=s.tables.add({rows:4,columns:5,left:66*SCALE,top:244*SCALE,width:1148*SCALE,height:300*SCALE,values,columnWidths:[235,200,200,256,257].map(v=>v*SCALE)});
 table.borders.assign({fill:C.grid,width:SCALE});
 for(let r=0;r<4;r++)for(let c=0;c<5;c++){
   const cell=table.getCell(r,c);cell.fill=r===0?C.ink:C.white;
   cell.text.style={fontFamily:FONT,typeface:FONT,fontSize:(r===0?23:27)*SCALE,bold:r===0||c===0,color:r===0?C.white:(c===0?mc[models[r-1]]:C.ink)};
 }
 tableOwners.push(p.slides.items.length);
 note(s,'Bank训练1k，Go训练6k；均摊预测时间不等于在线单请求延迟');
}
twoScale(['yelp','yahoo_answers_topics'],'补充：规模收益同样具有任务差异','补充规模对照');
sensitivity('bilstm','补充：BiLSTM对预算与学习率也敏感','补充S1对照');
learning('go_emotions','补充：GoEmotions配方同样满足早停','补充训练曲线');
transition(['go_emotions','yelp'],'补充：错误迁移需要按任务分别解释','补充S2轨迹');
// 18: retain the historical fixed-budget Q references without mixing recipe changes.
{
 const s=slide('补充：旧配方低分包含明显优化因素','旧固定预算seed42，对照V选定优化配方的三种子均值',
  '旧/新同时改变学习率、预算与调度。这里是固定配方Q参考对照，不把差距全部归于epoch或early stopping。\n'+JSON.stringify({refs,primary}),'历史Q参考');
 ['banking77','go_emotions'].forEach((ds,i)=>{
   const x=64+i*600;panel(s,dsNames[ds],x+8);
   col(s,['BiLSTM','DistilBERT'],[
    series('旧固定预算',['bilstm','distilbert'].map(m=>100*refs.find(r=>r.dataset===ds&&r.model===m).score),C.gray),
    series('优化配方',['bilstm','distilbert'].map(m=>100*primary.find(r=>r.dataset===ds&&r.model===m).mean),C.green)],
    {left:x,top:244,width:550,height:340},{yAxis:{min:0,max:ds==='banking77'?100:55,numberFormatCode:'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:ds==='banking77'?'Q Macro-F1（0–100）':'Q Macro-AP（0–100）',textStyle:{fontSize:20,fill:C.muted}}}});
 });
 note(s,'新旧差异包含多项配方变化，不能只归因于训练轮数');
}

assert.equal(p.slides.items.length,18);
const preview=process.argv.includes('--preview');
for(const i of preview?[0,1,3,4,5]:p.slides.items.map((_,i)=>i)){
  const image=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(images,`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
}
await fs.writeFile(path.join(build,'slide-data.json'),JSON.stringify({source:atlasPath,atlas_sha256:digest(await fs.readFile(atlasPath)),source_sha256:atlas.source_sha256,slides:records},null,2));
if(preview){console.log(JSON.stringify({preview:[1,2,4,5,6],slides:18}));process.exit(0);}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const output=finalOutput;
const finalization=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath:output,
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  explicitTotalSlideCount:18,requiredNativeChartOwnerSlides:chartOwners,requiredNativeTableOwnerSlides:tableOwners,
  layoutArgs:['--expected-slide-size-emu','18288000,10287000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({output,slides:18,chartOwners,tableOwners,sha256:finalization.finalSha256}));

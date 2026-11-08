import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=Presentation.create({slideSize:{width:1280,height:720}});const s=p.slides.add();
const table=s.tables.add({rows:2,columns:2,left:50,top:50,width:500,height:160,columnWidths:[250,250],values:[['Dataset','Train'],['IMDb','6000']]});
for(let r=0;r<2;r++)for(let c=0;c<2;c++)table.getCell(r,c).text.style={fontFamily:'Arial',fontSize:23,color:'#173c35'};
s.charts.add('bar',{position:{left:50,top:250,width:1100,height:400},title:'Planned training samples',categories:['IMDb','RT','Yelp'],series:[{name:'Training examples',values:[6000,6000,6000],fill:'#1b766e'}],hasLegend:false,xAxis:{title:'Dataset'},yAxis:{title:'Examples (count)'}});
const img=await p.export({slide:s,format:'png',scale:1});await fs.writeFile('work/native-evidence-check.png',new Uint8Array(await img.arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save('work/native-evidence-check.pptx');console.log('Editable chart/table render and export succeeded.');

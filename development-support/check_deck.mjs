import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=Presentation.create({slideSize:{width:1280,height:720}});const s=p.slides.add();
s.background.fill='#f7f6f0';const t=s.shapes.add({geometry:'textbox',position:{left:70,top:65,width:1100,height:100},fill:'none',line:{fill:'none',width:0}});t.text='Sentiment Classification';t.text.style={fontFamily:'Arial',fontSize:46,color:'#173c35'};
const blob=await p.export({slide:s,format:'png',scale:1});await fs.writeFile('work/deck-runtime-check.png',new Uint8Array(await blob.arrayBuffer()));console.log('Presentation rendering works.');

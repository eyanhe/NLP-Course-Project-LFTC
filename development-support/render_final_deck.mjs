import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from 'file:///C:/Users/LUZIJ/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const final='outputs/sentiment-project/report/project-presentation.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<p.slides.items.length;i++){
 const img=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`work/deck-build/final-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await img.arrayBuffer()));
}
console.log('Rendered all 13 final imported slides.');

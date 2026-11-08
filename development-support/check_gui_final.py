import json,sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]/'outputs'/'sentiment-project'
sys.path.insert(0,str(root))
records=[]
a=AppTest.from_file(str(root/'app.py')).run(timeout=90)
records.append({'stage':'initial','errors':[x.message for x in a.exception],'tabs':[x.label for x in a.tabs]})
for task in ['Multiclass news','Natural language inference','Binary sentiment']:
    a.radio[0].set_value(task).run(timeout=90)
    records.append({'stage':task,'errors':[x.message for x in a.exception]})
if (root/'results-imbalance'/'evaluation.json').exists():
    for c in ['mild','moderate','severe']:
        a.selectbox(key=None) if False else None
        box=next(x for x in a.selectbox if x.label=='Training imbalance')
        box.set_value(c).run(timeout=90)
        records.append({'stage':'imbalance-'+c,'errors':[x.message for x in a.exception]})
    box=next(x for x in a.selectbox if x.label=='Deterioration metric');box.set_value('minority_recall').run(timeout=90)
    records.append({'stage':'minority-recall-chart','errors':[x.message for x in a.exception]})
Path(__file__).with_name('gui-final-check.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
if any(r['errors'] for r in records):raise RuntimeError(records)

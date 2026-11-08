"""Read-only research explorer; model inference remains in the original app."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = 'research-v3/reports/overnight/analysis-v2/analysis-data.json'
INVENTORY = 'research-v3/reports/overnight/convergence-v1/experiment-inventory-current.json'
DATASETS = {'banking77': 'BANKING77 · intent classification', 'go_emotions': 'GoEmotions · multilabel emotion'}
MODELS = {'tfidf_svm': 'TF-IDF + SVM', 'bilstm': 'BiLSTM', 'distilbert': 'DistilBERT'}
METRICS = {'macro_f1': 'Macro-F1', 'macro_ap': 'Macro-AP'}
DOWNLOADS = [
    ('synthesis', 'Research synthesis', 'final-synthesis-zh.md', 'text/markdown'),
    ('report', 'Full analysis report', 'final-report-zh.md', 'text/markdown'),
    ('atlas', 'Complete research atlas · PDF', 'figures/完整研究图册.pdf', 'application/pdf'),
    ('logic', 'Research questions and experiment logic · PDF', 'figures/当前研究问题与实验逻辑链.pdf', 'application/pdf'),
    ('q_figures', 'Three-page Q summary · PDF', 'figures/Q测试与机制对照分析.pdf', 'application/pdf'),
    ('slides', 'Presentation · PowerPoint', 'final-presentation.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
    ('analysis_table', 'Quantitative comparisons · Excel', 'tables/研究问题定量分析.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    ('inventory_table', 'Experiment inventory · Excel', 'tables/累计训练验证推理清单.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    ('timing_table', 'Training and inference timing · Excel', 'tables/训练与推理耗时记录.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
]


def load_research(root=ROOT):
    root = Path(root)
    data = json.loads((root / ANALYSIS).read_text(encoding='utf-8'))
    inventory = json.loads((root / INVENTORY).read_text(encoding='utf-8'))
    data['counts'] = dict(complete=inventory['complete_total'], current=inventory['current_research_complete'], with_v=inventory['current_research_with_V'], q=inventory['formal_test_complete'])
    ids = [row['run_id'] for row in data['q_records']]
    if len(ids) != data['counts']['q'] or len(set(ids)) != len(ids):
        raise ValueError('The Q results and experiment inventory do not describe the same unique-model scope.')
    return data


def select_q(records, dataset, stage=None, models=None):
    return [row for row in records if row['dataset'] == dataset and (stage is None or row['stage'] == stage) and (models is None or row['model'] in models)]


def download_inventory(root=ROOT):
    items = []
    for key, label, relative, mime in DOWNLOADS:
        path = Path(root) / 'report/research-v3' / relative
        available = path.is_file()
        items.append(dict(key=key, label=label, path=path, mime=mime, available=available, bytes=path.stat().st_size if available else None))
    return items


def render(root=ROOT, legacy_navigation=False):
    import pandas as pd
    import streamlit as st

    root = Path(root)
    st.set_page_config(page_title='LFTC · Current Research', layout='wide')
    st.title('Limits and failures of text classification')
    st.caption('Explore model boundaries, optimization and failure mechanisms · frozen research snapshot, 8 November 2026')
    if legacy_navigation:
        st.page_link('app.py', label='Open the original live prediction demo')
    else:
        st.link_button('Prediction demo & setup guide', 'https://github.com/eyanhe/NLP-Course-Project-LFTC#使用与复现')
    try:
        data = load_research(root)
    except (OSError, ValueError, KeyError) as error:
        st.error('Research results are unavailable or inconsistent. Restore the complete GitHub snapshot to view this page.')
        st.caption(str(error))
        return
    columns = st.columns(4)
    for column, label, value in zip(columns, ['Current research records', 'With validation metrics', 'Fixed models evaluated on Q', 'All completed v3 records'], [data['counts']['current'], data['counts']['with_v'], data['counts']['q'], data['counts']['complete']]):
        column.metric(label, value)
    st.write('When do model advantages change, and what explains their failures? Explore the evidence below: model comparisons, optimization, pretraining, label support and lexical interventions.')
    st.info('V selects recipes; D supports exploration; Q evaluates frozen models. The 232 current records include 184 initial runs and 48 later runs. The original 2,859-item matrix was not fully executed.')
    dataset = st.selectbox('Dataset for Q results', list(DATASETS), format_func=DATASETS.get, key='research_dataset')
    q_tab, mechanism_tab, case_tab, figure_tab, download_tab = st.tabs(['Formal Q results', 'Mechanism comparisons', 'Error cases', 'Research figures', 'Reports & downloads'])
    with q_tab:
        st.subheader('Optimized primary models')
        primary = pd.DataFrame([row for row in data['primary'] if row['dataset'] == dataset])
        metric = primary['metric'].iloc[0]
        summary = primary[['model', 'mean', 'sd', 'n']].rename(columns={'model':'Model', 'mean':METRICS[metric], 'sd':'Seed SD', 'n':'Training runs'})
        summary['Model'] = summary['Model'].map(MODELS).fillna(summary['Model'])
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.bar_chart(summary.set_index('Model')[METRICS[metric]])
        st.caption('Only the designated primary recipes are compared here. Neural models average seeds 42/43/44; SVM is one deterministic fit. Seed SD is not a test confidence interval. Macro-F1 and Macro-AP are separate metrics.')
        st.subheader('Inspect fixed-model results')
        dataset_rows = select_q(data['q_records'], dataset)
        stages = list(dict.fromkeys(row['stage'] for row in dataset_rows))
        stage = st.selectbox('Experiment stage', ['All stages'] + stages, key='q_stage')
        choices = list(dict.fromkeys(row['model'] for row in dataset_rows))
        selected = st.multiselect('Models', choices, default=choices, format_func=lambda value:MODELS.get(value, value), key='q_models')
        rows = select_q(data['q_records'], dataset, None if stage == 'All stages' else stage, selected)
        st.caption(f'{len(rows)} matching fixed-model records. Results across different recipes are shown individually, not pooled into a model ranking.')
        st.caption('Timing: milliseconds_per_row is saved batch prediction time divided by the number of test rows. It is not online single-request latency.')
        if rows:
            table = pd.DataFrame(rows)
            columns = ['run_id','dataset','stage','model','role','seed','train_rows','lr','C','budget','metric','score','test_rows','support','risk','tail_f1','tail_recall','milliseconds_per_row']
            st.dataframe(table[columns], hide_index=True, use_container_width=True)
            st.download_button('Download filtered Q results · CSV', table.to_csv(index=False).encode('utf-8-sig'), file_name=f'{dataset}-Q-results.csv', mime='text/csv', key='filtered_q_csv')
        else:
            st.info('No fixed models match the current selection. Choose at least one model or change the experiment stage.')
    with mechanism_tab:
        st.subheader('What changes the outcome?')
        st.write('These are saved, context-specific contrasts. Read the metric and experimental scope before comparing differences.')
        sections = list(dict.fromkeys(row['section'] for row in data['contrasts']))
        section = st.selectbox('Comparison family', sections, key='mechanism_family')
        if section.startswith('M2'):
            st.caption('Scope: BANKING77, 1,000 training examples, validation Macro-F1. M2 is validation evidence.')
        elif section.startswith('S3'):
            st.caption('Scope: BANKING77 label-support and class-contribution experiments. Each context identifies its V or Q comparison.')
        contrasts = pd.DataFrame([row for row in data['contrasts'] if row['section'] == section])
        st.dataframe(contrasts[['context','metric','before','after','difference_pp','meaning']], hide_index=True, use_container_width=True)
        st.caption('difference_pp = (after − before) × 100. The comparison context may concern either dataset. V contrasts are exploratory; formal Q contrasts keep their frozen scope.')
        with st.expander('Pretraining and adaptation · validation evidence'):
            st.dataframe(pd.DataFrame(data['m2_summary']), hide_index=True, use_container_width=True)
            st.caption('BANKING77 V Macro-F1. Each arm contains three data draws × three training seeds (n=9); SD includes both sources of variation. These are not additional Q evaluations.')
    with case_tab:
        st.subheader('Lexical sensitivity in individual examples')
        st.caption('BANKING77 discovery examples; original, cue intervention and matched control. These are saved predictions, not new inference. The 37 reviewed pairs used Gemini-assisted review, not two independent human reviewers.')
        cases = data.get('cases', [])
        if cases:
            case_id = st.selectbox('Case', [case['case_id'] for case in cases], key='research_case')
            case = next(item for item in cases if item['case_id'] == case_id)
            edits = case['text_edits']
            for label, field in [('Original','original_text'), ('Cue intervention','intervention_text'), ('Matched control','control_text')]:
                st.markdown(f'**{label}**')
                st.write(edits[field])
            predictions = []
            for model, result in case['predictions'].items():
                for arm in ('original','cue','control'):
                    predictions.append({'Model':MODELS.get(model,model), 'Arm':arm, 'Gold':result['gold_name'], 'Prediction':result[arm]['prediction_name'], 'Correct':result[arm]['correct']})
            st.dataframe(pd.DataFrame(predictions), hide_index=True, use_container_width=True)
    with figure_tab:
        st.subheader('From the research question to the evidence')
        folder = root / 'research-v3/reports/overnight/complete-atlas-v1'
        figures = sorted(folder.glob('*.png'))
        if figures:
            figure = st.selectbox('Figure', [path.name for path in figures], key='research_figure')
            st.image(str(folder / figure), use_container_width=True)
            st.caption('Original figure from the verified atlas: one continuity page and eleven result figures. Figures preserve the reported experiment scope.')
        else:
            st.warning('Figure files are missing from this project snapshot.')
    with download_tab:
        st.subheader('Take the evidence with you')
        st.write('Reports, atlas, slides and workbooks are the same saved deliverables included in the repository.')
        for item in download_inventory(root):
            if item['available']:
                st.download_button(item['label'], item['path'].read_bytes(), file_name=item['path'].name, mime=item['mime'], key='artifact_'+item['key'])
            else:
                st.warning(item['label'] + ' is missing from this project snapshot.')
        st.link_button('Code, reports & prediction results · GitHub', 'https://github.com/eyanhe/NLP-Course-Project-LFTC')
        st.link_button('Datasets & checkpoints · Hugging Face', 'https://huggingface.co/datasets/Eyanhe/LFTC_Dataset_And_Checkpoints')
    st.divider()
    st.caption('The viewer reads saved evidence and never trains or re-evaluates models. The original prediction demo remains available through app.py with its model dependencies and checkpoints.')

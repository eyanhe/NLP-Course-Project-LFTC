"""Interactive companion to the Group 19 course presentation."""
import html
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lftc.reporting.presentation import (
    MODELS, MODEL_NAMES, DATASETS, COLORS, METRICS, CONDITIONS,
    load_presentation_data, model_summary, cost_summary, always_world,
)
from lftc.training.benchmarks import load_benchmark_model, predict_benchmarks

ROOT = Path(os.environ.get('LFTC_ROOT', Path(__file__).resolve().parents[1])).resolve()
ARCHIVE = ROOT / 'archive/2026-11-08'
APP_DIR = Path(__file__).resolve().parent
DECK = APP_DIR / 'assets/research-presentation.pptx'
PAGES = ['Overview', 'Performance & cost', 'Robustness', 'Mechanisms',
         'Live prediction', 'Methods & evidence', 'Conclusions']


@st.cache_data(show_spinner=False)
def evidence(root):
    return load_presentation_data(root)


@st.cache_resource(show_spinner=False)
def _saved_model(root, source, name):
    return load_benchmark_model(source, name, root=root)


def cached_model(source, name):
    return _saved_model(str(ROOT), source, name)


def run_prediction(source, text, premise='', models=None, loader=None):
    rows = predict_benchmarks(source, text, premise=premise, models=models, root=ROOT, loader=loader)
    displayed = []
    for row in rows:
        value = dict(row)
        if isinstance(value.get('score'), (int, float)):
            probability = 'probability' in str(value.get('score_label', '')).lower()
            value['score'] = f"{value['score']:.1%}" if probability else f"{value['score']:+.3f}"
            value['score_label'] = 'Uncalibrated probability' if probability else 'Decision margin'
        displayed.append(value)
    return displayed


def go(page, source=None):
    if source is not None:
        st.session_state['demo_source'] = source
    st.session_state['presentation_section'] = page

def heading(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div><div class="page-title">{title}</div><div class="lead">{description}</div>', unsafe_allow_html=True)

def takeaway(text):
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)

def scope(text):
    st.markdown(f'<div class="scope">{text}</div>', unsafe_allow_html=True)

def stats(values):
    for col, (value, label) in zip(st.columns(len(values)), values):
        col.markdown(f'<div class="stat"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

def chart_style(chart):
    return chart.configure_view(strokeOpacity=0).configure_axis(labelColor='#69717B', titleColor='#69717B', gridColor='#E3E3DC', labelFont='Segoe UI', titleFont='Segoe UI', domain=False, tickColor='#D7D8D2').configure_legend(labelColor='#4D555F', title=None).properties(background='#f7f5ef')

def score_chart(frame, metric='macro_f1', height=320):
    table = frame.copy()
    table['Model'] = table.model.map(MODEL_NAMES).fillna(table.model)
    order = [MODEL_NAMES.get(x, x) for x in MODELS if x in table.model.values]
    bars = alt.Chart(table).mark_bar(size=27, cornerRadiusEnd=3).encode(
        y=alt.Y('Model:N', sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X(f'{metric}:Q', title=METRICS.get(metric, metric), scale=alt.Scale(domain=[-1, 1] if metric == 'mcc' else [0, 1])),
        color=alt.Color('Model:N', scale=alt.Scale(domain=order, range=[COLORS[x] for x in MODELS if x in table.model.values]), legend=None),
        tooltip=['Model:N', alt.Tooltip(f'{metric}:Q', format='.4f')])
    labels = bars.mark_text(align='left', dx=7, color='#14202E').encode(text=alt.Text(f'{metric}:Q', format='.3f'))
    st.altair_chart(chart_style((bars + labels).properties(height=height)), use_container_width=True)

def demo(data):
    heading('LIVE DEMO / SLIDE 7', 'Same text. Five perspectives.', 'Enter a review, a news story or a sentence pair. Compare the predictions of the five trained models on this machine.')
    sources = {**DATASETS, 'mnli': 'MultiNLI · sentence pairs'}
    source = st.selectbox('Task / training dataset', list(sources), format_func=sources.get, key='demo_source')
    samples = {
        'imdb': [('Mixed review', 'The plot was predictable, but the acting saved it.'), ('Positive review', 'A beautifully acted film with a moving ending.'), ('Negative review', 'The characters were flat and the story never went anywhere.')],
        'rotten_tomatoes': [('Mixed snippet', 'Uneven, but saved by its wonderful cast.'), ('Positive snippet', 'Smart, funny and surprisingly moving.'), ('Negative snippet', 'A tedious exercise with nothing new to say.')],
        'yelp': [('Mixed review', 'The food was excellent, although we waited a long time for a table.'), ('Positive review', 'Friendly staff, wonderful food and great value.'), ('Negative review', 'We waited an hour and the food arrived cold.')],
        'ag_news': [('Sports', 'The team won the championship after a dramatic final match.'), ('Technology', 'Researchers unveiled a new processor for faster artificial intelligence systems.'), ('Business', 'The company reported higher quarterly profits as sales increased.')],
        'mnli': [('Entailment', 'Someone is reading indoors.'), ('Contradiction', 'No one in the library is reading.'), ('Neutral', 'The woman is preparing for an exam.')],
    }
    def apply_example():
        selected = st.session_state.get('demo_example', 0)
        st.session_state['demo_text'] = samples[source][selected][1]
        st.session_state['demo_premise'] = 'A woman is reading a book in a quiet library.'
        st.session_state.pop('demo_result', None)
    if st.session_state.get('demo_previous_source') != source:
        st.session_state['demo_previous_source'] = source
        st.session_state['demo_example'] = 0
        apply_example()
    st.selectbox('Start with an example', list(range(len(samples[source]))), format_func=lambda x: samples[source][x][0], key='demo_example', on_change=apply_example)
    with st.form('prediction_form'):
        if source == 'mnli':
            first, second = st.columns(2)
            first.text_area('Premise', key='demo_premise', height=150, max_chars=10000)
            second.text_area('Hypothesis', key='demo_text', height=150, max_chars=10000)
        else:
            st.text_area('Your English text', key='demo_text', height=145, max_chars=10000)
        models = st.multiselect('Models to compare', MODELS, default=MODELS, format_func=MODEL_NAMES.get, key='demo_models')
        submitted = st.form_submit_button('Compare models', type='primary')
    if source == 'mnli':
        scope('Entailment: the hypothesis follows from the premise. Contradiction: the two statements are incompatible. Neutral: neither is established. Example names describe the intended relation, not a guaranteed model prediction.')
    elif source == 'ag_news':
        scope('Four news labels: World, Sports, Business and Sci/Tech. Short or ambiguous texts can receive different predictions.')
    else:
        scope('Binary reviews receive a positive or negative label, including neutral or mixed reviews.')
    scope('All demos use balanced training checkpoints; neural models use seed 42. SVM margins and neural probabilities measure different things.')
    if submitted:
        st.session_state.pop('demo_result', None)
        if not st.session_state['demo_text'].strip() or (source == 'mnli' and not st.session_state['demo_premise'].strip()):
            st.warning('Enter some English text. For sentence pairs, fill in both the premise and hypothesis.')
        elif not models:
            st.warning('Choose at least one model to compare.')
        else:
            try:
                with st.spinner('Loading local checkpoints and comparing predictions…'):
                    rows = run_prediction(source, st.session_state['demo_text'], st.session_state.get('demo_premise', ''), models, cached_model)
                st.session_state['demo_result'] = {'source': source, 'text': st.session_state['demo_text'], 'premise': st.session_state.get('demo_premise', ''), 'models': list(models), 'rows': rows}
            except (OSError, ValueError, ImportError, RuntimeError) as error:
                st.error('Prediction could not run with the local checkpoints.')
                st.caption(str(error))
    result = st.session_state.get('demo_result')
    if result and result['source'] == source:
        changed = result['text'] != st.session_state['demo_text'] or result['models'] != models or (source == 'mnli' and result['premise'] != st.session_state['demo_premise'])
        if changed:
            st.info('These predictions are from your last submitted input. Select Compare models to update them.')
        st.subheader('The models say…')
        for col, row in zip(st.columns(len(result['rows'])), result['rows']):
            col.markdown(f'<div class="prediction" style="border-color:{COLORS[row["model"]]}"><div class="name">{MODEL_NAMES[row["model"]]}</div><div class="label">{html.escape(row["prediction"])}</div><div class="detail">{row["score_label"]}: {row["score"]}<br>{row["milliseconds"]:.1f} ms · CPU</div></div>', unsafe_allow_html=True)
        if any(r['truncated'] for r in result['rows']):
            st.warning('An input was truncated: BiLSTM uses 256 words per sentence; DistilBERT uses 256 subwords in total.')
        scope('Live times exclude checkpoint loading and are illustrative, rather than a repeat of the controlled latency benchmark. Agreement is not proof of correctness.')
    else:
        st.markdown('<div class="takeaway">Try a sentence with mixed sentiment, or change the hypothesis while keeping the premise fixed. The interesting cases are where the models disagree.</div>', unsafe_allow_html=True)

def performance(data):
    heading('RQ1 / SLIDES 8–9', 'The gap grows as the task gets harder.', 'Compare the five model families on the same held-out texts, then inspect what each prediction costs on a CPU.')
    a, b = st.columns([2, 1])
    ds = a.selectbox('Benchmark dataset', list(DATASETS), format_func=DATASETS.get, key='perf_dataset')
    metric = b.selectbox('Metric', ['macro_f1', 'accuracy', 'macro_precision', 'macro_recall', 'mcc'], format_func=METRICS.get, key='perf_metric')
    frame = model_summary(data, ds)
    scores = frame.set_index('model')
    best = scores.loc[MODELS[:3], 'macro_f1'].idxmax()
    gap = 100 * (scores.loc['distilbert', 'macro_f1'] - scores.loc[best, 'macro_f1'])
    stats([(f'{scores.loc["distilbert", "macro_f1"]:.3f}', 'DistilBERT · Macro-F1'), (f'{scores.loc[best, "macro_f1"]:.3f}', 'Best traditional · ' + MODEL_NAMES[best]), (f'+{gap:.1f} pp', 'Transformer advantage')])
    first, second = st.columns([1.8, 1])
    with first:
        st.subheader(METRICS[metric] + ' by model')
        score_chart(frame, metric)
    with second:
        st.subheader('Read the result')
        if ds in ['imdb', 'yelp']:
            takeaway('The best traditional model is within one Macro-F1 point of DistilBERT in this experiment. A simple baseline deserves consideration when latency matters.')
        else:
            takeaway('DistilBERT gains roughly four to five Macro-F1 points here. The size of the advantage depends on the task and the data.')
        scope('Neural results: mean of seeds 42 / 43 / 44. Traditional models: a fixed fit. These are task-specific comparisons, rather than a controlled estimate of task difficulty.')
    display = frame.rename(columns={'model': 'Model', **METRICS, 'runs': 'Runs', 'seed_sd': 'F1 seed SD'})
    display['Model'] = display['Model'].map(MODEL_NAMES)
    st.dataframe(display.style.format({c: '{:.4f}' for c in display.columns if c not in ['Model', 'Runs']}, na_rep='—'), hide_index=True, use_container_width=True)
    if ds in data['bootstrap']:
        ci = data['bootstrap'][ds]
        scope(f'Separate uncertainty check: DistilBERT seed42 minus TF-IDF + SVM, Δ Macro-F1 {ci["delta_macro_f1"]*100:+.2f} pp; conditional 95% interval [{ci["ci95_low"]*100:+.2f}, {ci["ci95_high"]*100:+.2f}] pp. This interval does not compare the three-seed mean with the best traditional model.')
    st.divider()
    st.subheader('The CPU trade-off')
    cost = cost_summary(data, ds)
    cost = cost[cost.threads == 4].sort_values('median_ms').copy()
    cost['Model'] = cost.model.map(MODEL_NAMES)
    chart = alt.Chart(cost).mark_bar(color='#2A78D6', size=25, cornerRadiusEnd=3).encode(y=alt.Y('Model:N', sort='x', title=None), x=alt.X('median_ms:Q', title='Median CPU latency (ms / text)'), tooltip=['Model', alt.Tooltip('median_ms:Q', format='.2f'), alt.Tooltip('p95_ms:Q', format='.2f')]).properties(height=220)
    st.altair_chart(chart_style(chart), use_container_width=True)
    scope('Common workload: the same 200 mixed review texts, batch 1, four CPU threads, five repeats. Includes preprocessing and prediction; excludes loading. AG News checkpoints also use this review workload. GPU and batched latency are not measured here.')
    st.download_button('Download model comparison · CSV', frame.to_csv(index=False).encode('utf-8-sig'), file_name=ds + '-comparison.csv', mime='text/csv')
    with st.expander('Inspect class errors and the confusion matrix'):
        name = st.selectbox('Inspect model', MODELS, format_func=MODEL_NAMES.get, key='perf_confusion')
        rows = [r for r in data['evaluation'] if r['source'] == ds and r['target'] == ds and r['model'] == name]
        row = next((r for r in rows if r['seed'] == 42), rows[0])
        labels = [k for k, v in row['classification_report'].items() if isinstance(v, dict) and k not in ['macro avg', 'weighted avg']]
        cm = pd.DataFrame(row['confusion_matrix'], index=labels, columns=labels)
        cm.index.name = 'True / predicted'
        st.dataframe(cm, use_container_width=True)
        scope('Rows are true classes; columns are predictions. Neural confusion counts use seed42, rather than averaged counts.')

def robustness(data):
    heading('RQ2 / SLIDES 10–11', 'Harder conditions widen the gap.', 'Sentence pairs, shifted class support and a new review corpus test different kinds of robustness.')
    tab = st.radio('Robustness experiment', ['Class imbalance', 'Sentence pairs', 'Cross-dataset transfer'], horizontal=True, key='robustness_experiment', label_visibility='collapsed')
    if tab == 'Class imbalance':
        st.subheader('What happens when one class takes over?')
        selected = st.select_slider('World share of the 6,000 training texts', options=list(CONDITIONS), value='severe', format_func=CONDITIONS.get, key='imbalance_condition')
        frame = data['imbalance']
        current = frame[frame.condition == selected].set_index('model')
        minority = {'balanced': 1500, 'mild': 600, 'moderate': 300, 'severe': 120}[selected]
        stats([(str(6000 - 3 * minority), 'World training texts'), (str(minority), 'Training texts per minority class'), ('250', 'Test texts in every class')])
        metric = st.selectbox('Robustness metric', ['macro_f1', 'minority_recall', 'accuracy'], format_func=METRICS.get, key='imbalance_metric')
        c = frame.copy()
        c['World share'] = c.condition.map({'balanced': 25, 'mild': 70, 'moderate': 85, 'severe': 94})
        c['Model'] = c.model.map(MODEL_NAMES)
        line = alt.Chart(c).mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5).encode(x=alt.X('World share:Q', title='World share of training data (%)', scale=alt.Scale(domain=[25, 94]), axis=alt.Axis(values=[25, 70, 85, 94])), y=alt.Y(metric + ':Q', title=METRICS[metric], scale=alt.Scale(domain=[0, 1])), color=alt.Color('Model:N', scale=alt.Scale(domain=[MODEL_NAMES[n] for n in MODELS], range=[COLORS[n] for n in MODELS]), legend=alt.Legend(orient='top')), tooltip=['Model', 'World share', alt.Tooltip(metric + ':Q', format='.4f')]).properties(height=320)
        st.altair_chart(chart_style(line), use_container_width=True)
        table = current.loc[MODELS, ['macro_f1', 'macro_f1_drop_pp', 'minority_recall', 'accuracy']].rename(index=MODEL_NAMES, columns={**METRICS, 'macro_f1_drop_pp': 'F1 decline (pp)'})
        st.dataframe(table.style.format('{:.4f}'), use_container_width=True)
        scope('Validation and test remain balanced. Decline is relative to each model’s own balanced baseline. Minority recall averages Sports, Business and Sci/Tech. Results concern this corpus and this choice of majority class.')
        a, b = st.columns(2)
        with a:
            st.subheader('Can class weights help?')
            mitigation = data['mitigation'].copy()
            mitigation['Model'] = mitigation.model.map(MODEL_NAMES)
            st.dataframe(mitigation[['Model', 'macro_f1', 'macro_f1_gain_pp', 'minority_recall']].rename(columns={**METRICS, 'macro_f1_gain_pp': 'F1 gain (pp)'}).style.format({'Macro-F1': '{:.3f}', 'F1 gain (pp)': '{:+.2f}', 'Minority recall': '{:.1%}'}), hide_index=True, use_container_width=True)
            scope('Predeclared weighting at 94% World. These scores remain on the fixed balanced test set.')
        with b:
            st.subheader('The accuracy paradox')
            mix = st.radio('Always-World rule evaluated on', ['Balanced test', 'Simulated 94/2/2/2 mix'], key='majority_distribution')
            rule = always_world(0.25 if mix == 'Balanced test' else 0.94)
            st.metric('Accuracy', f'{rule["accuracy"]:.0%}')
            st.metric('Macro-F1', f'{rule["macro_f1"]:.3f}')
            scope('Minority recall is 0% in both cases. The simulated mix changes evaluation priors; it is not another observed deployment test.')
    elif tab == 'Sentence pairs':
        condition = st.selectbox('Genre condition', ['mnli_matched', 'mnli_mismatched'], format_func=lambda v: 'Matched · seen genres' if v.endswith('_matched') else 'Mismatched · new genres', key='nli_condition')
        frame = model_summary(data, condition)
        score_chart(frame)
        scores = frame.set_index('model').macro_f1
        diagnostic = next(r for r in data['evaluation'] if r['target'] == condition and r['model'] == 'hypothesis_only_svm')
        a, b = st.columns(2)
        a.metric('DistilBERT − pair SVM · mean F1 gap', f'{100*(scores["distilbert"]-scores["linear_svm"]):+.1f} pp')
        b.metric('Hypothesis-only diagnostic · Accuracy', f'{diagnostic["accuracy"]:.1%}')
        ci = data['nli_bootstrap'][condition.removeprefix('mnli_')]
        scope(f'999 pairs per evaluation condition, sampled from official validation pools. Uniform guessing has expected Accuracy 33.3%. The diagnostic shows premise-free predictive cues; it does not prove every model uses them. Separate seed42 bootstrap: [{100*ci["ci95_low"]:+.1f}, {100*ci["ci95_high"]:+.1f}] pp, conditional on two trained models.')
        takeaway('The traditional baselines also receive both sentences, their absolute feature difference and their product. DistilBERT’s larger advantage here does not isolate reasoning ability or architecture alone.')
        st.button('Try your own sentence pair', on_click=go, args=('Live prediction', 'mnli'))
    else:
        name = st.selectbox('Transfer model', ['logistic_regression', 'linear_svm', 'distilbert'], format_func=MODEL_NAMES.get, key='transfer_model')
        frame = data['transfer']
        chosen = frame[frame.model == name]
        matrix = chosen.pivot_table(index='source', columns='target', values='macro_f1').reindex(index=list(DATASETS)[:3], columns=list(DATASETS)[:3])
        st.subheader('Train on one source. Test on another.')
        st.dataframe(matrix.rename(index=DATASETS, columns=DATASETS).style.format('{:.3f}').background_gradient(cmap='Blues', vmin=0.6, vmax=1), use_container_width=True)
        means = chosen.groupby(['source', 'target']).macro_f1.mean()
        losses = [means.loc[source, source] - value for (source, target), value in means.items() if source != target]
        st.metric('Average off-home Macro-F1 change', f'{-100*np.mean(losses):+.1f} pp')
        scope('Rows are training sources; columns are held-out target datasets. Average change compares each model’s off-home scores with its own home-corpus score. Different target difficulties remain part of this descriptive comparison. No target adaptation is performed.')

def mechanisms(data):
    research = data['research']
    heading('RQ3 / SLIDE 12 + SUPPORTING STUDIES', "Where does the Transformer's advantage come from?", 'Controlled BANKING77 studies examine what pretraining, label support and word choice contribute. Each experiment keeps its own evidence scope.')
    tab = st.radio('Mechanism experiment', ['Pretraining & adaptation', 'Label support', 'Word swaps', 'Optimization'], horizontal=True, key='mechanism_experiment', label_visibility='collapsed')
    if tab == 'Pretraining & adaptation':
        st.subheader('Same architecture. Four training conditions.')
        names = {'pretrained_unfrozen': 'Pretrained · fine-tuned', 'pretrained_frozen': 'Pretrained · frozen', 'random_unfrozen': 'Random · fine-tuned', 'random_frozen': 'Random · frozen'}
        frame = pd.DataFrame(research['m2_summary'])
        frame['Condition'] = frame.arm.map(names)
        frame['Type'] = frame.arm.map(lambda x: 'Pretrained' if x.startswith('pretrained') else 'Random')
        bars = alt.Chart(frame).mark_bar(size=33, cornerRadiusEnd=3).encode(y=alt.Y('Condition:N', sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=240)), x=alt.X('mean:Q', title='Validation Macro-F1', scale=alt.Scale(domain=[0, 1])), color=alt.Color('Type:N', scale=alt.Scale(domain=['Random', 'Pretrained'], range=['#A9ADB1', '#2A78D6']), legend=None), tooltip=['Condition', alt.Tooltip('mean:Q', format='.4f'), alt.Tooltip('sd:Q', format='.4f'), 'n']).properties(height=260)
        labels = bars.mark_text(align='left', dx=7, color='#152230').encode(text=alt.Text('mean:Q', format='.3f'))
        st.altair_chart(chart_style(bars + labels), use_container_width=True)
        scope('M2 · V evidence only · BANKING77, 1,000 training texts · 3 data draws × 3 training seeds per arm. The SD combines both sources of variation. This does not establish the cause of the original MultiNLI gap.')
        takeaway('Pretrained representations and downstream adaptation both contribute in this setting. Being a deep network alone does not guarantee the same result.')
        st.subheader('A separate check on frozen Q models')
        primary = pd.DataFrame([r for r in research['primary'] if r['dataset'] == 'banking77'])
        primary['Model'] = primary.model.map(MODEL_NAMES)
        st.dataframe(primary[['Model', 'mean', 'sd', 'n']].rename(columns={'mean': 'Q Macro-F1', 'sd': 'Seed SD', 'n': 'Runs'}).style.format({'Q Macro-F1': '{:.4f}', 'Seed SD': '{:.4f}'}, na_rep='—'), hide_index=True, use_container_width=True)
        scope('Q comparison: 1,000 training texts; 3,079 held-out examples. Neural models use three seeds. These optimized primary models are separate from the M2 ablation.')
    elif tab == 'Label support':
        st.subheader('Missing examples and unequal contribution are different problems.')
        frame = pd.DataFrame([r for r in research['q_records'] if r['stage'] == 'S3'])
        model = st.selectbox('Fixed representation', ['fixed_tfidf_svm', 'frozen_encoder_head'], format_func=lambda n: 'Fixed TF-IDF + SVM' if n == 'fixed_tfidf_svm' else 'Frozen encoder + classifier head', key='support_model')
        # Preserve source identifiers rather than assuming a display name is a run ID.
        if model == 'frozen_encoder_head':
            chosen = frame[frame.model != 'fixed_tfidf_svm'].copy()
        else:
            chosen = frame[frame.model == model].copy()
        chosen['Support'] = chosen.support.map({'full': 'Full · 781', 'reduced': 'Reduced · 555'})
        if chosen.Support.isna().any():
            chosen['Support'] = chosen.apply(lambda r: ('Full' if r.support == 'full' else 'Reduced') + f' · {r.train_rows}', axis=1)
        chosen['Contribution'] = chosen.risk.map({'uniform': 'Uniform', 'skew': 'Skewed'})
        st.dataframe(chosen[['Support', 'Contribution', 'macro_f1', 'tail_f1', 'tail_recall']].rename(columns={'macro_f1': 'Overall Q Macro-F1', 'tail_f1': 'Tail F1', 'tail_recall': 'Tail recall'}).style.format({'Overall Q Macro-F1': '{:.4f}', 'Tail F1': '{:.4f}', 'Tail recall': '{:.1%}'}), hide_index=True, use_container_width=True)
        takeaway('With fixed TF-IDF + SVM and uniform contribution, removing tail-class training examples cuts tail recall from 65.6% to 10.4%. Equal contribution does not replace missing independent supervision in this experiment.')
        scope('S3 · Q evidence · eight cells in total, one seed per cell. Representations are fixed within each family. The frozen-head model is not the end-to-end DistilBERT, so its absolute score is not an architecture ranking.')
        if model == 'frozen_encoder_head':
            takeaway('In the reduced-support condition, overall F1 rises from 0.0436 to 0.1033 as contribution becomes skewed, while tail recall falls from 4.8% to 0.4%. An overall gain can hide a local failure.')
    elif tab == 'Word swaps':
        st.subheader('Does a small edit change the answer?')
        frame = pd.DataFrame(research['s4'])
        frame['Model'] = frame.model.map(MODEL_NAMES)
        st.dataframe(frame[['Model', 'original', 'cue', 'control', 'drop_pp']].rename(columns={'original': 'Original accuracy', 'cue': 'Cue edit accuracy', 'control': 'Control edit accuracy', 'drop_pp': 'Cue vs control loss (pp)'}).style.format({'Original accuracy': '{:.1%}', 'Cue edit accuracy': '{:.1%}', 'Control edit accuracy': '{:.1%}', 'Cue vs control loss (pp)': '{:.2f}'}), hide_index=True, use_container_width=True)
        case_id = st.selectbox('Inspect a real example', [r['case_id'] for r in research['cases']], key='mechanism_case')
        case = next(r for r in research['cases'] if r['case_id'] == case_id)
        for col, (label, field) in zip(st.columns(3), [('Original', 'original_text'), ('Cue edit', 'intervention_text'), ('Control edit', 'control_text')]):
            col.markdown(f'<div class="example"><strong>{label}</strong><p>{html.escape(case["text_edits"][field])}</p></div>', unsafe_allow_html=True)
        preds = []
        for name, result in case['predictions'].items():
            preds.append({'Model': MODEL_NAMES[name], 'Gold label': result['gold_name'], **{arm.title(): result[arm]['prediction_name'] + (' ✓' if result[arm]['correct'] else ' ✕') for arm in ['original', 'cue', 'control']}})
        st.dataframe(pd.DataFrame(preds), hide_index=True, use_container_width=True)
        scope('S4 · selected discovery subset · 37 Gemini-assisted reviewed pairs, with matched control edits. Models were trained on 6,000 examples. These are saved predictions, not live edits or a double-human review. This sample does not measure population-wide robustness.')
    else:
        st.subheader('A low score may also be an optimization problem.')
        q = pd.DataFrame([r for r in research['q_records'] if r['stage'] == 'S1'])
        dataset = st.selectbox('Optimization dataset', ['banking77', 'go_emotions'], format_func=lambda n: 'BANKING77 · Macro-F1' if n == 'banking77' else 'GoEmotions · Macro-AP', key='optimization_dataset')
        chosen = q[q.dataset == dataset].copy()
        chosen['Model'] = chosen.model.map(MODEL_NAMES)
        chosen['Learning rate'] = chosen.lr.map(lambda v: f'{v:g}')
        st.dataframe(chosen[['Model', 'Learning rate', 'budget', 'metric', 'score']].sort_values(['Model', 'Learning rate', 'budget']).rename(columns={'budget': 'Epoch budget', 'metric': 'Q metric', 'score': 'Q score'}).style.format({'Q score': '{:.4f}'}), hide_index=True, use_container_width=True)
        takeaway('BANKING77 DistilBERT at learning rate 5e-5: increasing the budget from 5 to 10 epochs raises Q Macro-F1 from 0.3600 to 0.6689. On GoEmotions, the higher-rate Macro-AP result is essentially flat over the same budget change.')
        scope('S1 · 16 frozen Q checkpoints · seed42 only. Budget also changes the learning-rate schedule, so this is a recipe/budget contrast. The Q results are not used to select new hyperparameters.')

def methods(data):
    heading('EXPERIMENT DESIGN / SLIDES 4–7, 14', 'Know what was compared.', 'The original benchmark and the mechanism extension answer related questions under different data and evaluation protocols.')
    st.subheader('Original benchmark')
    table = pd.DataFrame([
        ['IMDb', 'Sentiment', 2, '6,000', '1,000', '1,000'], ['Rotten Tomatoes', 'Sentiment', 2, '6,000', '1,000', '1,000'], ['Yelp', 'Sentiment', 2, '6,000', '1,000', '1,000'], ['AG News', 'News topics', 4, '6,000', '1,000', '1,000'], ['MultiNLI', 'Sentence pairs', 3, '6,000', '999', '999 × 2'],
    ], columns=['Dataset', 'Task', 'Classes', 'Train', 'Validation', 'Evaluation'])
    st.dataframe(table, hide_index=True, use_container_width=True)
    takeaway('De-duplicate across splits. Fit features on training data only. Select recipes on validation Macro-F1. Freeze checkpoints before evaluation.')
    scope('The original V2 work extends previously inspected sentiment results. AG News has its own reserved evaluation sample. MultiNLI evaluation comes from official validation pools; its tuning set is drawn from the training pool. Word/token budgets and external pretraining differ between model families.')
    st.subheader('The three approaches')
    models = [('Traditional NLP', 'TF-IDF + LR / SVM, and averaged local Word2Vec + SVM. Efficient linear classifiers; mean word vectors lose word order.'), ('Sequential deep learning', 'A one-layer bidirectional LSTM initialized with training-only Word2Vec. It learns task-specific sequence features.'), ('Pretrained Transformer', 'DistilBERT with six Transformer layers and a new classification head. External pretraining is followed by task-specific fine-tuning.')]
    for col, (name, description) in zip(st.columns(3), models):
        col.markdown(f'<div class="research-question"><h3>{name}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.subheader('Different failures need different metrics')
    metrics = [
        ['Accuracy', 'Overall correctness', 'Can conceal majority-class bias'],
        ['Precision', 'Reliability of a predicted class', 'Sensitive to false positives'],
        ['Recall', 'Coverage of a true class', 'Sensitive to false negatives'],
        ['Macro-F1', 'Equal weight for every class', 'Primary single-label benchmark score'],
        ['MCC', 'Balanced association', 'Another view of class-imbalanced predictions'],
        ['CPU latency', 'Milliseconds per prediction', 'Measured at batch 1, excluding model loading'],
    ]
    st.dataframe(pd.DataFrame(metrics, columns=['Metric', 'What it measures', 'How to read it']),
                 hide_index=True, use_container_width=True)
    scope('PPT slide 6. Confidence intervals compare fixed saved models; variation across training seeds is reported separately.')
    st.subheader('Extension evidence')
    st.write('V selects and checks recipes. D supports discovery and error inspection. Q evaluates models and rules fixed in advance. The extension has 232 current research records and 42 completed Q evaluations; the larger planned matrix was not fully executed.')
    scope('No universal model ranking, isolated architecture effect, GPU latency claim or tested cascade is implied. The TF-IDF → DistilBERT cascade remains a proposal. S4 review was AI-assisted; human review and member sign-off remain separate activities.')
    st.subheader('Read or take the evidence')
    files = [('Original benchmark report', 'report/final-report.md'), ('Mechanism extension report', 'report/research-v3/final-report-zh.md'), ('Current research synthesis', 'report/research-v3/final-synthesis-zh.md')]
    for col, (label, relative) in zip(st.columns(3), files):
        path = ARCHIVE / relative
        if path.is_file():
            col.download_button(label, path.read_bytes(), file_name=path.name, mime='text/markdown', use_container_width=True)
        else:
            col.caption(label + ': local evidence not provided')
    a, b = st.columns(2)
    a.link_button('Code & saved results · GitHub', 'https://github.com/eyanhe/NLP-Course-Project-LFTC', use_container_width=True)
    b.link_button('Data & checkpoints · Hugging Face', 'https://huggingface.co/datasets/Eyanhe/LFTC_Dataset_And_Checkpoints', use_container_width=True)


def overview(data):
    st.markdown('''<div class="hero-panel">
<div class="eyebrow">EE6405 / GROUP 19 / INTERACTIVE RESEARCH PRESENTATION</div>
<div class="course-title">Traditional NLP, Deep Learning and Pretrained Transformers<br>for Text Classification</div>
<div class="hero-title">When does<br><em>pretraining pay off?</em></div>
<div class="hero-copy">A strong baseline, a network trained from scratch, and a pretrained Transformer.<br>Compare what they get right, what each prediction costs, and where they fail.</div>
<div class="approaches"><span class="approach traditional">TRADITIONAL NLP</span><span class="approach sequential">DEEP LEARNING FROM SCRATCH</span><span class="approach transformer">PRETRAINED TRANSFORMER</span></div>
<div class="hero-meta">Five models / Five datasets / Accuracy, cost and robustness / November 2026</div>
</div>''', unsafe_allow_html=True)
    with st.container(key='overview_actions'):
        first, second, _ = st.columns([1, 1, 1])
        first.button('Compare the models', type='primary', on_click=go,
                     args=('Live prediction',), use_container_width=True)
        second.button('Explore the results', on_click=go,
                      args=('Performance & cost',), use_container_width=True)
    stats([('5', 'Model families'), ('5', 'Benchmark datasets'),
           ('6,000', 'Training texts per task'), ('3', 'Research questions')])
    st.subheader('What happens, when does it change, and why?')
    questions = [
        ('RQ1 / WHAT HAPPENS?', 'Performance & cost',
         'When is a strong traditional baseline good enough, and what does extra accuracy cost?',
         'Performance & cost'),
        ('RQ2 / WHEN DOES IT CHANGE?', 'Robustness',
         'Test sentence pairs, a new review source, and training labels dominated by one class.',
         'Robustness'),
        ('RQ3 / WHY?', 'Failure mechanisms',
         'Separate pretraining, adaptation, missing labels and the effect of a single cue word.',
         'Mechanisms'),
    ]
    for column, (tag, title, text, page) in zip(st.columns(3), questions):
        with column:
            st.markdown(f'<div class="research-question"><small>{tag}</small><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
            st.button('Explore ' + tag.split(' / ')[0], key='home_' + page,
                      on_click=go, args=(page,), use_container_width=True)
    if not data['available']:
        st.info('The presentation is available, but the saved benchmark results are not included in this checkout. Add the preserved local experiment materials to explore the charts and trained models.')
        return
    st.divider()
    st.subheader('The advantage depends on the conditions')
    rows = []
    contexts = {**DATASETS, 'mnli_matched': 'MultiNLI · seen genres',
                'mnli_mismatched': 'MultiNLI · new genres'}
    for dataset, label in contexts.items():
        frame = model_summary(data, dataset).set_index('model')
        rows.append({'Condition': label,
                     'Gap': 100 * (frame.loc['distilbert', 'macro_f1'] -
                                   frame.loc[MODELS[:3], 'macro_f1'].max())})
    severe = data['imbalance']
    if not severe.empty:
        frame = severe[severe.condition == 'severe'].set_index('model')
        rows.append({'Condition': 'AG News · 94% one class',
                     'Gap': 100 * (frame.loc['distilbert', 'macro_f1'] -
                                   frame.loc[MODELS[:3], 'macro_f1'].max())})
    gap = pd.DataFrame(rows).sort_values('Gap')
    left, right = st.columns([1.6, 1])
    with left:
        bars = alt.Chart(gap).mark_bar(color='#2c7bd9', size=25).encode(
            y=alt.Y('Condition:N', sort=list(gap.Condition), title=None,
                    axis=alt.Axis(labelLimit=230)),
            x=alt.X('Gap:Q', title='DistilBERT minus best traditional · Macro-F1 percentage points'),
            tooltip=['Condition', alt.Tooltip('Gap:Q', format='+.1f')])
        labels = bars.mark_text(align='left', dx=7, color='#152230').encode(
            text=alt.Text('Gap:Q', format='+.1f'))
        st.altair_chart(chart_style((bars + labels).properties(height=300)), use_container_width=True)
    with right:
        takeaway('<strong>Long reviews:</strong> the best traditional model is within one point of DistilBERT.<br><br><strong>Sentence pairs and skewed labels:</strong> the gap is much larger.<br><br>The next question is what those extra points cost, and what explains them.')
        scope('PPT slide 8. Neural scores average three training seeds. Each condition retains its original dataset and evaluation scope.')


def conclusions(data):
    heading('CONCLUSIONS / SLIDE 13', 'Which model should you use?',
            'The answer changes with the task, the available labels and the prediction budget.')
    answers = [
        ('RQ1 / PERFORMANCE & COST', 'The gap depends on the task.',
         'Within one Macro-F1 point on long reviews; four to five points on snippets and news. TF-IDF + LR answers about 37 times faster on the measured CPU workload.'),
        ('RQ2 / ROBUSTNESS', 'Harder conditions widen the gap.',
         'About sixteen points on sentence pairs. At 94% class skew, DistilBERT loses 12.9 points against 77.8 for SVM.'),
        ('RQ3 / MECHANISMS', 'Pretraining and adaptation both matter.',
         'A deep network from scratch often trails TF-IDF. Missing class examples and single cue words explain additional failures.'),
    ]
    for column, (tag, title, text) in zip(st.columns(3), answers):
        column.markdown(f'<div class="research-question"><small>{tag}</small><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    st.subheader('Recommendations from these experiments')
    recommendations = [
        ['Long reviews, tight CPU budget', 'TF-IDF + LR', 'Within 1 pp of DistilBERT; about 37× faster'],
        ['Short snippets or news topics', 'DistilBERT', '+4–5 pp on Rotten Tomatoes and AG News'],
        ['Two-sentence inference', 'DistilBERT', 'About 16 pp ahead on MultiNLI'],
        ['Skewed or scarce labels', 'DistilBERT + class weights', 'Weighting helps, but does not replace independent examples'],
        ['A new review source', 'DistilBERT', 'Average transfer loss 5.2 pp, against about 13 pp for TF-IDF'],
        ['Huge volume, mixed difficulty', 'TF-IDF → DistilBERT cascade', 'A proposal to test; no cascade result was measured'],
    ]
    st.dataframe(pd.DataFrame(recommendations, columns=['Situation', 'Starting point', 'Evidence / scope']),
                 hide_index=True, use_container_width=True)
    takeaway('<strong>The cascade remains a proposal.</strong> Start with TF-IDF and send uncertain inputs to DistilBERT. This architecture follows from the observed trade-off; its quality, cost and routing rule still need an experiment.')
    st.subheader('What these results do not establish')
    st.write('The study uses 6,000 training texts per benchmark task. DistilBERT includes external pretraining. MultiNLI evaluation uses public validation pools, class imbalance is tested on one corpus, and the lexical cases are a selected AI-reviewed subset.')
    scope('PPT slides 13–14. Future work: learning curves, other pretrained models, a tested cascade, and independent human review of error cases.')
    st.button('Try the five models on your own text', type='primary', on_click=go,
              args=('Live prediction',))


def render():
    st.set_page_config(page_title='LFTC · When does pretraining pay off?', page_icon='◈',
                       layout='wide', initial_sidebar_state='expanded')
    st.markdown('<style>' + (APP_DIR / 'assets/presentation.css').read_text(encoding='utf-8') + '</style>',
                unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand">LFTC<span>.</span></div><div class="brand-sub">TEXT CLASSIFICATION LAB</div>', unsafe_allow_html=True)
        page = st.radio('Explore the research', PAGES, key='presentation_section', label_visibility='collapsed')
        st.markdown('<div class="sidebar-note">EE6405 / GROUP 19<br>Traditional NLP, deep learning<br>and pretrained Transformers<br><br>Presentation companion</div>', unsafe_allow_html=True)
        if DECK.is_file():
            st.download_button('Source presentation · PPTX', DECK.read_bytes(),
                               file_name='Traditional NLP, Deep Learning and Pretrained Transformers.pptx',
                               mime='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                               use_container_width=True)
    data = evidence(str(ROOT))
    source_slides = {'Overview': '01–03', 'Performance & cost': '08–09',
                     'Robustness': '10–11', 'Mechanisms': '12 + appendix',
                     'Live prediction': '07 / trained-model demo',
                     'Methods & evidence': '04–07, 14–15', 'Conclusions': '13–14'}
    st.markdown(f'<div class="source-note">GROUP 19 / RESEARCH PRESENTATION &nbsp; · &nbsp; {page.upper()} &nbsp; · &nbsp; SLIDES {source_slides[page]}</div>', unsafe_allow_html=True)
    if not data['available'] and page in {'Performance & cost', 'Robustness', 'Mechanisms'}:
        heading('SAVED EXPERIMENT EVIDENCE', page, 'These pages read the original experiment results.')
        st.info('The saved benchmark snapshot is not present here. Keep the preserved results and checkpoints with this local project to explore the presentation evidence; the source PPT is available in the sidebar.')
    elif page == 'Mechanisms' and not data['research']:
        heading('RQ3 / SLIDE 12', 'Mechanism evidence', 'The extension study is separate from the original benchmark.')
        st.info('The saved BANKING77 extension results are not included in this checkout.')
    else:
        {'Overview': overview, 'Performance & cost': performance, 'Robustness': robustness,
         'Mechanisms': mechanisms, 'Live prediction': demo, 'Methods & evidence': methods,
         'Conclusions': conclusions}[page](data)
    st.markdown('<div class="footer">EE6405 / GROUP 19 &nbsp; · &nbsp; TRADITIONAL NLP, DEEP LEARNING AND PRETRAINED TRANSFORMERS FOR TEXT CLASSIFICATION</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    render()

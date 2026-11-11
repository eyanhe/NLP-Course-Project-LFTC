# PPT-based frontend implementation

**Goal:** Replace the generic workbench with an interactive research presentation grounded in `project ppt/Traditional NLP, Deep Learning and Pretrained Transformers (v2).pptx`.

The user has requested direct implementation. Keep changes local and preserve all experiment results and weights. Do not retrain models or add asset hashing/lock machinery.

## Design

The source deck has 32 slides. Slides 1–15 are the presentation, 16–32 are supporting material. Main navigation follows the question “When does pretraining pay off?” and its three research questions:

- Overview: title, three approaches, five models, five primary datasets, three RQs.
- Performance & cost: frozen benchmark comparisons, uncertainty and CPU latency (slides 8–9).
- Robustness: MultiNLI, transfer, AG News class imbalance and weighting (slides 10–11).
- Mechanisms: pretraining/adaptation, missing-label support and lexical interventions (slide 12 and appendix).
- Live prediction: compare the five existing trained benchmark models on the same input, including premise/hypothesis inputs.
- Methods & evidence: data, protocol, model explanations, metrics, code, source PPT and report downloads (slides 4–7,14–15).
- Conclusions: task-dependent model recommendations and the explicitly untested cascade proposal (slide 13).

Use the deck's dark navy, ivory, burnt orange, blue and pink. Preserve explanatory charts and findings; do not replace them with raw experiment management tables. The 200-row quickstart is not a source for benchmark claims or the default prediction demo.

## Implementation

- [x] Add `lftc.reporting.presentation` to read original baseline, cost, transfer, imbalance and extension evidence from preserved local data. No imports of archived Python.
- [x] Add `lftc.training.benchmarks` plus minimal model compatibility classes to load the existing saved models without running old scripts. Public APIs: `load_benchmark_model(source,model_id,root=None)` and `predict_benchmarks(source,text,premise='',models=None,root=None,loader=None)`.
- [x] Rebuild `apps/streamlit_app.py` from the source deck and the former presentation page, retaining real multi-model comparison, custom layout, result charts and narrative explanations. Separate reusable presentation helpers as needed.
- [x] Copy the small source deck into `apps/assets/` for local download; leave the original PPT untouched.
- [x] Replace generic-workbench smoke tests with PPT-page, actual saved-model and result-scope checks. Verify all pages and study controls through Streamlit AppTest; inspect the corrected overview buttons with an isolated headless browser.
- [x] Update README/app instructions and source bundle. Keep GitHub unchanged.

## Boundaries

Historical results keep their original scopes: deep-model means, seed-42 bootstrap intervals, MultiNLI official validation pools, validation-only M2 and the 37 AI-reviewed lexical pairs. Proposed cascades remain proposals. Original metrics are read as data, not recomputed or overwritten.

## Local validation · 2026-11-11

- `python -B -m unittest discover -s tests -v`: 53 tests passed, including all seven pages, robustness/mechanism controls, missing-evidence states, saved-model inference and result scopes. Output: `artifacts/ppt-frontend-tests.txt`.
- Actual inference with every original task/model combination: 25/25 checkpoints loaded and predicted successfully, without training or downloads.
- Streamlit health endpoint on port 8501 returned `ok`; PowerShell launcher parsed without errors.
- Initial delivery did not include browser pixel inspection because browser control was unavailable. After the button feedback, background headless Chrome verified the corrected button rendering at 1150 px and 900 px viewport widths under a dark system theme: equal widths/heights and vertical alignment; readable normal and hover states. Screenshots and computed styles are in `.superpowers/ppt-frontend/`. This check used an independent temporary browser profile without desktop input.
- Button correction: equal-width columns, shared 56 px height, explicit light secondary-button background and inherited label colors. The four Streamlit page tests passed again after this correction.
- Source bundle: `artifacts/delivery/lftc-source.zip`. No GitHub submission or push.

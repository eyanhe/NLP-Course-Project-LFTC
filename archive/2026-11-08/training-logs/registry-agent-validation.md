# Pure registry implementation validation

Owned files: research_v3/registry.py, tests/research/test_registry.py, this note.
No existing source/configuration/lock/data/real run was edited. No training,
worker interruption, Q prediction, or real manifest generation was performed.

## API

```python
build_run_manifest(design, selections, prepared_metadata, protocol,
                   core_code_hash, diagnostics_code_hash,
                   selection_ids_hashes, *, asset_hash) -> dict
build_contrast_manifest(design, run_manifest, prepared_metadata,
                        protocol, pilot_plans) -> dict
```

prepared_metadata accepts either the datasets mapping or its actual
{'datasets': ...} envelope. selections accepts actual {'choices': ...,
'discovery_pilots': ...}. selection_ids_hashes contains each dataset's
SHA256('\n'.join(sorted(select_ids))). diagnostics_code_hash is the caller's
separate combined mechanism code identity. Core code hash remains unchanged.
pilot_plans accepts the pilot mapping or a discovery_pilots envelope; every
declared pilot needs status=complete and integer n_planned between 400 and 1200.

## TDD evidence

1. RED: `.venv-research/Scripts/python.exe -m unittest tests.research.test_registry -v`
   exited 1: 11 assertion failures, all saying the resolved run and contrast
   registry was not implemented (missing registry.py).
2. GREEN: same command exited 0: 11 tests passed in 2.971 seconds.
3. Additional RED: selected-anchor identity mismatch test exited 1,
   `AssertionError: ValueError not raised`.
4. Final GREEN: same complete targeted module command exited 0:
   14 tests passed in 3.761 seconds. Includes the statistics engine consuming
   each generated family from synthetic PredictionRows.

Interpreter execution used the required sandbox escalation. No full suite was
run because concurrent source writers affect the freeze source inventory.

## Schema and boundaries

Runs have one entry per planned physical ID, even if artifact IDs coincide;
unique_fit_count counts actual artifact IDs. Each stores the executed RunSpec,
engine, expected_model_request_hash=canonical_hash(asdict(spec)), semantic
aliases, and explicit dataset/data-seed/training-seed identities. Main specs
use the existing anchor identity calculation. Strict specs bind only fixed
defaults and empty select/discovery hashes. Nonbaseline BiLSTM mechanisms use
the diagnostics engine, while baseline M5 aliases remain core runs.

Contrasts contain F1–F6 and AUX_M2/M3/M5, plus items==contrasts for freeze
compatibility. All actual full-pool draws remain seed0 once. F3 duplicate
binary head permutations are head_seed_aliases, not independent fits. F6 uses
distinct evaluation views of one fixed checkpoint, with a common base-group
namespace. Infeasible declarations remain explicitly inactive.

These pure builders cannot authenticate saved human reviews, checkpoint bytes,
pilot responses or plan calculations. Those remain mandatory in freeze.py.
Resolved metadata therefore explicitly leaves actual_checkpoints_verified and
actual_human_pilots_verified false. Actual F6 rows must preserve matching base
sample IDs/group IDs across views; the statistics engine checks pairing later.

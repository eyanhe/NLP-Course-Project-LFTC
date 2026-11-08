# Threshold metadata validation

Owned addition: research_v3/thresholds.py and tests/research/test_thresholds.py.
No existing source was edited, no real root was invoked, and no GPU inference,
training, worker interruption or Q prediction was performed.

## API

```python
choose_global_threshold(labels, scores, score_kind, label_mask) -> dict
freeze_thresholds(root, run_manifest, prepared_metadata,
                  *, loader=None, scorer=None) -> dict
```

The selection API takes full binary N-by-C targets and full finite scores.
Probability scores must be within [0,1]. The frozen label mask must be nonempty,
unique, in range and have positive/negative V support. Candidates are the linear
5th through 95th percentiles (step 5) of all full-target V scores, plus the
primary cutoff. Selection maximizes masked Macro-F1, zero_division=0, then
prefers nearer the primary cutoff, then larger. Result keys include operative
(always .5 probability, otherwise 0), chosen (auxiliary optimum), candidates,
metrics [{threshold, masked_macro_f1}], label_mask and explicit rule metadata.
There is no labelwise optimization.

freeze_thresholds accepts the actual prepared_manifest datasets envelope or
dataset map. Output covers unique artifact IDs in thresholds.json and includes
design_hash, run_manifest_hash, all planned aliases, standard model-request
identity, status bytes hash and validation-source provenance. It authenticates
the entire checkpoint set before any V load; all must be complete, reload
verified and contain hash-verified saved model files. Validation uses only the
actual select parquet and its common visible inputs, verifies source file
hashes, model input identity, selection role ID hash and actual text hashes,
and uses the fixed select_ap_mask. It rechecks checkpoints/source hashes after
scoring before atomic output. The default scorer lazily imports score_saved;
the common core-compatible diagnostics bundle format is supported.

Strict E2 and single-label tasks never invoke the V loader or scorer. Strict
multilabel keeps its fixed primary cutoff with auxiliary=None. Single-label
operative=None indicates argmax. Auxiliary optima do not affect primary Q
predictions. The module never reads D/Q labels or fits a model.

loader(root, metadata)->(Example rows, dict[sample_id, VisibleInput]) and
scorer(checkpoint_directory, tuple[VisibleInput])->scores are trusted injection
hooks for isolated fixtures only; production callers should omit them.

## TDD evidence

Initial RED command:
`.venv-research/Scripts/python.exe -m unittest tests.research.test_thresholds -q`
exited 1: 7 methods / 16 subtest assertion failures for missing thresholds.py.

Initial GREEN: same module with -v exited 0, 7 tests in 1.113 seconds.

Expanded final GREEN: same module with -v exited 0, 10 tests in 1.110 seconds.
Coverage includes candidate population and masked objective, primary rule,
nearest/larger exact ties, invalid masks/scores/targets, unique alias coverage,
strict/single-label V sentinels, pending/reload/artifact/model-request failures,
whole-set preflight before V, V IDs/input/source tampering, and artifact changes
during scoring. Test scoring is CPU-only injected arrays in temporary roots.

Full suite was left to the root agent after concurrent writers finish because
freeze tests snapshot global source inventory.

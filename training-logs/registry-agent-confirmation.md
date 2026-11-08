# Confirmation/F6 workflow software validation

Owned files: research_v3/confirmation.py, tests/research/test_confirmation.py,
and this note. No other source/configuration/lock or real research artifact was
edited. No real Q, actual pilot/human state, real training worker, or GPU was
used. Only temporary synthetic CPU fixtures were fitted/scored.

## Production API

```python
prepare_confirmation(root, pilot_id, *, audit_frames) -> manifest
freeze_confirmation_edits(root, pilot_id, reviewed_pairs) -> edits_freeze
predict_confirmation(root, pilot_id) -> result_manifest
verify_confirmation(root, pilot_id) -> result_manifest  # read-only, no scoring
verify_confirmation_audit_frame(root, pilot_id, actual_global_frame) -> summary
```

All entry points verify the actual evaluation.freeze before any Q/model or
saved-edits read, then verify the actual selection lock. No verifier injection
or synthetic receipt option exists in the production API.

audit_frames maps all four declared audit dataset IDs to {file,sha256}. Each
referenced JSON contains dataset_id, purpose='audit',
selection_policy='uniform_groups_then_uniform_row',
model_outcomes_used_for_sampling=False, sampling_seed=7301, and
records=[{sample_id,group_id,labels}]. There must be 200 independent actual Q
groups, or all available groups if fewer. The frames are source-authenticated
and must exactly reproduce draw_audit(data,'audit',200,7301,set()), including the
one uniform row chosen within each group. Alternate valid 200-group frames and
caller-selected seeds are rejected. Their bytes remain bound to later
edits/results. Groups appearing in any
dataset's A frame or any prepared D partition are excluded globally, including
cross-dataset shared base groups.

The selected pilot's authenticated selection-lock metadata supplies pilot_n=50,
n_planned=400..1200, real human review status and the checkpoint. Registry views
must be exactly eval:<pilot_id>:intervention/control and match the selected
E1 6k/data2026 SVM or DistilBERT train42 checkpoint. Prepared/source/input bytes,
model request/status/files and tokenizer identities are checked. M1 cue lists
are recomputed solely from the actual fixed 6000-row labeled training subset
and its common visible inputs. M4 human spans/windows are recomputed from the
actual normalized Q source and tokenizer.

Preparation exports the entire algorithmically eligible remaining group frame
in uniform random order with seed7301 and one uniform eligible row per group.
Human review templates remain blank. No response/error filtering occurs.
Explicit human decisions must cover every preceding candidate until the frozen
planned valid count is reached, or the entire frame is exhausted. Valid pairs
require two independent blind reviewer records and interventions.validate_pair,
including the new explicit M1 content-word replacement attestations. Rejected
or invalid pairs and reasons are retained. Short sets use all valid pairs once,
mark underpowered=True, and never relax eligibility or duplicate groups.
An empty set does not claim human review verification or generate responses.

## Artifact layout

Under root/confirmation/<pilot_id>/:

- manifest.json and candidates.json: bound source/audit/checkpoint packet.
- edits.freeze.json and edits.json: all selected human edits frozen immutably.
- prediction-started.json: edited-input request identity before model scoring.
- results.json and result-manifest.json: exact frozen pairs, responses and hashes.
- predictions-intervention.parquet and predictions-control.parquet: full-target
  PredictionRows using exact F6 view IDs and original sample_id/gold/group_id.
- M1 only, predictions-original.parquet: separate auxiliary original responses.

Result views include file/sha256/rows/checkpoint_run_id/evaluation_id/
base_group_namespace/dataset_id. Model request and edited VisibleInput content
bind the cache request. Reuse authenticates frozen-pair alignment, source,
checkpoint, timestamps, response columns, target/group identities, parquet
bytes and count/power summaries. Rehashed results cannot replace another edit.

Read-only verify_confirmation performs those checks without scoring/writing.
It derives n_actual/n_planned/underpowered/mechanism_evidence_sufficient rather
than accepting receipt booleans. Exact McNemar is reconstructed from independent
base-group correctness and remains sensitivity-only; F6 simultaneous bands
remain primary. Per-role scoring time is persisted and summed as
checkpoint_to_scores_seconds, including saved-artifact verification/loading,
tokenization/feature transforms/forward, excluding global freeze/source I/O and
output writing.

## TDD evidence

Initial RED: targeted unittest module exited1, 8 missing-module assertion
failures. Subsequent fixture serialization (empty Example.fields in parquet)
and integer-key cue hash errors were corrected during implementation.
First complete GREEN: 8 tests, 19.366 seconds.

Additional witnessed RED checks:

- missing read-only verifier: explicit missing-feature assertion failure;
- missing McNemar sensitivity receipt: assertIn failure;
- zero eligible frame incorrectly claiming human reviews: assertFalse failure.

Final targeted command:
`.venv-research/Scripts/python.exe -m unittest tests.research.test_confirmation -q`
exited0: 10 tests passed in 20.451 seconds.

Fixtures patch only the global freeze verifier with clearly synthetic temporary
receipts. They use actual parquet sources/tokenizers and actual saved CPU SVM
fit_run/score_saved inference. The M4 test authenticates actual tokenizer windows
and deliberately stubs scoring with CPU arrays; tiny DistilBERT saved inference
is independently covered by existing pilot tests. This suite does not claim to
replace a real global freeze or real human confirmation. Full-suite execution
is left to the root after concurrent source writers finish.

## Provenance closure review

The initial implementation checked only that A frames had valid source IDs and
declared a uniform policy. That allowed a different valid 200-group frame than
the one actually used by reporting. The focused fix now reconstructs the
predeclared seed7301 draw from actual Q sources for each dataset and requires
the exact sample/group/gold mapping. Synthetic factories were changed from
first200 rows to actual draw_audit output; no real A frame exists or was changed.

New read-only hook verify_confirmation_audit_frame takes the full report
envelope {records: global draw_audit-shaped records, seeds: {dataset:7301},
created_at_utc, evaluation_freeze_hash, guide_freeze_sha256}. It re-verifies the
evaluation lock, prepared manifest and Q/D source bytes, confirmation-bound
per-dataset frame bytes and fixed source-derived draws. It requires precisely
the four audit datasets with seed7301, matching evaluation freeze, no duplicate
or extra identities, and exact global-vs-bound dataset/sample/group/gold maps.
The reporting caller separately authenticates the envelope's byte reference,
guide-freeze bytes and chronology.

The hook returns pilot_id, binding_verified=True, evaluation_freeze_hash,
actual_global_frame_hash, confirmation_audit_frames ({file,sha256} by dataset),
and datasets ({sampling_seed,samples,frame_sha256,sample_group_gold_hash}). It
performs no model scoring or writes. All five public entry points retain the
evaluation-freeze guard before Q/model access.

Focused RED: two methods / three assertion failures demonstrated acceptance of
an alternate valid200 frame, acceptance of seed7302, and the missing global
frame hook. Focused GREEN: complete confirmation module, 12 tests passed in
23.922 seconds, including positive actual fixed samples/global agreement and
negative independent-valid global A, seed, freeze and missing-dataset cases.

# Independent Technical Review

**Reviewer:** Codex (independent reviewer)
**Reviewed:** repository at `14ae67bd18467c2bcb8a36b297cb14aadc6b5a2f` (`main`), plus `PROJECT_CONTEXT.md`, `README.md`, `PROJECT_RECOVERY_AUDIT.md`, and `DEVELOPMENT_LOG.md`
**Mode:** Read-only source review. No source files changed and no tests run.

## Summary

The source implements the core proposed architecture as EfficientNet image features, a PennyLane variational layer when enabled and installed, a demographics MLP, a two-layer GraphSAGE branch, disease-specific three-way fusion, and 14 output logits. The training and prediction paths both construct patient-isolated, all-to-all edges within the supplied batch/query and add self-loops. The provided recovery audit correctly identifies sampler length/drop-last defects and the evaluator's stale three-model defaults.

Several reported claims need qualification. Training does not use the separate `ScanGraphBuilder` or PyG `Batch` collation: it builds an edge index directly from batch patient IDs and passes `meta` as a two-column `graph_x`; the model then discards that tensor and builds graph nodes from image features plus metadata. More significantly, the notebook's subgroup results are not subgroup metrics. Checkpoint ranking uses validation AUROC, but accepts checkpoints with missing validation scores. The quantum comparison also lacks a hard guard proving that the quantum circuit was active.

## Confirmed findings

1. **Architecture dimensions and flow:** `build_model("quantum_gnn_multimodal")` configures EfficientNet-B0, image projection to the configured quantum output size, age/sex `MetadataEmbedder`, `GNNBranch` with `q_out + 2` node input, three-modality `DiseaseSpecificFusion`, and `DiseaseSpecificClassifier`. `MultiModalCXRModel.forward` concatenates image features with the raw two metadata features for GNN input. See `src/models/model_factory.py`, `src/models/classifier.py`, `src/models/fusion.py`, `src/models/gnn_branch.py`, and `src/models/quantum_layer.py`.
2. **GNN implementation:** `GNNBranch` uses two `SAGEConv` layers in the proposed factory variant, with LayerNorm/ReLU and dropout between layers. The last layer also applies LayerNorm/ReLU. Its edges are consumed by PyG convolution, but the input batch is not a PyG `Batch` object.
3. **Training and inference edge rule:** `build_batch_scan_graph` connects each pair of distinct scans with equal patient IDs in both directions and adds one self-loop per scan. Trainer uses it for train and validation/test batches; inference uses the same helper for patient history and a self-loop for a single scan. This is functionally consistent for a query consisting of the scans in one patient, given identical ordering and features.
4. **Data isolation:** the splitter groups by patient ID and explicitly checks patient overlap across train/validation/test. IDs are returned by the dataset for graph construction/monitoring, not concatenated into the model features. `src/dataset/splitter.py`, `src/dataset/dataloader.py`.
5. **Sampler findings from Antigravity are reproducible from source:** `PatientGroupedBatchSampler.__len__` returns a total-scan quotient/ceiling unrelated to its actual patient-boundary flush behavior; `__iter__` does not consult `drop_last`; large-patient chunks can have a one-scan remainder. `src/dataset/dataloader.py:146-183`.
6. **Metric aggregation:** AUROC/AP are calculated per disease over accumulated predictions; undefined one-class disease AUROCs become NaN and are omitted from the mean. F1/precision/recall use a fixed 0.5 threshold. `src/training/metrics.py`.

## Issues missed or requiring correction in `PROJECT_RECOVERY_AUDIT.md`

### Critical / high priority

1. **Subgroup analysis is fabricated as an estimate, not measured — critical reporting defect.** `evaluate_scan_count_subgroups` counts single-scan and multi-scan rows, but for both strata assigns the exact overall test AUROC to `Single-Scan AUROC (Est.)` and `Multi-Scan AUROC (Est.)`. It never receives predictions or targets. The notebook labels this section “PATIENT SUBGROUP ANALYSIS” and exports these values as metrics. These numbers must not be presented as subgroup performance. Fix by computing per-row test predictions and labels and slicing both by test-patient scan-count subgroup; report NaN where classes lack both labels. **Files:** `src/training/evaluator.py::evaluate_scan_count_subgroups`; `notebooks/02_model_training.ipynb` Cell 9 (and generator `scripts/generate_training_notebook.py`). **Tests:** fixtures with deliberately different subgroup ranking/performance; assert distinct subgroup results and correct row counts; confirm one-class handling.

2. **Checkpoint selection is mostly validation based, but score validation is permissive — medium.** Cell 9 ranks fallback checkpoints by recorded `metrics.val_auc_mean`, and active experiments retain restored best weights, so the intended selection criterion is validation AUROC and test labels do not select the checkpoint. However, malformed checkpoints lacking that field receive a default score of `-1.0`; if all available files lack it, one is still selected and called verified. Reject missing/non-finite validation scores and verify checkpoint provenance/config before reporting. **Files:** Cell 9 in notebook and `scripts/generate_training_notebook.py:934-968`; `src/training/trainer.py::CheckpointManager`.

3. **Quantum ablation may silently be classical — high validity risk.** `get_quantum_layer` returns `IdentityQuantumLayer` when PennyLane is unavailable, even when `enabled=True`; factory then trains/evaluates the experiment under a quantum name. Notebook's architecture inspection prints Active/Identity but does not fail the comparison on Identity. A run can therefore be reported as quantum without a variational circuit. **Files:** `src/models/quantum_layer.py::get_quantum_layer`; `src/models/model_factory.py::build_model`; notebook architecture verification and training cells. **Tests:** simulate `PENNYLANE_AVAILABLE=False` and assert a quantum experiment is rejected or explicitly relabeled; assert a valid quantum run contains `QuantumLayer` and finite nonzero gradients for `quantum_weights`, `input_proj`, and `output_proj`.

4. **Test examples CLI can visualize randomly initialized weights — high.** `scripts/generate_model_comparison.py` builds each model and logs a warning if there is no checkpoint, but still places that model in `models_dict` and calls the “Real Test-Set Radiographs” visualizer. This can produce predictions on held-out images from random weights, contradicting the README's strict research-inference guarantee and contaminating presented examples. Require a verified checkpoint for every model included in real comparison, or omit/label the model explicitly. **Files:** `scripts/generate_model_comparison.py::main`; `src/training/evaluator.py::visualize_test_predictions`; `src/inference/checkpoint_loader.py`.

### Important methodological / functional issues

5. **No true PyG graph batching or prebuilt graph path — clarify architecture.** `Trainer.train_epoch` and `eval_epoch` create a new batch-local adjacency from patient IDs. The `graph_x` passed by trainer is just `[B,2]` metadata, and `MultiModalCXRModel.forward` ignores it for the proposed model because it is not `[B, q_out+2]`, then forms `[img_feats, meta]`. This is internally coherent, but the recovery audit's descriptions of precomputed graph features / PyG batching should not imply a `Data`/`Batch` pipeline or use of the separately built label-Jaccard patient graph. `PatientGraphBuilder` is a distinct label-derived graph and is not the proposed training path. **Files:** `src/training/trainer.py::_model_forward`, `train_epoch`, `eval_epoch`; `src/graph/graph_builder.py::build_batch_scan_graph`; `src/models/classifier.py::forward`; `src/graph/graph_builder.py::PatientGraphBuilder`.

6. **Inference only represents scans supplied together.** The UI/API does not retrieve a patient's historical scans; callers must manually supply a same-patient image list or `patient_history`. Thus ordinary single-image inference uses only a self-loop, even for a patient with companion scans in the dataset. The training benefit is present only for batches where a patient's companion scans co-occur. Avoid claims of automatic longitudinal retrieval. **Files:** `src/inference/predictor.py::predict`; `src/inference/preprocessing.py::build_patient_history_graph`.

7. **Sampler's documented cross-patient rule is right, but batches are not necessarily full or all rows retained under future fixes.** Current implementation yields every scan despite `drop_last=True`; `drop_last` is ignored rather than dropping remainders. Any remainder-packing patch must also prove every index appears exactly once per epoch and preserve the patient isolation invariant. For patient groups larger than batch size, chunks split longitudinal context across mini-batches by design.

8. **Four-model comparison is not fully controlled for isolating components.** Models 3 and 4 differ in both quantum layer and model path, while Model 1/2 use shared concat fusion and Model 3/4 disease-specific fusion. Comparing Model 4 vs Model 3 changes quantum presence but the factory also gives both disease-specific fusion, so this is the cleaner quantum contrast; Model 4 vs Model 2 changes quantum, graph, and fusion together and cannot isolate combined causal contributions. The optional `quantum_cnn` is needed for quantum-without-graph comparison but is optional/manual. Results should report these contrasts accurately and train planned ablations systematically. **Files:** `src/models/model_factory.py` experiment branches; notebook Cells 8A–8I; evaluator defaults.

9. **Validation/test loss is a batch mean, not sample-weighted.** `Trainer.train_epoch` and `eval_epoch` average each batch's mean loss equally. Validation/test loaders retain a smaller final batch (`drop_last=False`), so the final batch is over-weighted relative to its number of examples. Sampler-produced training batch sizes vary even more. Aggregate loss should weight each batch loss by batch size. AUROC is correctly accumulated per sample. **Files:** `src/training/trainer.py::train_epoch`, `eval_epoch`. **Tests:** compare reported loss against direct full-dataset loss for uneven batch sizes.

10. **Fixed-threshold F1 is not validation-calibrated.** Metrics use threshold 0.5 throughout, even though class weighting can shift probability calibration substantially. AUROC/AP remain threshold-independent, but F1/precision/recall comparisons should either prespecify 0.5 as the protocol or choose class thresholds on validation only and freeze them before test evaluation. **Files:** `src/training/metrics.py::compute_metrics`; notebook evaluation cells.

11. **Faithfulness methods have limitations that make aggregate “faithfulness” claims weak.** Image masking replaces normalized pixels with `image.mean()` (single scalar across channels) and ranks pixels rather than spatial regions; modality/tabular metrics compare raw prediction deltas and can call explanations faithful based on one perturbation; graph explainer ablates individual directed edges, so one direction of a reciprocal edge is retained. These are useful diagnostics but not strong causal validation without controls/repeated random baselines and paired edge removal. **Files:** `src/explainability/faithfulness.py`; `src/explainability/gnn_explainer.py`.

12. **No confidence intervals or patient-level uncertainty in reported comparison.** AUROC means are computed across diseases, but the code does not supply confidence intervals, paired comparisons, or account for within-patient scan correlation. Given multiple scans per patient, uncertainty should bootstrap patients (not individual images) and preserve paired model predictions. **Files:** `src/training/metrics.py`; notebook Cell 9/report generation. **Tests:** bootstrap reproducibility and patient-cluster resampling.

## Incorrect or overstated claims in lead audit/context

- “PyG batching” should be described as batch-local edge-index construction consumed by `SAGEConv`, not PyG `Batch` collation.
- `graph_x=meta` is not used as GNN node input for the proposed model; the model reconstructs node input using learned image features plus metadata. Antigravity calls it “redundant” and that is accurate for current behavior, but the mechanism matters for claims about graph construction.
- The main notebook does not prove quantum training merely by checkpoint existence; it must record/verify circuit activation and backend.
- “subgroup analysis” output in `src/training/evaluator.py` is explicitly marked `Est.` but copied from overall AUC, and should be treated as invalid rather than a rough estimate.
- The README's “163 tests” is stale relative to the lead audit's reported 179 collected tests; the lead audit's local dependency failure explanation cannot be independently confirmed here because no tests were run as requested by no-test policy.

## Recommended fixes, ordered

1. Stop publishing current scan-count subgroup AUROC; implement it using predictions/targets grouped at patient level.
2. Enforce active `QuantumLayer` for quantum-named experiments, or save/checkpoint metadata that reports a classical fallback and excludes it from quantum claims.
3. Make comparison CLI require verified weights before generating real test-set examples.
4. Fix `PatientGroupedBatchSampler` so `drop_last`, `__len__`, remainder packing, and exact sample coverage agree; write exhaustive small synthetic partition tests.
5. Clarify graph documentation: online batch-local graph, node features `[quantum image embedding; age; sex]`, no PyG `Batch`, no use of training label-derived graph in this model.
6. Weight aggregate losses by sample count; specify validation-only threshold calibration (or lock 0.5 as a declared decision rule).
7. Make checkpoint selection a single shared utility that verifies payload structure, architecture/config/backend, and maximum validation score; fail on missing or invalid validation scores.
8. Add patient-cluster bootstrap intervals and paired model comparisons; strengthen explainer controls and symmetrically ablate undirected graph links.

## Required verification suite after fixes

- Sampler property tests over empty/singleton groups, exact-fit groups, overlarge groups with size-one remainders, `drop_last` on/off, multiple seeds, and epoch changes: every yielded index valid and unique, coverage policy explicit, `len(list(sampler)) == len(sampler)`, batch size policy satisfied.
- Graph parity tests: training/validation/inference for identical scans and IDs yields the same directed edges, self-loops, and node order; no cross-patient edges; singleton graph has one self-loop.
- Model forward/backward tests for all 4 core variants and `quantum_cnn`; dimensions `[B,14]`, finite loss/gradients, and active VQC gradients where quantum is claimed.
- Checkpoint tests: wrong architecture, partial state dict, missing/NaN val score, and ties; selected epoch must be the maximum validation score; test targets must never affect selection.
- Notebook generation/validation test: notebook equals generator output; all required cells parse; execute a tiny synthetic pipeline through training, validation selection, one-time test evaluation, and subgroup computations.
- Evaluator tests: subgroup metrics derived from distinct predictions/targets, correct patient counts, per-disease undefined classes handled, loss correct for unequal batch sizes.
- Explainability tests: controlled synthetic model where salient image region/feature/edge is known; region deletion vs matched random baseline; reciprocal-edge removal; deterministic patient bootstrap.

## Scope and reproducibility notes

The report is based on source inspection, not execution. I did not install dependencies or run tests. Repository status at review start was `main` at `14ae67b`, with `PROJECT_CONTEXT.md`, `PROJECT_RECOVERY_AUDIT.md`, and `DEVELOPMENT_LOG.md` already untracked; those files were read but not modified.

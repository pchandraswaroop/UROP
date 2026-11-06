# Comprehensive Project Recovery Audit

**Project:** Explainable Multimodal Quantum–GNN Framework for Multi-Disease Chest X-ray Diagnosis
**Lead AI Agent:** Google Antigravity (Gemini 3.8)
**Independent Reviewer:** Codex (`CODEX_AUDIT.md`)
**Audit Date:** 2026-11-06
**Git Baseline Commit:** `14ae67bd18467c2bcb8a36b297cb14aadc6b5a2f` (Branch: `main`)
**Audit Type:** Read-Only Verification, Architecture Audit, Codex Review Reconciliation

---

## 1. Executive Summary

A comprehensive read-only audit of the recovered codebase was conducted following local project corruption and AI conversation loss. The repository was recovered from GitHub at commit `14ae67b`.

### Key Audit Findings:
1. **Core Repository Integrity:** All core architecture files, data loaders, split files, graph builders, explainability modules, tests, and notebook generators are intact and systematically organized.
2. **Current Training vs. Proposed Collation (Fact vs Proposal):**
   - *Current Implementation:* `src/training/trainer.py` constructs a batch-local `edge_index` directly by scanning `batch["patient_id"]` with `build_batch_scan_graph()`. It does **not** currently use PyG `Batch.from_data_list()`.
   - *Proposed Refactoring:* Modularize patient graph creation into individual PyG `Data` objects and assemble using `Batch.from_data_list()`.
3. **Training vs. Inference Context Boundaries:**
   - *Training:* Oversized patient scan groups ($K > 32$, up to 184 scans) are partitioned into subgraphs to prevent GPU VRAM exhaustion. Scans in separate chunks do not exchange messages.
   - *Inference:* All $K$ scans uploaded by a user ($K \le 10$) are loaded into a single graph with all-to-all connectivity.
   - *Conclusion:* Complete training-inference history equivalence cannot be claimed because training message passing is capacity-bounded by mini-batch limits.
4. **Patient-Aware GNN Batching Defects (Confirmed):**
   - Evaluated on `train.csv` (79,006 scans):
     * Emits **19 singleton chunks (size 1)** due to unconstrained remainder slicing of large patient records.
     * Sampler `__len__` reports 2,468 batches, but `__iter__` yields 3,012 batches (22% mismatch).
     * `drop_last` parameter is ignored during batch iteration.
5. **Codex Review Findings (Confirmed & Prioritized):**
   - Placeholder subgroup AUROCs in `evaluate_scan_count_subgroups`.
   - Silent quantum fallback to `IdentityQuantumLayer` when PennyLane is missing.
   - Random weight fallback in `scripts/generate_model_comparison.py` when checkpoints are absent.
   - Unweighted batch loss aggregation in `Trainer`.
   - Permissive default `-1.0` ranking for corrupt checkpoints in notebook Cell 9.
6. **Local Test Suite Discrepancy (Confirmed):**
   - 179 collected tests: 108 passed, 63 failed, 4 skipped, 4 errors.
   - 100% of failures and errors stem from missing optional packages (`timm`, `datasets`, `pennylane`, `shap`) in the local environment, NOT corrupted source code. Independent modules (`test_training.py`, `test_graph.py`, `test_splitter.py`, `test_label_encoder.py`) passed 100% cleanly (44/44).

---

## 2. Codebase & Environment Inventory

### 2.1 File Structure
```
UROP/
├── config/
│   └── config.yaml                     # Central hyperparameters & 9 experiment definitions
├── data/
│   └── splits/                         # full_index.csv (112,120), train.csv (79,006),
│                                       # val.csv (16,460), test.csv (16,654)
├── notebooks/
│   ├── 01_dataset_preparation.ipynb    # Dataset download, stats, graph construction
│   ├── 02_model_training.ipynb         # 39-cell Colab training notebook
│   └── 03_explainability.ipynb         # Standalone explainability notebook
├── pages/
│   ├── 1_Prediction.py                 # Multi-label inference interface
│   ├── 2_Explainability.py             # Multimodal attribution dashboard
│   ├── 3_Model_Information.py          # Architecture matrix & specs
│   └── 4_Research_Evaluation.py        # Faithfulness & benchmark metrics
├── scripts/
│   ├── analyze_statistics.py           # Dataset statistics generator
│   ├── build_graph.py                  # Graph builder CLI
│   ├── build_splits.py                 # GroupShuffleSplit generator
│   ├── generate_model_comparison.py    # Metric & curve visualizer CLI
│   ├── generate_training_notebook.py   # Deterministic generator for 02_model_training.ipynb
│   ├── inspect_dataset.py              # HF dataset inspector
│   ├── run_experiment.py               # Headless experiment runner
│   ├── run_explainability.py           # Offline attribution runner
│   ├── smoke_test_notebook_logic.py     # Runtime smoke tests for notebook cells
│   ├── validate_notebook.py            # AST & JSON notebook validator
│   └── verify_pipeline.py              # End-to-end smoke test
├── src/
│   ├── dataset/                        # dataloader, hf_loader, label_encoder, preprocessor, splitter
│   ├── explainability/                 # gradcam, tabular_shap, gnn_explainer, modality_attribution, faithfulness, visualizer
│   ├── graph/                          # graph_builder (ScanGraphBuilder & Jaccard PatientGraphBuilder)
│   ├── inference/                      # predictor, checkpoint_loader, preprocessing, explanation_service
│   ├── models/                         # classifier, fusion, gnn_branch, image_branch, mlp_baseline, model_factory, quantum_layer
│   ├── training/                       # evaluator, experiment, losses, metrics, trainer
│   ├── ui/                             # Streamlit widgets, plots
│   └── utils/                          # logger, reproducibility
├── tests/                              # 10 test modules (179 collected test cases)
├── app.py                              # Streamlit main entrypoint
├── CODEX_AUDIT.md                      # Independent reviewer technical audit
├── presentation_for_madam.md           # Defense & presentation outline
├── PROJECT_CONTEXT.md                  # Master context & agent collaboration protocol
├── requirements.txt                    # Local dependencies
└── requirements_colab.txt              # Colab GPU dependencies
```

---

## 3. Discrepancy & Defect Detailed Breakdown

### Defect 1: Sampler Length Mismatch, Ignored `drop_last`, and Singleton Chunks
- **Affected File:** `src/dataset/dataloader.py`
- **Affected Class / Function:** `PatientGroupedBatchSampler.__init__`, `__len__`, `__iter__`
- **Current Defect:**
  1. `__len__` returns `total_scans // batch_size` (2,468), but iterator flushes early at patient boundaries yielding 3,012 batches.
  2. `drop_last=True` is ignored during chunking and trailing batch output.
  3. Patients with $>32$ scans (up to 184) are sliced with `scans[i:i+32]`, producing 19 single-scan remainder batches in `train.csv`.
- **Expected Behavior:**
  1. For $K > B$, partition into $M = \lceil K / B \rceil$ balanced chunks of size $\approx K / M \ge 16$.
  2. Pre-generate `self._batches` at epoch start; `__len__` returns `len(self._batches)`.
  3. When `drop_last=True`, drop batches with $< B$ scans.
- **Verification Tests:**
  - `test_sampler_len_equals_iter_len`: Assert `len(sampler) == len(list(sampler))` across seeds.
  - `test_no_singleton_chunks`: Assert `min(len(b) for b in sampler) >= 4`.
  - `test_sampler_drop_last`: Assert no incomplete batch is yielded when `drop_last=True`.

### Defect 2: Placeholder Subgroup AUROC Calculation
- **Affected File:** `src/training/evaluator.py`
- **Affected Function:** `evaluate_scan_count_subgroups`
- **Current Defect:**
  Counts single-scan and multi-scan patient rows, but copies the overall test AUROC into both `Single-Scan AUROC (Est.)` and `Multi-Scan AUROC (Est.)`.
- **Expected Behavior:**
  Slice test predictions and labels into single-scan ($K=1$) and multi-scan ($K \ge 2$) cohorts, and compute true per-disease and mean AUROCs on each subset separately.
- **Verification Tests:**
  - `test_subgroup_metrics_computed_authentically`: Provide synthetic predictions with intentionally different performance on single vs multi-scan cohorts; assert distinct AUROCs.

### Defect 3: Silent Quantum Fallback
- **Affected File:** `src/models/quantum_layer.py` & `src/models/model_factory.py`
- **Affected Function:** `get_quantum_layer`, `build_model`
- **Current Defect:**
  When `enabled=True` but PennyLane is missing, silently returns `IdentityQuantumLayer`. A training run can execute classical layers under a quantum model name.
- **Expected Behavior:**
  If an experiment has `quantum` in its name or `enabled=True` is specified, raise `ImportError` or require explicit `allow_classical_fallback=True`. In notebook training cells, verify `assert isinstance(model.quantum_layer, QuantumLayer)`.
- **Verification Tests:**
  - `test_quantum_model_fails_when_pennylane_unavailable_without_fallback`: Mock `PENNYLANE_AVAILABLE=False`; assert building a quantum model raises an error unless explicitly permitted.

### Defect 4: Random-Weight Visualization in Evaluation CLI
- **Affected File:** `scripts/generate_model_comparison.py`
- **Affected Function:** `main` (Step 3: Loading Models)
- **Current Defect:**
  If a model checkpoint is missing, initializes random weights and calls `visualize_test_predictions`, producing misleading prediction figures.
- **Expected Behavior:**
  Strictly require a verified checkpoint file for every model in the comparison CLI; if missing, omit the model or halt with a clear error message.
- **Verification Tests:**
  - `test_comparison_cli_rejects_missing_checkpoints`: Verify CLI raises or cleanly skips uncheckpointed models.

### Defect 5: Unweighted Batch Loss Aggregation
- **Affected File:** `src/training/trainer.py`
- **Affected Function:** `Trainer.train_epoch`, `Trainer.eval_epoch`
- **Current Defect:**
  `total_loss += loss.item()` divides by `n_batches`. When batch sizes vary, small batches have disproportionate weight on the epoch loss.
- **Expected Behavior:**
  Accumulate sample-weighted loss: `total_loss_sum += loss.item() * batch_size` and divide by total samples in the epoch.
- **Verification Tests:**
  - `test_sample_weighted_loss_aggregation`: Verify that a DataLoader with uneven batch sizes matches direct dataset-level loss.

### Defect 6: Permissive Checkpoint Selection in Notebook Cell 9
- **Affected File:** `scripts/generate_training_notebook.py` & `notebooks/02_model_training.ipynb`
- **Affected Cell:** Cell 9 (Phase 4: Held-Out Test Evaluation)
- **Current Defect:**
  Ranks disk checkpoints by `val_auc_mean`, defaulting to `-1.0` if missing. If all checkpoints are corrupt or missing validation scores, one is still selected and labeled "verified".
- **Expected Behavior:**
  Reject checkpoints where `val_auc_mean <= 0.0` or non-finite. Require strictly positive, verified validation scores.
- **Verification Tests:**
  - `test_checkpoint_selection_rejects_invalid_scores`: Verify checkpoint loader rejects non-positive/missing validation scores.

---

## 4. Technical Risks & Mitigations Summary

| Risk | Origin | Severity | Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **Silent Classical Execution** | Missing PennyLane on Colab | High | Hard assertion in `build_model` and Cell 6/8G. |
| **Fabricated Subgroup Metrics** | Placeholder in evaluator | High | Compute true per-cohort sliced AUROCs in `evaluator.py`. |
| **GPU OOM on Oversized Patients** | Patients with up to 184 scans | High | Balanced sub-graph partitioning ($\approx 23$ scans/chunk). |
| **Misleading Visualizations** | Missing checkpoint fallback | Medium | Strict checkpoint validation before visualization CLI. |
| **Epoch Loss Skew** | Variable batch sizes | Low | Sample-weighted loss accumulation in `Trainer`. |

---
*End of Updated Audit Report.*

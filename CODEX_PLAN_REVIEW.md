# Independent Review of Antigravity's Proposed Implementation Plan

**Reviewer:** Codex
**Scope:** Read-only review of the eight proposed fixes in `PROJECT_CONTEXT.md` / `PROJECT_RECOVERY_AUDIT.md`, against the source at baseline `14ae67b`.
**No source changes, tests, commits, or pushes were made.**

## Recommendation summary

| # | Proposed fix | Classification | Recommendation |
|---|---|---|---|
| 1 | Refactor patient grouped sampler | Necessary bug fix, with a research/policy choice | **Revise** the specification first; `drop_last=True` cannot coexist with exactly-once coverage in that same epoch when `N % B != 0`. |
| 2 | Use `PyG Batch.from_data_list()` | Optional refactor | **Reject as currently justified**; no demonstrated correctness, maintainability, or performance gain over the flat edge index. |
| 3 | Compute authentic scan-count subgroup AUROC | Necessary bug fix | **Approve**, with scan-level metric definition and patient-cluster uncertainty clearly stated. |
| 4 | Enforce quantum execution | Necessary validity guard, with explicit classical compatibility | **Approve with revision**; fail for quantum-named runs unless classical fallback is explicitly opted into and relabeled. |
| 5 | Block visualizations with missing checkpoints | Necessary bug fix | **Approve**, with an explicit skip-or-fail policy and verified weight loading. |
| 6 | Sample-weight epoch loss | Necessary bug fix | **Approve** for the existing mean-reduced per-example loss; tests should compare against a full-split calculation. |
| 7 | Strict checkpoint validation | Necessary bug fix | **Revise**; reject absent, non-finite, and out-of-range scores, but do not require AUROC `> 0`. |
| 8 | Four-model comparison defaults | Recommended improvement / protocol correction | **Approve with scope**; synchronize all defaults, reports, plots, and notebook candidates to the four declared core models. |

## A. Sampler: define a coherent coverage policy

The requirements “every dataset index occurs exactly once per epoch” and “`drop_last=True` drops incomplete batches” conflict whenever the number of retained scans is not divisible by batch size. A scan omitted by `drop_last=True` did not occur that epoch; a scan carried to a later epoch did not occur exactly once in the current epoch. There is no sampler implementation that can satisfy both literally.

**Recommended default for this patient graph training:** use `drop_last=False`, emit every index once per epoch, and make `__len__` equal the actual number of batches. Patient groups may be split into bounded chunks, but edges must only connect same-patient nodes in the same chunk. Pack small patient groups together to approach the target batch size, while keeping each patient's nodes disconnected from other patients. Permit a smaller final batch and include its samples in training. This is the policy most consistent with full data utilization and current intent.

If fixed-size batches are operationally required, set `drop_last=True` and document that exactly `N % B` samples are omitted from each epoch after packing. To avoid a fixed subset being systematically omitted, rotate/shuffle the ordering across epochs; coverage is then a multi-epoch property, not an exactly-once-per-epoch guarantee. Never claim that both modes have exact per-epoch coverage.

The current sampler does not actually omit data: `drop_last` is ignored and all samples are yielded. It does have incorrect `__len__`, early boundary flushes, and singleton remainders for large groups. A suitable sampler should have an explicit policy for groups larger than `B`, small groups, and the final partial batch, then test exact index multiplicity and graph component isolation.

## B. Balanced partitioning and minimum chunk size

For an oversized patient group of size `K`, the proposed balanced rule is `M = ceil(K/B)`, then divide `K` as evenly as possible over `M` chunks. This guarantees each chunk is at most `B`; the minimum size is `floor(K/M)`. It does **not** guarantee a minimum of 16 for all configured batch sizes.

| Batch size `B` | `K=33` | `K=65` | `K=184` | Minimum 16 guaranteed? |
|---:|---:|---:|---:|---|
| 32 | 17,16 | 22,22,21 | 31,31,31,31,30,30 | Yes for these examples |
| 16 | 11,11,11 | 13,13,13,13,13 | 16,16,16,16,16,16,16,16,15,15,15,15 | No |
| 8 | chunks cannot reach 16 due to cap | same | same | No |
| 4 | chunks cannot reach 16 due to cap | same | same | No |

For `K=184, B=16`, `M=12`, so balanced chunks are 8 chunks of 16 and 4 of 15; the minimum is 15, below 16.

More generally, a partition into `M` chunks each in `[16,B]` is possible only when `B >= 16` and `16M <= K <= BM`. For `B=16, K=33`, `M` must be at least 3 to satisfy the maximum, but then minimum required total is 48: impossible. For `B<16`, the constraint is immediately impossible. Configured batch sizes include CPU 4, GPU 32, model-specific 8/16, and a documented recommendation to reduce GPU batch size to 16. Therefore a hard minimum of 16 is not valid for all supported configurations. Use a configurable minimum bounded by `B`, or simply eliminate singleton chunks by balanced division and permit smaller chunks when arithmetic requires it. Singleton avoidance is feasible even when a 16-node lower bound is not.

There is also no requirement that each resulting training batch be exactly one patient group. Combining multiple patient chunks is sound if the edge construction preserves disconnected components. If the sampler packs small groups, those groups remain separate graph components within one flat batch.

## C. Is PyG `Batch.from_data_list()` necessary?

No. The current `build_batch_scan_graph` emits an ordinary flat `edge_index` with batch-local node offsets. `SAGEConv(x, edge_index)` is designed to consume exactly this representation. The model has one flat node tensor `[B, q_out+2]`, and each patient forms a disconnected component. This is a standard and correct disjoint-union representation.

`Batch.from_data_list()` would formalize the disconnected-component collation and can be useful when each graph carries separate graph-level attributes, graph labels, or heterogeneous edge/node attributes. The current task has scan-level labels and no graph pooling; patient IDs already provide all information needed to form edges. Creating `Data` objects then batching them is additional object construction and offset bookkeeping, and may add CPU overhead. It does not fix sampler coverage, cross-patient edge errors, or train/inference context differences by itself.

**Recommendation:** keep flat edge-index construction unless an identified future requirement needs per-graph attributes or PyG graph transforms. Add invariant tests that edges stay within patient IDs and indices are in range. The proposed PyG conversion is an optional refactor, not a necessary fix.

## D. Training versus inference context

Both paths use the same semantic edge rule: complete directed connections among supplied scans for one patient, plus self-loops. Training is capacity-bounded: a patient with more than `B` scans is divided into chunks, so scans in different chunks cannot exchange messages in that optimizer step. Inference creates one graph from all scans uploaded together. The model therefore has edge-rule parity, but not history-context parity when inference contains more scans than one training chunk (and inference inputs may also be a selected/ad-hoc subset rather than the same chunking policy).

This difference is a research-methodology decision, not a graph-batching bug. It should be explicit in the methods and evaluation. Suggested validation: compare inference on a selected subset of size `<= B` with the same scans as one training-style chunk; additionally measure sensitivity to context-set size and chunk/subset selection. Do not describe inference as using full historical context unless those scans are supplied and the model has been evaluated for that context size. `PROJECT_CONTEXT.md` currently says inference inputs `K <= 10`; verify this against UI validation before relying on the bound.

## E. Quantum validity and backward compatibility

The source currently allows `get_quantum_layer(enabled=True)` to return `IdentityQuantumLayer` if PennyLane is missing. In addition, the notebook sets `quantum_enabled = (device == "cuda")`; a CPU quantum-named run could therefore intentionally receive an identity layer. The proposed strict guard is necessary for a run reported as quantum.

Recommended API policy:

- If experiment name/protocol claims quantum, require both explicit `quantum_enabled=True` and an active `QuantumLayer`; fail early if PennyLane/backend is unavailable.
- Classical experiment keys (`cnn_only`, `classical_multimodal`, `cnn_gnn_multimodal`, etc.) must continue to use the intended classical/identity feature path.
- If a developer deliberately needs a compatibility fallback for debugging, require an explicit opt-in, record it in run/checkpoint metadata, and relabel the run as classical. Never compare or report it under the quantum result key.
- Validate gradients through input projection, circuit weights, and output projection for active circuit runs. A nonzero gradient is a verification target, though a particular batch can legitimately yield a near-zero value; test multiple seeded inputs/weights and check finiteness plus parameter connectivity.

Thus preserve backward compatibility for intentionally classical models, not silent fallback for a quantum claim.

## F. Subgroup AUROC, unit of analysis, and dependence

The project predicts a label per scan and the current evaluation computes AUROC per scan. Scan-count subgroup metrics should therefore also be computed per scan: assign each test scan to its patient's test-set count (`1` versus `>=2`), slice the scan-level labels and probabilities, then compute per-disease AUROC within each subset. These are scan-level subgroup metrics; they do not estimate patient-level diagnostic performance and scans from a patient are correlated.

For a disease with no positive or no negative examples inside a subgroup, AUROC is undefined: return/report NaN and its valid-class denominator. Do not replace it with zero, and do not silently imply a 14-class macro used all diseases. The existing metric convention of excluding undefined disease AUROCs from the macro is acceptable if `n_valid_auc` and the affected diseases are also visible. For threshold-dependent F1/recall, state the fixed 0.5 threshold or tune per disease on validation only.

Uncertainty and between-model tests must account for patient clustering. Use a patient-cluster bootstrap: sample patient IDs with replacement within the relevant split/subgroup and include all their scans, preserving paired model predictions when comparing models. Do not bootstrap scans independently. This is a recommended methodology improvement; it is separate from fixing fabricated subgroup point estimates.

## G. Checkpoint selection

Checkpoint selection must use validation data only. The active `Experiment` path restores best validation weights before `evaluate_test`; the notebook's disk fallback ranks payloads by `metrics.val_auc_mean`. The bug is permissive metadata handling (`-1.0` for missing score), not use of test labels for selection.

Reject a missing score, non-numeric score, NaN/Inf, or score outside the AUROC range `[0,1]`; verify required state dict structure and strict model compatibility before loading. Do **not** require score `> 0`: AUROC 0.0 is a valid metric, even if it is poor; selection should still choose the highest valid validation score. Likewise, do not call a checkpoint “verified” solely because `torch.load` succeeds. Ties need deterministic handling (e.g. highest score then a documented epoch tie-break). Keep test evaluation downstream of selection and never feed test metrics back into selection.

## Per-fix disposition and scope notes

### 1. PatientGroupedBatchSampler refactoring

**Classification:** Necessary bug fix.
**Recommendation: REVISE.** Fix `__len__`, ensure declared coverage/drop policy, avoid pathological singleton chunks where possible, and preserve no-cross-patient edges. First choose either full per-epoch coverage with a final partial batch (`drop_last=False`, recommended) or fixed full batches with explicit omissions (`drop_last=True`). A universal minimum chunk size of 16 is invalid for configured `B=4,8,16`; make it configurable or avoid specifying it as a guarantee. Tests must cover exact per-index multiplicity under both policies.

### 2. PyG `Batch.from_data_list()` graph collation

**Classification:** Optional refactor.
**Recommendation: REJECT as a required implementation item.** The flat edge-index approach is correct for current scan-level node supervision and disconnected patient components. Reconsider only if a concrete PyG graph-level feature/transform or maintainability need arises and benchmark its overhead.

### 3. Authentic scan-count subgroup AUROC

**Classification:** Necessary bug fix.
**Recommendation: APPROVE.** Compute scan-level AUROC from held-out scan predictions/targets and each patient's scan-count group. Return NaN for disease/subgroup with one class, plus valid disease count. Add patient-cluster bootstrap intervals as a separate recommended methodology enhancement.

### 4. Strict quantum execution guard

**Classification:** Necessary validity guard.
**Recommendation: APPROVE WITH REVISION.** A quantum-named run must prove an active variational circuit and fail when the dependency/backend/config is unavailable. Keep classical experiment keys working. Any explicit fallback must be labeled and stored as classical, never as a quantum result.

### 5. Blocking visualizations when checkpoints are missing

**Classification:** Necessary bug fix.
**Recommendation: APPROVE.** The real-test visualization CLI must not run a randomly initialized model. Either fail before producing figures or skip the model with a conspicuous recorded reason; load with strict architecture checks. A partial model comparison is acceptable only if output lists exactly which checkpoint-backed models were included.

### 6. Sample-weighted training/evaluation loss

**Classification:** Necessary bug fix.
**Recommendation: APPROVE.** With a loss reduced as a mean over examples (and same class weights for each batch), aggregate `batch_mean * batch_size`, then divide by number of examples actually processed. Handle empty loaders explicitly. This makes the reported epoch loss equal the per-example average across batches despite variable sampler batch sizes. Test both train/eval paths using uneven batch sizes. If the criterion later uses nonuniform per-example weights/reduction, aggregate the unreduced loss numerator and denominator consistent with its definition instead.

### 7. Strict checkpoint validation

**Classification:** Necessary bug fix.
**Recommendation: REVISE.** Reject missing, malformed, non-finite, and out-of-range validation AUROC values, and require strict state-dict/model compatibility. Do not reject score `0.0` merely for being non-positive. Validation selection should sort all valid scores and use a deterministic tie-break. Ensure checkpoint config records quantum-active state and architecture so the right inference model is instantiated.

### 8. Four-model comparison defaults

**Classification:** Recommended improvement / alignment with declared protocol.
**Recommendation: APPROVE WITH SCOPE.** The declared core protocol contains `cnn_only`, `classical_multimodal`, `cnn_gnn_multimodal`, and `quantum_gnn_multimodal`; evaluator defaults and three-model documentation omit Model 3. Synchronize defaults, CLI, plot colors/layout, comparison report text, and notebook candidate lists. Make output reflect only successfully trained/verified models, never imply an unrun model was evaluated. This is not itself a fix to model-training correctness, and contrasts must be interpreted carefully because Model 4 vs Model 2 changes graph, quantum, and fusion; Model 4 vs Model 3 is the more direct quantum contrast.

## Verification tests required before implementation is considered complete

- **Sampler:** property tests for `N < B`, `N == B`, `N % B != 0`, small patient groups, and oversized groups `K=33,65,184`, under `B=4,8,16,32`; prove index coverage/multiplicity matches the documented drop policy, all edge endpoints are valid, no cross-patient edges, and `len(sampler) == len(list(sampler))` for each epoch/seed.
- **Graph representation:** identical patient/node order gives equivalent edge sets for training and inference helpers; singleton self-loop, pair, mixed-patient disconnected components, and bounded chunks.
- **Subgroups:** synthetic cases with intentionally different single/multi scan rankings, per-disease one-class subsets, correct counts and NaN/valid-class reporting; deterministic patient-cluster bootstrap.
- **Quantum:** active VQC required for quantum key; missing package/backend fails; classical keys remain buildable; forward/backward produces finite outputs and circuit parameter gradient connectivity.
- **Visualization/checkpoints:** absent checkpoint never invokes model prediction; malformed payload, wrong architecture, absent/NaN/Inf/out-of-range AUROC rejected; AUROC 0.0 remains valid; tie-break deterministic; selection unaffected by test metrics.
- **Loss:** train and eval losses match a full-example weighted calculation for deliberately uneven batches and weighted BCE.
- **Four-model defaults:** evaluator, CLI, report, and notebook list the same intended four keys, gracefully report missing artifacts, and don't generate random-weight predictions.

## Additional inconsistency in revised context

`PROJECT_CONTEXT.md` Section 6 lists metadata encoding as `(age - 63.0) / 100.0`, but `src/dataset/preprocessor.py::MetadataEncoder` defaults to `(age - 46.9) / 16.6`. That section's “identical demographic encoding” claim is false as written; update the documentation to reflect the actual encoder or explicitly change the code in a separately reviewed plan. This discrepancy also affects reproducibility of any claimed training/inference parity.

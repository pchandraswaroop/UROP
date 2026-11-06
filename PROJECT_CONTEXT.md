# Master Project Context: Explainable Multimodal Quantum-GNN Framework

## 1. Project Overview & Scientific Objectives
- **Target Task:** 14-disease multi-label thoracic pathology diagnosis from frontal chest radiographs.
- **Dataset:** NIH ChestX-ray14 (`chehablab/NIHChestXR`) containing 112,120 frontal radiographs from 30,805 unique patients.
- **Disease Classes (14 Targets):**
  1. Atelectasis, 2. Cardiomegaly, 3. Consolidation, 4. Effusion, 5. Emphysema,
  6. Fibrosis, 7. Hernia, 8. Infiltration, 9. Mass, 10. Nodule,
  11. Pleural Thickening, 12. Pneumonia, 13. Pneumothorax, 14. Edema.
- **Zero Label Leakage Protocol:**
  - NIH index 0 is "No Finding" (mapped to an all-zero target tensor).
  - NIH indices 1–14 are mapped to 0-indexed [0..13].
  - Patient ID and scan ID are strictly excluded from model feature inputs.
  - All scans from the same patient are strictly assigned to only one split (`train.csv`, `val.csv`, or `test.csv`) using `GroupShuffleSplit` on `patient_id`.

---

## 2. Multimodal Architecture Specifications
The primary research framework is a **3-Branch Multimodal Model** with disease-specific attention gating:

1. **Image Branch + Parameterized Quantum Layer:**
   - Backbone: `EfficientNet-B0` (extracting 1,280-dimensional image representation) or `ViT-Small` (384d).
   - Quantum Circuit: PennyLane 4-qubit Variational Quantum Circuit (`QuantumLayer`).
   - Circuit Architecture: Linear projection to 4 features $\to \text{Tanh} \to R_y(\theta_i \pi)$ Angle Embedding $\to$ `StronglyEntanglingLayers` $\to \langle \sigma_z \rangle$ expectation measurements $\to$ projection to 64 dimensions.
   - Under CUDA, PennyLane executes on CPU in float32 and transitions back to GPU seamlessly.
   - Strict Quantum Guard: Quantum-named experiments require an active `QuantumLayer` with PennyLane. Classical experiments use classical identity/projection pathways.
2. **Clinical Demographic Branch:**
   - Input: Standardized patient age and binary sex ($\text{M}=1.0, \text{F}=0.0, \text{unknown}=0.5$).
   - Normalization Formula (Verified from `src/dataset/preprocessor.py`):
     $$\text{age\_norm} = \frac{\text{clamp}(\text{age}, 0, 120) - 46.9}{16.6}$$
   - Embedder: `MetadataEmbedder` (2-layer MLP with LayerNorm, ReLU, Dropout) $\to$ 64 dimensions.
3. **Patient Scan Graph GNN Branch:**
   - Representation: Operates over mini-batch scan graphs where each node is an individual radiograph.
   - Node Features: Concatenation of image quantum features and demographic metadata ($64 + 2 = 66$ dimensions).
   - Graph Edges: Directed longitudinal links connecting companion scans of the same patient (`patient_id[i] == patient_id[j]`) plus self-loops (`(i, i)`).
   - Graph Collation: Direct flat `edge_index` with block-diagonal patient components (no PyG `Batch.from_data_list()` required).
   - Message Passing: 2-layer GraphSAGE (`SAGEConv`) with LayerNorm and ReLU $\to$ 64 dimensions.
4. **Disease-Specific Multimodal Attention Fusion:**
   - 14 independent gating heads: For each disease $d \in \{0..13\}$, a learned 3-way softmax gate dynamically computes attention weights $[g_{\text{img}}, g_{\text{clin}}, g_{\text{gnn}}]$.
   - Fused Representation: Weighted sum projected into 128 dimensions per disease ($[B, 14, 128]$).
5. **Multi-Label Classification Heads:**
   - 14 dedicated linear classification heads mapping $[B, 128] \to [B, 1]$ logit $\to$ sigmoid probability.

---

## 3. Four-Model Experimental Comparison Protocol & Compatibility

### 3.1 Declared Core Protocol
The core evaluation protocol evaluates four declared models:

| Model ID | Experiment Key | Visual Modality | Demographic Modality | Graph Modality | Fusion Scheme | Scientific Role |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | `cnn_only` | EfficientNet-B0 (1280d) | None (`ZeroClinicalBranch`) | None | Simple Concat | Unimodal vision baseline |
| **Model 2** | `classical_multimodal` | EfficientNet-B0 (1280d) | Age, Sex (`MetadataEmbedder`) | None | Simple Concat | Standard multimodal baseline |
| **Model 3** | `cnn_gnn_multimodal` | EfficientNet-B0 (1280d) | Age, Sex (`MetadataEmbedder`) | Patient GraphSAGE | Disease-Specific | Classical GNN ablation (no quantum) |
| **Model 4** | `quantum_gnn_multimodal` | EfficientNet + 4-Qubit VQC | Age, Sex (`MetadataEmbedder`) | Patient GraphSAGE | Disease-Specific | **Proposed Full Framework** |
| *Model 5* | `quantum_cnn` *(Ablation)* | EfficientNet + 4-Qubit VQC | Age, Sex (`MetadataEmbedder`) | None | Disease-Specific | Quantum ablation (no graph) |

### 3.2 Backward Compatibility with Existing Callers
- `THREE_MODELS = ["cnn_only", "classical_multimodal", "quantum_gnn_multimodal"]` is preserved in `src/training/evaluator.py` to maintain 100% backward compatibility with existing unit tests and legacy fixtures.
- `FOUR_MODELS = ["cnn_only", "classical_multimodal", "cnn_gnn_multimodal", "quantum_gnn_multimodal"]` is explicitly defined and wired into all four-model benchmark paths (`scripts/generate_model_comparison.py`, `scripts/generate_training_notebook.py`, and notebook Cell 9).

---

## 4. Patient-Aware Training Design & Mini-Batching Policy

### 4.1 Graph Construction Strategy
- **Flat `edge_index` Collation:** We retain the verified direct batch-local edge construction. PyG `Batch.from_data_list()` is not required because `SAGEConv` naturally processes disjoint patient components, and the model uses scan-level multi-hot supervision with no graph-level pooling.
- **Node Definition:** Each scan is a distinct graph node.
- **Intra-Patient Edges:** Complete directed clique with self-loops ($K^2$ edges for $K$ companion scans in that batch/chunk).
- **Inter-Patient Edges:** Strictly 0 cross-patient edges. Packing multiple patients into one batch places their cliques side-by-side in a block-diagonal adjacency matrix with zero cross-talk.
- **Scan-Level Multi-Label Supervision:** Each scan preserves its individual 14-disease label vector $y_i \in \{0, 1\}^{14}$. Loss is applied scan-by-scan via BCEWithLogitsLoss (`WeightedBCELoss`).

### 4.2 Patient-Aware Mini-Batching Policy
- **Coverage Policy (`drop_last=False`, Default):**
  - Every single dataset index appears **exactly once per epoch** ($1:1$ multiplicity).
  - The final partial batch is retained and processed with sample-weighted loss.
- **Fixed Batch Mode (`drop_last=True`):**
  - When fixed-size batches are operationally required, trailing batches with $< B$ scans are dropped.
  - Exactly $\text{len}(current\_batch) < B$ samples are omitted from that epoch. Coverage is a multi-epoch property through dynamic shuffling across epochs (`RandomState(seed + epoch)`).
- **Capacity & Isolation Invariants:**
  - **No batch ever exceeds $B$** ($\text{len}(b) \le B$ strictly enforced).
  - **Zero cross-patient edges** are ever created.
- **Balanced Partitioning for Oversized Patients ($K > B$):**
  - When a patient has $K > B$ scans (up to 184 scans), divide into $M = \lceil K / B \rceil$ balanced chunks of size $\lfloor K / M \rfloor$ or $\lceil K / M \rceil$.
  - Guaranteed: Every chunk is $\le B$; avoids singleton remainders where arithmetic allows.
  - Bounded by $B$: We do not enforce an invalid universal lower bound of 16, as batch sizes of 4, 8, and 16 are supported.
- **Length Consistency:** `_generate_batches()` builds all batches at epoch start. `__len__` returns `len(self._batches)` and `__iter__` yields from `self._batches`, guaranteeing `len(sampler) == len(list(sampler))` exactly.

---

## 5. Inference Design & Context Boundaries

### 5.1 Single-Image Inference (Ad-Hoc Consultation)
- User inputs 1 radiograph and demographic metadata (Age, Sex).
- Constructs a 1-node graph with self-loop $(0, 0)$ ($1$ edge).
- Runs through the full model; outputs 14 disease probabilities for that image.

### 5.2 Multiple-Image Inference (Longitudinal Patient Consultation)
- User uploads $K$ scans confirmed to belong to the same patient and selects a `primary_scan_index`.
- Constructs a $K$-node graph with complete directed connectivity ($K^2$ edges including self-loops).
- Outputs predictions for the `primary_scan_index` only; companion scans provide inductive relational context during GraphSAGE message passing.
- Does not require a patient ID or database lookup.

### 5.3 Important Boundary: Training vs. Inference Patient Context
**There is NO complete training-inference history equivalence.**
- **During Training:** Large patient scan histories are capacity-bounded and partitioned into sub-graphs of at most `batch_size` (e.g. 32) to prevent GPU VRAM exhaustion. A patient with 184 scans is partitioned across 6 separate batches; scans in chunk 1 never exchange messages with scans in chunk 6.
- **During Inference:** A clinician uploads an ad-hoc set of scans (typically $K \le 10$). All $K$ scans reside in memory simultaneously in a single graph, allowing cross-scan message passing among all uploaded scans.
- Consequently, training models message passing over capacity-bounded temporal chunks, while inference models message passing over the uploaded subset.

---

## 6. Training-Inference Consistency Invariants
1. **Identical Node Features:** Always $[B, 66]$, formed by `torch.cat([img_feats (64), meta (2)], dim=-1)`.
2. **Identical Edge Rule:** Complete directed clique with self-loops ($K^2$ edges for $K$ nodes).
3. **Identical Self-Loops:** Explicitly included in both training cliques and single/multi-image inference graphs.
4. **Identical GraphSAGE Layers:** Exact same 2-layer `SAGEConv` weights with LayerNorm, ReLU, and Dropout.
5. **Identical Demographic Encoding:** Exact same formula: `(clamp(age, 0, 120) - 46.9) / 16.6` and binary sex flag.
6. **Identical Quantum Feature Extraction:** Exact same projection and 4-qubit parameterized PennyLane circuit.
7. **Robust Graph Sizing:** Native support for 1-node graphs ($1$ edge) and $K$-node graphs ($K^2$ edges) with zero code branching.
8. **Primary Scan Output Isolation:** Multi-scan inference cleanly maps index `primary_scan_index` to the reported 14 disease outputs.

---

## 7. Confirmed Implementation Defects to be Resolved

1. **Placeholder Subgroup AUROC (`src/training/evaluator.py::evaluate_scan_count_subgroups`):**
   Currently copies overall test AUROC into both single-scan and multi-scan columns. Must slice test predictions and labels into single-scan ($K=1$) and multi-scan ($K \ge 2$) cohorts and compute authentic scan-level AUROCs. For one-class disease subsets, return `NaN` and report `n_valid_classes`.
2. **Silent Quantum Fallback (`src/models/quantum_layer.py::get_quantum_layer`):**
   Currently falls back to `IdentityQuantumLayer` silently. Must raise `ImportError` for quantum-named runs if PennyLane is unavailable.
3. **Random-Weight Visualization (`scripts/generate_model_comparison.py`):**
   Currently visualizes random weights if checkpoints are missing. Must strictly require verified checkpoints.
4. **Unweighted Batch Loss Aggregation (`src/training/trainer.py`):**
   Currently averages batch mean loss equally. Must accumulate sample-weighted loss: $\sum (\text{loss}_i \times B_i) / \sum B_i$.
5. **Strict Checkpoint Validation (`src/inference/checkpoint_loader.py` & notebook Cell 9):**
   Reject `strict=False` as the verification criterion. Require exact state-dict key and tensor-shape match with strict loading (`strict=True`). Reject partially compatible checkpoints. In notebook Cell 9, accept valid AUROCs in $[0.0, 1.0]$ and reject missing, non-finite, and out-of-range values.
6. **Four-Model Alignment (`src/training/evaluator.py`):**
   Explicitly define `FOUR_MODELS` alongside legacy `THREE_MODELS` and wire into comparison tools.

---

## 8. Collaboration Protocol & Team Responsibilities
- **Google Antigravity (Gemini 3.8)**: LEAD AI AGENT. Coordinates work, maintains master project context, designs architecture, validates technical correctness before training, and manages development documentation.
- **Codex**: Independent technical reviewer. Reviews architecture, training loops, GNN batching, and model correctness.
- **GitHub Copilot**: Coding assistant for discrete functions, debugging, and targeted refactoring.
- **Rules of Engagement:**
  - No commits or pushes without explicit user approval.
  - No modification of source code without prior proposal and approval.
  - Verify claims against actual repository code; distinguish confirmed facts from assumptions.
  - Shared repository on branch `main` at baseline commit `14ae67b`.

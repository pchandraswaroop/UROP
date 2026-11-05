# Explainable Multimodal Quantum-GNN Framework for Multi-Disease Chest X-ray Diagnosis

## Overview

Research-grade implementation of an explainable multimodal system that diagnoses **14 thoracic diseases** from chest X-rays using:

- **Image Branch**: EfficientNet-B0 / ViT backbone + optional Quantum circuit layer
- **Clinical Branch**: Patient-level Graph Neural Network (GraphSAGE) or MLP baseline
- **Fusion**: Disease-specific attention-gated multimodal fusion
- **Classification**: 14 independent sigmoid outputs (multi-label)
- **Explainability**: Grad-CAM++, SHAP, graph explanations, modality attribution

**Dataset**: NIH ChestX-ray14 via [chehablab/NIHChestXR](https://huggingface.co/datasets/chehablab/NIHChestXR) — 112,120 chest X-rays from 30,805 patients.

---

## Dataset Schema (Verified)

| Field | Type | Notes |
|-------|------|-------|
| `patient_id` | string | Patient identifier — never used as feature |
| `scan_id` | int32 | Scan identifier — never used as feature |
| `image` | PIL Image (244×244) | Frontal X-ray |
| `age` | int32 | Patient age |
| `sex` | string | M / F |
| `labels` | list[int32] | NIH label indices: 0=No Finding, 1–14=diseases |

**14 Disease Classes (our targets):**
Atelectasis, Cardiomegaly, Consolidation, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, Edema

---

## Project Structure

```
urop/
├── config/
│   └── config.yaml              # Central configuration
├── data/
│   ├── raw/                     # HF cache (auto-populated on first run)
│   ├── splits/                  # train.csv, val.csv, test.csv
│   └── stats/                   # Visualizations and dataset_report.md
├── src/
│   ├── dataset/
│   │   ├── hf_loader.py         # HuggingFace loading + schema inspection
│   │   ├── label_encoder.py     # list[int32] → [14] multi-hot tensor
│   │   ├── splitter.py          # Patient-level GroupShuffleSplit
│   │   ├── preprocessor.py      # Image transforms + metadata encoder
│   │   └── dataloader.py        # PyTorch Dataset + DataLoader factory
│   ├── graph/
│   │   └── graph_builder.py     # Patient graph (Jaccard co-morbidity edges)
│   ├── models/
│   │   ├── image_branch.py      # timm backbone (EfficientNet / ViT)
│   │   ├── quantum_layer.py     # PennyLane variational circuit (CPU/GPU)
│   │   ├── gnn_branch.py        # GraphSAGE clinical branch
│   │   ├── mlp_baseline.py      # MLP on metadata (GNN comparison baseline)
│   │   ├── fusion.py            # Disease-specific attention fusion
│   │   └── classifier.py        # 14-head classifier + top-level model
│   ├── training/
│   │   ├── losses.py            # Weighted BCE for class imbalance
│   │   ├── metrics.py           # AUC, AP, F1 per-class + macro
│   │   └── trainer.py           # Training loop + CheckpointManager
│   └── utils/
│       ├── logger.py            # Logging setup
│       └── reproducibility.py  # Seed setting
├── scripts/
│   ├── inspect_dataset.py       # Schema + sample inspection
│   ├── analyze_statistics.py    # Full statistics + visualizations
│   ├── build_splits.py          # Generate split CSVs
│   ├── build_graph.py           # Build patient graph
│   └── verify_pipeline.py       # End-to-end smoke test
├── tests/
│   ├── test_label_encoder.py
│   ├── test_splitter.py
│   └── test_dataloader.py
├── notebooks/
│   └── 01_dataset_preparation.ipynb  # Google Colab notebook
├── requirements.txt             # Local CPU dependencies
├── requirements_colab.txt       # Colab GPU dependencies
└── README.md
```

---

## Quick Setup

### Local (CPU development)

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Inspect dataset (downloads ~2.5GB on first run)
python scripts/inspect_dataset.py --sample_n 200

# 4. Generate statistics and visualizations (sample)
python scripts/analyze_statistics.py --sample_n 2000

# 5. Build patient-level splits (full dataset)
python scripts/build_splits.py

# 6. Run smoke test
python scripts/verify_pipeline.py --n_samples 64
```

### Google Colab (GPU training)

Open `notebooks/01_dataset_preparation.ipynb` in Colab. The notebook handles:
- Environment setup
- Full dataset download and caching
- Statistics + visualizations
- Split generation
- Graph building
- Pipeline verification

---

## Configuration

All hyperparameters are in `config/config.yaml`. Key settings:

```yaml
dataset:
  hf_repo: "chehablab/NIHChestXR"
  num_classes: 14

model:
  image_backbone: "efficientnet_b0"
  quantum:
    enabled: false          # Set true on Colab GPU
  gnn:
    enabled: true
    edge_threshold: 0.3

training:
  batch_size: 4             # CPU
  colab_overrides:
    batch_size: 32          # GPU
```

---

## Running Tests

```bash
cd urop
pytest tests/ -v
```

Expected: 24+ tests pass without network access (mocked HF dataset).

---

## Design Decisions and Limitations

### GNN Clinical Branch — Known Limitations

> **This is documented transparently per the architecture requirements.**

The NIH ChestX-ray14 dataset provides only `age` and `sex` as non-label metadata.
This creates a fundamental limitation for the GNN:

1. **Node features (age_norm, sex_binary)** are weak discriminators — patients with similar demographics will have indistinguishable node representations.
2. **Edges (Jaccard co-morbidity)** are constructed from *training* labels. These labels are never available for val/test patients at inference time, so new patients cannot be connected to the training graph.
3. **Transductive limitation**: The graph is static and cannot generalise inductively without additional clinical features (e.g., lab results, vitals, smoking history — not present in this dataset).

**Mitigation**:
- An **MLP baseline** (age + sex only) is implemented and compared.
- If GNN AUC ≤ MLP AUC + 1%, the GNN branch is dropped from fusion.
- The disease-specific fusion gate learns to down-weight the clinical branch if it adds no signal.

### Label Encoding

The HF dataset stores `labels` as `list[int32]` with NIH canonical indices:
- Index 0 = "No Finding" (not a disease target — produces all-zero label tensor)
- Indices 1–14 = 14 diseases → mapped to 0-indexed [0..13] in our tensors

### Patient Leakage Prevention

All scans from the same patient are assigned to the same split via `GroupShuffleSplit`. This is verified at split time with an explicit assertion.

### Class Imbalance

BCEWithLogitsLoss with per-class positive weights (capped at 10×) handles extreme imbalance (e.g., Hernia at ~0.2% prevalence).

---

## Citation

```bibtex
@inproceedings{Wang_2017,
  doi    = {10.1109/cvpr.2017.369},
  year   = 2017,
  author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and
            Mohammadhadi Bagheri and Ronald M. Summers},
  title  = {ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks
            on Weakly-Supervised Classification and Localization of Common
            Thorax Diseases},
  booktitle = {2017 IEEE CVPR}
}
```

Dataset: [chehablab/NIHChestXR](https://huggingface.co/datasets/chehablab/NIHChestXR) — CC0 1.0 Universal License.

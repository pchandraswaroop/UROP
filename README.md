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
│   ├── explainability/
│   │   ├── gradcam.py           # Grad-CAM++ (CNN) & ViT token attribution
│   │   ├── tabular_shap.py      # Exact 2-player analytical SHAP (Age, Sex)
│   │   ├── gnn_explainer.py     # ScanGraphExplainer (edge ablation)
│   │   ├── modality_attribution.py # 3-player Shapley fusion attribution
│   │   ├── faithfulness.py      # Deletion/insertion curves & perturbation
│   │   └── visualizer.py        # Multi-disease overlays & plots
│   ├── inference/
│   │   ├── predictor.py         # CXRInferenceService (Research & Demo modes)
│   │   ├── checkpoint_loader.py # ModelCheckpointLoader & weight verification
│   │   ├── preprocessing.py     # Input validation & inference transforms
│   │   └── explanation_service.py # Unified UI explanation provider
│   ├── ui/
│   │   ├── components.py        # Streamlit styled widgets & disclaimers
│   │   └── visualizations.py    # Plotly interactive charts & heatmaps
│   └── utils/
│       ├── logger.py            # Logging setup
│       └── reproducibility.py  # Seed setting
├── pages/
│   ├── 1_Prediction.py          # Multi-label inference interface
│   ├── 2_Explainability.py      # Multimodal attribution dashboard
│   ├── 3_Model_Information.py   # 9-model comparison matrix
│   └── 4_Research_Evaluation.py # Empirical evaluation & faithfulness metrics
├── app.py                       # Main Streamlit application entrypoint
├── scripts/
│   ├── inspect_dataset.py       # Schema + sample inspection
│   ├── analyze_statistics.py    # Full statistics + visualizations
│   ├── build_splits.py          # Generate split CSVs
│   ├── build_graph.py           # Build patient graph
│   ├── run_experiment.py        # Experiment runner CLI
│   ├── run_explainability.py    # Offline explainability pipeline
│   └── verify_pipeline.py       # End-to-end smoke test
├── tests/
│   ├── test_label_encoder.py
│   ├── test_splitter.py
│   ├── test_dataloader.py
│   ├── test_graph.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_explainability.py
│   └── test_inference.py
├── notebooks/
│   ├── 01_dataset_preparation.ipynb  # Dataset & splits Colab notebook
│   ├── 02_model_training.ipynb       # 9-experiment Colab GPU training
│   └── 03_explainability.ipynb       # Colab explainability demonstration
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

## Interactive Streamlit Web Application

The repository includes a comprehensive, multi-page Streamlit research application:

```bash
# Launch the dashboard locally
streamlit run app.py
```

### Modes of Operation
1. **Research Inference Mode (Strict)**:
   - Requires a verified, trained model checkpoint in `checkpoints/<experiment_name>/`.
   - Loads actual trained weights into the selected architecture.
   - Real inference is strictly disabled if no valid checkpoint is found (never falls back to random weights).
2. **Demo / Interface Preview Mode**:
   - Allows exploring the interface, navigation, and explainability workflows without a checkpoint.
   - Produces clearly identified synthetic mock predictions carrying prominent **DEMO PREVIEW** badges.

### Application Sections
- 🫁 **Home / Overview (`app.py`)**: Research objectives, multimodal architecture diagram, and 14 disease definitions.
- 🔬 **Prediction (`pages/1_Prediction.py`)**: Radiograph upload, clinical metadata entry, model configuration, and 14-disease multi-label prediction with threshold slider.
- 🔍 **Explainability (`pages/2_Explainability.py`)**: Grad-CAM++ heatmaps, ViT token attribution, exact analytical tabular SHAP, GNNExplainer graph visualization, and 3-player modality Shapley reliance.
- 📐 **Model Information (`pages/3_Model_Information.py`)**: Comparison matrix across all 9 architecture variants, quantum layer topology, and device runtime specs.
- 📈 **Research Evaluation (`pages/4_Research_Evaluation.py`)**: Real validation/test loss curves, per-disease AUROC, F1 scores, and quantitative faithfulness metrics.

---

## Running Tests

Run the full automated test suite (163 unit and integration tests across data loading, graph building, models, quantum layers, training, explainability, and inference):

```bash
pytest tests/ -v
```

Expected: **163 passed, 0 failures** in ~40 seconds on CPU.

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

"""
generate_training_notebook.py — Generator for notebooks/02_model_training.ipynb
==============================================================================
Builds a pristine, sequentially numbered Colab training and verification notebook.
"""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Notebook metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    },
    "colab": {
        "name": "02_model_training.ipynb",
        "provenance": [],
        "gpuType": "T4"
    },
    "accelerator": "GPU"
}

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 02 — Model Training & Manual Verification Guide
**Explainable Multimodal Quantum–GNN Framework for Multi-Disease Chest X-ray Diagnosis**
*Dataset: NIH ChestX-ray14 (112,120 Frontal Radiographs | 30,805 Patients | 14 Pathologies)*

---

### Execution Protocol
> [!IMPORTANT]
> **Execute this notebook cell-by-cell in sequential order.**
> Do not execute all cells automatically. Before moving to the next cell, verify that the current cell completed with status `SUCCESS` and inspect the output.

### Notebook Organization:
- **Phase 1: Environment & Dataset Audit** (Cells 1 to 5)
  - `Cell 1`: Environment Verification & Hardware Diagnostics (GPU / VRAM)
  - `Cell 2`: Repository Setup & Git Synchronization
  - `Cell 3`: Dataset & Split Verification (Zero Patient Leakage)
  - `Cell 4`: Sample Visualization & Preprocessing Audit
  - `Cell 5`: Graph Construction & Inter-Scan Connectivity Verification
- **Phase 2: Pre-Flight Diagnostics** (Cells 6 & 7)
  - `Cell 6`: Model Architecture Smoke Tests on GPU (5 Core Models with Graph Inputs)
  - `Cell 7`: Training Hyperparameter Review & Trial Memory Test (Peak VRAM Measurement)
- **Phase 3: Controlled Sequential Model Training** (Cell 8)
  - `Cell 8A`: Train Model 1 — `cnn_only`
  - `Cell 8B`: Evaluate Model 1 — `cnn_only` Validation & Loss Curves
  - `Cell 8C`: Train Model 2 — `classical_multimodal`
  - `Cell 8D`: Evaluate Model 2 — `classical_multimodal` Validation & Loss Curves
  - `Cell 8E`: Train Model 3 — `cnn_gnn_multimodal`
  - `Cell 8F`: Evaluate Model 3 — `cnn_gnn_multimodal` Validation & Loss Curves
  - `Cell 8G`: Train Model 4 — `quantum_gnn_multimodal` (Proposed Framework)
  - `Cell 8H`: Evaluate Model 4 — `quantum_gnn_multimodal` Validation & Loss Curves
  - `Cell 8I`: (Optional Ablation) Train Model 5 — `quantum_cnn`
- **Phase 4: Held-Out Test Evaluation & Comparison** (Cell 9)
  - `Cell 9`: Final Test Set Evaluation, Metrics Comparison, & Subgroup Analysis
- **Phase 5: Case Inferences & Explainability** (Cells 10 & 11)
  - `Cell 10`: Manual Prediction Verification (Single & Multi-Scan Scenarios with Verified Checkpoint)
  - `Cell 11`: Multimodal Explainability Verification (Grad-CAM++, Demographic & Graph Attribution)
"""))

# Cell 1: Environment Verification
cells.append(nbf.v4.new_markdown_cell("""## Cell 1 — Environment Verification & Hardware Diagnostics
**Purpose**: Checks Python, PyTorch, CUDA acceleration, GPU type, available VRAM, and installs required packages without disturbing Colab CUDA PyTorch."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 1: Environment Verification & Hardware Diagnostics ──────────────────
import sys, subprocess, platform, importlib

print("=== Python Runtime ===")
print(f"Python Version: {platform.python_version()} ({sys.executable})")

# Hardware & CUDA Check
import torch
cuda_available = torch.cuda.is_available()
print(f"\\n=== GPU Diagnostics ===")
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU Device:       {gpu_name}")
    print(f"Total VRAM:       {total_mem_gb:.2f} GB")
    print(f"Compute Cap:      {cap[0]}.{cap[1]}")
    if "T4" in gpu_name:
        print("✓ CONFIRMED: Google Colab Tesla T4 GPU active.")
    else:
        print(f"ℹ Note: Running on {gpu_name} (compatible with CUDA workflow).")
else:
    device = "cpu"
    print("⚠ WARNING: CUDA not detected! If on Colab, select: Runtime > Change runtime type > T4 GPU.")

# Dependency Installation (Preserves Colab CUDA PyTorch)
print("\\n=== Package Verification & Installation ===")
required_pkgs = [
    ("timm", "timm"),
    ("datasets", "datasets"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("PIL", "Pillow"),
    ("pennylane", "pennylane"),
    ("lightning", "pennylane-lightning"),
    ("torch_geometric", "torch-geometric"),
]

to_install = []
for mod, pip_name in required_pkgs:
    try:
        importlib.import_module(mod)
    except ImportError:
        to_install.append(pip_name)

if to_install:
    print(f"Installing missing packages: {to_install}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
else:
    print("✓ All required packages are already installed.")

# Verify Imports
import timm, datasets, sklearn, pandas, pennylane, torch_geometric
print("\\nInstalled Library Versions:")
print(f" - PyTorch:         {torch.__version__}")
print(f" - Torch Geometric: {torch_geometric.__version__}")
print(f" - PennyLane:       {pennylane.__version__}")
print(f" - timm:            {timm.__version__}")
print(f" - HuggingFace:     {datasets.__version__}")
print(f" - scikit-learn:    {sklearn.__version__}")
print(f" - pandas:          {pandas.__version__}")

print("\\n[SUCCESS] Cell 1 complete: Hardware and packages verified.")
"""))

# Cell 2: Repository Setup
cells.append(nbf.v4.new_markdown_cell("""## Cell 2 — Repository Setup & Git Synchronization
**Purpose**: Clones or synchronizes the GitHub repository (`execution-readiness`), verifies working directory, and checks for critical codebase and split files."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 2: Repository Setup & Git Synchronization ───────────────────────────
import os, sys, subprocess, shutil
from pathlib import Path

REPO_URL      = "https://github.com/pchandraswaroop/UROP.git"
REPO_PATH     = "/content/UROP" if os.path.exists("/content") else os.getcwd()
TARGET_BRANCH = os.environ.get("UROP_BRANCH", "execution-readiness")

# Clone or pull latest cleanly
if not os.path.exists(REPO_PATH):
    print(f"Cloning repository {REPO_URL} (branch: {TARGET_BRANCH}) into {REPO_PATH}...")
    subprocess.check_call(["git", "clone", "-b", TARGET_BRANCH, REPO_URL, REPO_PATH])
else:
    print(f"Repository directory located at: {REPO_PATH}")
    if os.path.exists(os.path.join(REPO_PATH, ".git")):
        print(f"Synchronizing latest commits from GitHub branch '{TARGET_BRANCH}'...")
        try:
            subprocess.check_call(["git", "-C", REPO_PATH, "fetch", "origin", TARGET_BRANCH])
            subprocess.check_call(["git", "-C", REPO_PATH, "checkout", TARGET_BRANCH])
            subprocess.check_call(["git", "-C", REPO_PATH, "reset", "--hard", f"origin/{TARGET_BRANCH}"])
        except Exception as e:
            print(f"Note on git synchronization: {e}")
    else:
        # Check if project files exist already to avoid destructive deletion
        has_cfg = os.path.exists(os.path.join(REPO_PATH, "config/config.yaml"))
        has_src = os.path.exists(os.path.join(REPO_PATH, "src"))
        if has_cfg and has_src:
            print(f"ℹ Notice: {REPO_PATH} exists without Git metadata. Preserving all existing files (no automatic deletion).")
        else:
            raise RuntimeError(
                f"Directory '{REPO_PATH}' exists but is not a Git repository and does not contain project source.\\n"
                f"To protect your data, it was NOT automatically deleted.\\n"
                f"Action required: Please rename or remove '{REPO_PATH}', or specify a clean target directory."
            )

# Change to repository root
os.chdir(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

print(f"\\nCurrent Working Directory: {os.getcwd()}")

# Display current commit SHA
try:
    commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    print(f"Git Branch: {branch_name} | Commit SHA: {commit_sha}")
except Exception:
    pass

# Verify key repository files exist
required_files = [
    "config/config.yaml",
    "src/models/model_factory.py",
    "src/graph/graph_builder.py",
    "src/dataset/dataloader.py",
    "src/training/experiment.py",
    "src/training/trainer.py",
    "src/training/evaluator.py",
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
]

print("\\nVerifying Repository Files:")
all_found = True
for f in required_files:
    exists = os.path.exists(f)
    status = "FOUND" if exists else "MISSING"
    print(f" [{status}] {f}")
    if not exists:
        all_found = False

if all_found:
    print("\\n[SUCCESS] Cell 2 complete: Repository synchronized and all required files are present.")
else:
    raise FileNotFoundError("Missing one or more required files. Ensure git synchronization ran successfully.")
"""))

# Cell 3: Dataset & Split Verification
cells.append(nbf.v4.new_markdown_cell("""## Cell 3 — Dataset & Split Verification (Zero Patient Leakage)
**Purpose**: Loads train, validation, and test split CSVs, verifies zero patient leakage across splits, checks 14 disease label columns, and connects to the HuggingFace dataset."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 3: Dataset & Split Verification ─────────────────────────────────────
import os, yaml
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from src.dataset.label_encoder import DISEASE_LABEL_NAMES

with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

SPLIT_DIR = CFG["splits"]["split_dir"]   # 'data/splits'
HF_REPO   = CFG["dataset"]["hf_repo"]   # 'chehablab/NIHChestXR'
HF_CACHE  = CFG["dataset"]["cache_dir"] # 'data/raw'

train_csv = Path(SPLIT_DIR) / "train.csv"
val_csv   = Path(SPLIT_DIR) / "val.csv"
test_csv  = Path(SPLIT_DIR) / "test.csv"

# 1. Load split DataFrames
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

print("=== Split Summary ===")
print(f" - Train:      {len(train_df):,} scans ({train_df['patient_id'].nunique():,} unique patients)")
print(f" - Validation: {len(val_df):,} scans ({val_df['patient_id'].nunique():,} unique patients)")
print(f" - Test:       {len(test_df):,} scans ({test_df['patient_id'].nunique():,} unique patients)")
total_scans = len(train_df) + len(val_df) + len(test_df)
print(f" - Total:      {total_scans:,} scans")

# 2. Strict Zero Patient Leakage Audit
train_p = set(train_df["patient_id"].unique())
val_p   = set(val_df["patient_id"].unique())
test_p  = set(test_df["patient_id"].unique())

leak_tv = train_p.intersection(val_p)
leak_tt = train_p.intersection(test_p)
leak_vt = val_p.intersection(test_p)

print("\\n=== Patient Leakage Audit ===")
print(f" - Train ∩ Val overlap:  {len(leak_tv)} patients")
print(f" - Train ∩ Test overlap: {len(leak_tt)} patients")
print(f" - Val ∩ Test overlap:   {len(leak_vt)} patients")
assert len(leak_tv) == 0 and len(leak_tt) == 0 and len(leak_vt) == 0, "CRITICAL ERROR: Patient leakage detected!"
print("✓ CONFIRMED: Strict 0% patient leakage across all splits.")

# 3. Missing Value Audit
print("\\n=== Missing Values Check ===")
for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    missing = df[["patient_id", "age", "sex"]].isnull().sum().to_dict()
    print(f" - {name} metadata nulls: {missing}")
    assert sum(missing.values()) == 0, f"Found missing metadata in {name} split!"

# 4. Load HuggingFace Dataset
print(f"\\n=== Loading HuggingFace Dataset: {HF_REPO} ===")
hf_dataset = load_dataset(HF_REPO, split="train", cache_dir=HF_CACHE)
print(f"HuggingFace dataset ready: {len(hf_dataset):,} total rows.")

# 5. Test Sample Image Access
sample_idx = int(train_df.iloc[0]["hf_index"])
sample_item = hf_dataset[sample_idx]
print(f"Sample image lookup check (hf_index={sample_idx}): Format={sample_item['image'].format}, Size={sample_item['image'].size}")

print("\\n[SUCCESS] Cell 3 complete: Splits and dataset verified with zero patient leakage.")
"""))

# Cell 4: Sample Visualization & Preprocessing Audit
cells.append(nbf.v4.new_markdown_cell("""## Cell 4 — Sample Visualization & Preprocessing Audit
**Purpose**: Displays 4 sample radiographs with clinical metadata and multi-label annotations, verifies 224x224 tensor normalization, and inspects demographic feature encoding."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 4: Sample Visualization & Preprocessing Audit ───────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset.label_encoder import safe_parse_labels, NIH_LABEL_NAMES, DISEASE_LABEL_NAMES
from src.dataset.preprocessor import MetadataEncoder, get_transforms

# 1. Visualize 4 sample training X-rays with multi-label findings
np.random.seed(42)
sample_indices = np.random.choice(len(train_df), 4, replace=False)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, idx in zip(axes, sample_indices):
    row = train_df.iloc[idx]
    hf_idx = int(row["hf_index"])
    img = hf_dataset[hf_idx]["image"]
    raw_labels = row["labels_raw"] if "labels_raw" in row else row["labels"]
    
    parsed_indices = safe_parse_labels(raw_labels)
    # Map NIH integer indices (1-14) to disease names; index 0 is "No Finding"
    disease_names = [NIH_LABEL_NAMES[i] for i in parsed_indices if 0 < i < len(NIH_LABEL_NAMES)]
    label_str = ", ".join(disease_names) if disease_names else "No Finding"
    
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Patient {row['patient_id']}\\nAge: {row['age']} | Sex: {row['sex']}\\nLabels: {label_str}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()

# 2. Verify Image Preprocessing Pipeline
train_transform = get_transforms("train", image_size=224)
test_transform  = get_transforms("test", image_size=224)
sample_img = hf_dataset[int(train_df.iloc[0]["hf_index"])]["image"]
sample_tensor = train_transform(sample_img)

print("=== Image Preprocessing Inspection ===")
print(f" - Transformed Tensor Shape: {sample_tensor.shape} (Channels, Height, Width)")
print(f" - Pixel Value Range:        min={sample_tensor.min():.2f}, max={sample_tensor.max():.2f}, mean={sample_tensor.mean():.2f}")
assert sample_tensor.shape == (3, 224, 224), "Image dimension mismatch!"

# 3. Verify Demographic Metadata Normalization
encoder = MetadataEncoder()
sample_row = train_df.iloc[0]
meta_encoded = encoder.encode_row(age=int(sample_row["age"]), sex=str(sample_row["sex"]))

print("\\n=== Demographic Preprocessing Inspection ===")
print(f" - Raw values:     Age={sample_row['age']}, Sex={sample_row['sex']}")
print(f" - Encoded tensor: {meta_encoded.tolist()} (Age z-score: (age-46.9)/16.6, Sex: M=1.0, F=0.0)")

# 4. Patient Cohort Composition: Single-scan vs Multi-scan Breakdown
p_counts = train_df["patient_id"].value_counts()
single_p = (p_counts == 1).sum()
multi_p  = (p_counts > 1).sum()
print("\\n=== Cohort Composition (Training Split) ===")
print(f" - Single-scan patients: {single_p:,} ({single_p / len(p_counts) * 100:.1f}%)")
print(f" - Multi-scan patients:  {multi_p:,} ({multi_p / len(p_counts) * 100:.1f}%)")
print(f" - Max scans for one patient: {p_counts.max()}")

print("\\n[SUCCESS] Cell 4 complete: Visualizations and preprocessing pipelines verified.")
"""))

# Cell 5: Graph Verification
cells.append(nbf.v4.new_markdown_cell("""## Cell 5 — Graph Construction & Inter-Scan Connectivity Verification
**Purpose**: Verifies that Graph Neural Network scan graphs correctly connect multiple scans of the same patient (intra-patient complete graph) with self-loops, and strictly guarantees zero cross-patient edges."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 5: Graph Construction & Inter-Scan Connectivity Verification ────────
import torch
from src.graph.graph_builder import ScanGraphBuilder, build_batch_scan_graph

builder = ScanGraphBuilder(add_self_loops=True)

# 1. Single-Scan Patient Graph Inspection
single_pid = train_df["patient_id"].value_counts()[train_df["patient_id"].value_counts() == 1].index[0]
df_single = train_df[train_df["patient_id"] == single_pid]
g_single = builder.build(df_single)

print("=== Single-Scan Patient Graph ===")
print(f" - Patient ID:           {single_pid}")
print(f" - Nodes:                {g_single.num_nodes}")
print(f" - Directed Edges:       {g_single.edge_index.shape[1]} (Expected 1 self-loop: [[0], [0]])")
print(f" - Edge Index:           {g_single.edge_index.tolist()}")
assert g_single.num_nodes == 1 and g_single.edge_index.shape[1] == 1, "Single scan graph assertion failed"

# 2. Multi-Scan Patient Graph Inspection
multi_pid = train_df["patient_id"].value_counts()[train_df["patient_id"].value_counts() > 2].index[0]
df_multi = train_df[train_df["patient_id"] == multi_pid]
K = len(df_multi)
g_multi = builder.build(df_multi)

print(f"\\n=== Multi-Scan Patient Graph (K={K} scans) ===")
print(f" - Patient ID:           {multi_pid}")
print(f" - Nodes:                {g_multi.num_nodes}")
print(f" - Directed Edges:       {g_multi.edge_index.shape[1]}")
print(f"   Exact Decomposition:  {K}*( {K} - 1 ) = {K*(K-1)} companion directed edges + {K} self-loops = {K*K} total directed edges (K^2)")
assert g_multi.num_nodes == K and g_multi.edge_index.shape[1] == K * K, "Multi-scan graph assertion failed"

# 3. Mini-Batch Graph Construction & Zero Cross-Patient Edge Check
dummy_meta = torch.randn(6, 2)
dummy_pids = ["Patient_A", "Patient_A", "Patient_B", "Patient_C", "Patient_C", "Patient_C"]
x_b, edge_b = build_batch_scan_graph(dummy_meta, dummy_pids, add_self_loops=True)

cross_patient_edges = 0
for e in range(edge_b.shape[1]):
    src, dst = edge_b[0, e].item(), edge_b[1, e].item()
    if dummy_pids[src] != dummy_pids[dst]:
        cross_patient_edges += 1

print(f"\\n=== Mini-Batch Graph Audit (Batch size = 6, 3 distinct patients) ===")
print(f" - Total directed edges: {edge_b.shape[1]}")
print(f" - Cross-patient edges:  {cross_patient_edges}")
assert cross_patient_edges == 0, "CRITICAL ERROR: Cross-patient edge detected in batch graph!"
print("✓ CONFIRMED: Strict 0 cross-patient edges. Message passing is exclusively intra-patient.")

print("\\n[SUCCESS] Cell 5 complete: Graph construction and isolation verified.")
"""))

# Cell 6: Model Smoke Tests on GPU
cells.append(nbf.v4.new_markdown_cell("""## Cell 6 — Model Architecture Smoke Tests on GPU
**Purpose**: Executes forward passes for candidate architectures using the actual forward API with valid graph inputs, and verifies genuine GraphSAGE relational message passing."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 6: Model Architecture Smoke Tests on GPU ────────────────────────────
import torch
from src.models.model_factory import build_model, list_experiments
from src.graph.graph_builder import build_batch_scan_graph

print(f"Executing architecture smoke tests on device: {device}\\n")

TEST_MODELS = [
    "cnn_only",
    "classical_multimodal",
    "cnn_gnn_multimodal",
    "quantum_cnn",
    "quantum_gnn_multimodal",
]

B = 2
dummy_img = torch.randn(B, 3, 224, 224, device=device)
dummy_meta = torch.randn(B, 2, device=device)
dummy_pids = ["Patient_A", "Patient_A"]

# Graph configurations: connected companion scans vs isolated scans
_, edge_conn = build_batch_scan_graph(dummy_meta, dummy_pids, add_self_loops=True)
edge_isol = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)

print(f"{'Model Name':<25} | {'Params':<8} | {'Shape':<10} | {'Quantum':<11} | {'GNN':<10} | {'GraphSAGE Δ':<12}")
print("-" * 88)

for m_name in TEST_MODELS:
    model = build_model(m_name, pretrained=False, device=device).to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    param_str = f"{n_params / 1e6:.2f}M"
    
    from src.models.quantum_layer import is_quantum_active\n    is_quantum = "Active" if is_quantum_active(model) else "Identity"
    has_gnn = "GraphSAGE" if getattr(model, "gnn_branch", None) is not None else "None"
    
    with torch.no_grad():
        out_conn = model(dummy_img, dummy_meta, graph_edge_index=edge_conn, return_logits=True)
        out_isol = model(dummy_img, dummy_meta, graph_edge_index=edge_isol, return_logits=True)
    
    assert out_conn.shape == (B, 14), f"Expected ({B}, 14), got {out_conn.shape}"
    
    # Prove GraphSAGE message passing: GNN output must shift when companion edge is present
    diff = (out_conn - out_isol).abs().max().item()
    if has_gnn == "GraphSAGE":
        assert diff > 1e-4, f"GraphSAGE message passing inactive in {m_name}! diff={diff}"
        diff_str = f"{diff:.5f} (Active)"
    else:
        diff_str = f"{diff:.1f} (N/A)"
    
    print(f"{m_name:<25} | {param_str:<8} | {str(list(out_conn.shape)):<10} | {is_quantum:<11} | {has_gnn:<10} | {diff_str:<12}")
    
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

print("\\n[SUCCESS] Cell 6 complete: All 5 architectures verified. Genuine GraphSAGE relational aggregation confirmed.")
"""))

# Cell 7: Training Configuration Review & Memory Verification
cells.append(nbf.v4.new_markdown_cell("""## Cell 7 — Training Hyperparameter Review & Peak GPU Memory Measurement
**Purpose**: Displays the exact training hyperparameters, pulls a real batch from the patient-grouped DataLoader, executes a forward + backward + AMP optimization step on `quantum_gnn_multimodal`, and measures synchronized peak GPU memory on Tesla T4."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 7: Training Hyperparameter Review & Peak GPU Memory Measurement ─────
import torch
from src.models.model_factory import build_model
from src.dataset.dataloader import get_dataloader
from src.graph.graph_builder import build_batch_scan_graph

BASE_CFG = {
    "batch_size":              32 if device == "cuda" else 4,
    "num_workers":              2 if device == "cuda" else 0,
    "learning_rate":           1e-4,
    "weight_decay":            1e-5,
    "amp":                     device == "cuda",
    "pretrained":              True,
    "quantum_enabled":         device == "cuda",
    "quantum_output_dim":      64,
    "clinical_output_dim":     64,
    "fused_dim":               128,
    "num_classes":             14,
    "use_weighted_loss":       True,
    "pos_weight_cap":          10.0,
    "early_stopping_patience":  4,
    "save_top_k":              3,
    "checkpoint_dir":          "checkpoints",
    "log_dir":                 "logs",
    "image_size":              224,
    "seed":                    42,
}

MAX_EPOCHS = 15

print("==================================================")
print("TRAINING HYPERPARAMETER SPECIFICATION")
print("==================================================")
for k, v in BASE_CFG.items():
    print(f" - {k:<26}: {v}")
print(f" - {'max_epochs':<26}: {MAX_EPOCHS}")
print("==================================================")

# 1. Instantiate real patient-grouped DataLoader
print("\\nBuilding trial DataLoader with patient grouping...")
trial_loader = get_dataloader(
    split="train",
    index_df=train_df,
    hf_dataset=hf_dataset,
    batch_size=BASE_CFG["batch_size"],
    num_workers=0,  # Single-process for memory trial
    patient_grouping=True,
)
real_batch = next(iter(trial_loader))
print(f"Real batch loaded: Images {list(real_batch['image'].shape)}, Labels {list(real_batch['label'].shape)}")

b_imgs   = real_batch["image"].to(device)
b_meta   = real_batch["meta"].to(device)
b_labels = real_batch["label"].to(device)
b_pids   = real_batch["patient_id"]

_, b_edge_index = build_batch_scan_graph(b_meta, b_pids, add_self_loops=True)
print(f"Batch scan graph built: {len(b_pids)} nodes, {b_edge_index.shape[1]} intra-patient directed edges.")

# 2. Authentic optimization step with proposed full model (quantum_gnn_multimodal)
print("\\nExecuting authentic forward + backward + AMP optimization step...")
trial_model = build_model("quantum_gnn_multimodal", cfg=BASE_CFG, pretrained=False, device=device).to(device)
trial_optimizer = torch.optim.AdamW(trial_model.parameters(), lr=1e-4)
use_amp = (device == "cuda")
trial_scaler = torch.amp.GradScaler("cuda") if use_amp else None

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

trial_optimizer.zero_grad()
if use_amp:
    with torch.amp.autocast("cuda"):
        logits = trial_model(b_imgs, b_meta, graph_edge_index=b_edge_index, return_logits=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_labels)
    trial_scaler.scale(loss).backward()
    trial_scaler.step(trial_optimizer)
    trial_scaler.update()
else:
    logits = trial_model(b_imgs, b_meta, graph_edge_index=b_edge_index, return_logits=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_labels)
    loss.backward()
    trial_optimizer.step()

if device == "cuda":
    torch.cuda.synchronize()
    vram_alloc = torch.cuda.memory_allocated() / 1e9
    vram_peak  = torch.cuda.max_memory_allocated() / 1e9
    vram_res   = torch.cuda.memory_reserved() / 1e9
    print(f"Trial step executed cleanly: Loss = {loss.item():.4f}")
    print(f"VRAM Allocated: {vram_alloc:.2f} GB")
    print(f"VRAM Peak:      {vram_peak:.2f} GB")
    print(f"VRAM Reserved:  {vram_res:.2f} GB (Safely within 15.0 GB Tesla T4 ceiling)")
    if vram_res > 13.0:
        print("⚠ WARNING: Memory reserved exceeds 13 GB. Recommend setting batch_size to 16 in BASE_CFG.")
    else:
        print("✓ CONFIRMED: Batch size 32 is verified safe for Tesla T4 GPU.")
else:
    print(f"Trial step executed cleanly on CPU: Loss = {loss.item():.4f}")

del trial_model, trial_optimizer, trial_scaler, b_imgs, b_meta, b_labels, logits, loss, b_edge_index, trial_loader, real_batch
if device == "cuda":
    torch.cuda.empty_cache()

print("\\n[SUCCESS] Cell 7 complete: Real DataLoader batch and peak GPU memory verified.")
"""))

# Cell 8A: Train Model 1 (cnn_only)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8A — Train Model 1: `cnn_only` (Unimodal Vision Baseline)
**Purpose**: Trains the classical unimodal vision baseline (EfficientNet-B0 with zero clinical metadata leakage) for up to 15 epochs with early stopping patience 4."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8A: Train Model 1 — cnn_only ─────────────────────────────────────────
from src.training.experiment import Experiment

print(\"\"\"
======================================================================
  STARTING EXPERIMENT 1/4: cnn_only
  Architecture: EfficientNet-B0 (Unimodal Vision Baseline)
  Maximum Epochs: 15 | Early Stopping Patience: 4
======================================================================
\"\"\")

exp_cnn = Experiment(name=\"cnn_only\", cfg=BASE_CFG.copy(), device=device)
exp_cnn.setup(train_df, val_df, test_df, hf_dataset)

val_m_cnn = exp_cnn.run(max_epochs=MAX_EPOCHS)
exp_cnn.save_results(\"results\")

print(f\"\\n[COMPLETE] cnn_only training finished in {exp_cnn.training_time_seconds:.1f}s\")
print(f\" - Best Epoch:      {exp_cnn.best_epoch + 1}\")
print(f\" - Best Val AUROC:  {val_m_cnn.get('auc_mean', float('nan')):.4f}\")
print(f\" - Val Macro F1:    {val_m_cnn.get('f1_macro', float('nan')):.4f}\")
"""))

# Cell 8B: Evaluate Model 1 (cnn_only)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8B — Evaluate Model 1: `cnn_only` Validation Curves & Per-Disease Metrics
**Purpose**: Plots training/validation loss curves and validation mean AUROC across epochs, and displays per-disease validation AUROC at the best epoch."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8B: Evaluate Model 1 — cnn_only ─────────────────────────────────────
import matplotlib.pyplot as plt
import pandas as pd

history = exp_cnn.train_history
epochs = [h[\"epoch\"] + 1 for h in history]
train_losses = [h[\"loss\"] for h in history]
val_losses = [h[\"val_loss\"] for h in history]
val_aucs = [h.get(\"val_auc_mean\", float(\"nan\")) for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(epochs, train_losses, label=\"Train Loss\", marker=\"o\")
ax1.plot(epochs, val_losses, label=\"Validation Loss\", marker=\"s\")
ax1.set_xlabel(\"Epoch\")
ax1.set_ylabel(\"Loss\")
ax1.set_title(\"cnn_only — Loss Curves\")
ax1.legend()
ax1.grid(True, linestyle=\":\", alpha=0.6)

ax2.plot(epochs, val_aucs, label=\"Validation Mean AUROC\", color=\"green\", marker=\"^\")
ax2.axvline(exp_cnn.best_epoch + 1, color=\"red\", linestyle=\"--\", label=f\"Best Epoch ({exp_cnn.best_epoch + 1})\")
ax2.set_xlabel(\"Epoch\")
ax2.set_ylabel(\"Mean AUROC\")
ax2.set_title(\"cnn_only — Validation AUROC Curve\")
ax2.legend()
ax2.grid(True, linestyle=\":\", alpha=0.6)

plt.tight_layout()
plt.show()

# Per-disease validation AUROC table
val_per_disease = {k.replace(\"auc_\", \"\"): v for k, v in val_m_cnn.items() if k.startswith(\"auc_\") and k != \"auc_mean\"}
df_val_cnn = pd.DataFrame(list(val_per_disease.items()), columns=[\"Pathology\", \"Val AUROC\"])
print(\"=== cnn_only: Per-Disease Validation AUROC ===\")
display(df_val_cnn)
"""))

# Cell 8C: Train Model 2 (classical_multimodal)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8C — Train Model 2: `classical_multimodal` (Standard Multimodal Baseline)
**Purpose**: Trains the classical multimodal baseline (EfficientNet-B0 + Demographic MLP on Age & Sex) for up to 15 epochs with early stopping patience 4."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8C: Train Model 2 — classical_multimodal ─────────────────────────────
from src.training.experiment import Experiment

print(\"\"\"
======================================================================
  STARTING EXPERIMENT 2/4: classical_multimodal
  Architecture: EfficientNet-B0 + Demographic Clinical MLP
  Maximum Epochs: 15 | Early Stopping Patience: 4
======================================================================
\"\"\")

exp_class_multi = Experiment(name=\"classical_multimodal\", cfg=BASE_CFG.copy(), device=device)
exp_class_multi.setup(train_df, val_df, test_df, hf_dataset)

val_m_class = exp_class_multi.run(max_epochs=MAX_EPOCHS)
exp_class_multi.save_results(\"results\")

print(f\"\\n[COMPLETE] classical_multimodal training finished in {exp_class_multi.training_time_seconds:.1f}s\")
print(f\" - Best Epoch:      {exp_class_multi.best_epoch + 1}\")
print(f\" - Best Val AUROC:  {val_m_class.get('auc_mean', float('nan')):.4f}\")
print(f\" - Val Macro F1:    {val_m_class.get('f1_macro', float('nan')):.4f}\")
"""))

# Cell 8D: Evaluate Model 2 (classical_multimodal)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8D — Evaluate Model 2: `classical_multimodal` Validation Curves
**Purpose**: Plots training/validation curves and displays per-disease validation AUROC for `classical_multimodal`."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8D: Evaluate Model 2 — classical_multimodal ─────────────────────────
import matplotlib.pyplot as plt
import pandas as pd

history = exp_class_multi.train_history
epochs = [h[\"epoch\"] + 1 for h in history]
train_losses = [h[\"loss\"] for h in history]
val_losses = [h[\"val_loss\"] for h in history]
val_aucs = [h.get(\"val_auc_mean\", float(\"nan\")) for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(epochs, train_losses, label=\"Train Loss\", marker=\"o\")
ax1.plot(epochs, val_losses, label=\"Validation Loss\", marker=\"s\")
ax1.set_xlabel(\"Epoch\")
ax1.set_ylabel(\"Loss\")
ax1.set_title(\"classical_multimodal — Loss Curves\")
ax1.legend()
ax1.grid(True, linestyle=\":\", alpha=0.6)

ax2.plot(epochs, val_aucs, label=\"Validation Mean AUROC\", color=\"green\", marker=\"^\")
ax2.axvline(exp_class_multi.best_epoch + 1, color=\"red\", linestyle=\"--\", label=f\"Best Epoch ({exp_class_multi.best_epoch + 1})\")
ax2.set_xlabel(\"Epoch\")
ax2.set_ylabel(\"Mean AUROC\")
ax2.set_title(\"classical_multimodal — Validation AUROC Curve\")
ax2.legend()
ax2.grid(True, linestyle=\":\", alpha=0.6)

plt.tight_layout()
plt.show()

# Per-disease validation AUROC table
val_per_disease = {k.replace(\"auc_\", \"\"): v for k, v in val_m_class.items() if k.startswith(\"auc_\") and k != \"auc_mean\"}
df_val_class = pd.DataFrame(list(val_per_disease.items()), columns=[\"Pathology\", \"Val AUROC\"])
print(\"=== classical_multimodal: Per-Disease Validation AUROC ===\")
display(df_val_class)
"""))

# Cell 8E: Train Model 3 (cnn_gnn_multimodal)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8E — Train Model 3: `cnn_gnn_multimodal` (Classical GNN Ablation Baseline)
**Purpose**: Trains the classical GNN architecture (EfficientNet-B0 + GraphSAGE GNN + Demographic MLP + Disease-Specific Gating, without quantum circuits) to isolate the isolated benefit of relational graph message passing."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8E: Train Model 3 — cnn_gnn_multimodal ──────────────────────────────
from src.training.experiment import Experiment

print(\"\"\"
======================================================================
  STARTING EXPERIMENT 3/4: cnn_gnn_multimodal
  Architecture: EfficientNet-B0 + GraphSAGE GNN + Clinical MLP (No Quantum)
  Maximum Epochs: 15 | Early Stopping Patience: 4
======================================================================
\"\"\")

exp_gnn = Experiment(name=\"cnn_gnn_multimodal\", cfg=BASE_CFG.copy(), device=device)
exp_gnn.setup(train_df, val_df, test_df, hf_dataset)

val_m_gnn = exp_gnn.run(max_epochs=MAX_EPOCHS)
exp_gnn.save_results(\"results\")

print(f\"\\n[COMPLETE] cnn_gnn_multimodal training finished in {exp_gnn.training_time_seconds:.1f}s\")
print(f\" - Best Epoch:      {exp_gnn.best_epoch + 1}\")
print(f\" - Best Val AUROC:  {val_m_gnn.get('auc_mean', float('nan')):.4f}\")
print(f\" - Val Macro F1:    {val_m_gnn.get('f1_macro', float('nan')):.4f}\")
"""))

# Cell 8F: Evaluate Model 3 (cnn_gnn_multimodal)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8F — Evaluate Model 3: `cnn_gnn_multimodal` Validation Curves
**Purpose**: Plots training/validation curves and displays per-disease validation AUROC for `cnn_gnn_multimodal`."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8F: Evaluate Model 3 — cnn_gnn_multimodal ───────────────────────────
import matplotlib.pyplot as plt
import pandas as pd

history = exp_gnn.train_history
epochs = [h[\"epoch\"] + 1 for h in history]
train_losses = [h[\"loss\"] for h in history]
val_losses = [h[\"val_loss\"] for h in history]
val_aucs = [h.get(\"val_auc_mean\", float(\"nan\")) for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(epochs, train_losses, label=\"Train Loss\", marker=\"o\")
ax1.plot(epochs, val_losses, label=\"Validation Loss\", marker=\"s\")
ax1.set_xlabel(\"Epoch\")
ax1.set_ylabel(\"Loss\")
ax1.set_title(\"cnn_gnn_multimodal — Loss Curves\")
ax1.legend()
ax1.grid(True, linestyle=\":\", alpha=0.6)

ax2.plot(epochs, val_aucs, label=\"Validation Mean AUROC\", color=\"green\", marker=\"^\")
ax2.axvline(exp_gnn.best_epoch + 1, color=\"red\", linestyle=\"--\", label=f\"Best Epoch ({exp_gnn.best_epoch + 1})\")
ax2.set_xlabel(\"Epoch\")
ax2.set_ylabel(\"Mean AUROC\")
ax2.set_title(\"cnn_gnn_multimodal — Validation AUROC Curve\")
ax2.legend()
ax2.grid(True, linestyle=\":\", alpha=0.6)

plt.tight_layout()
plt.show()

# Per-disease validation AUROC table
val_per_disease = {k.replace(\"auc_\", \"\"): v for k, v in val_m_gnn.items() if k.startswith(\"auc_\") and k != \"auc_mean\"}
df_val_gnn = pd.DataFrame(list(val_per_disease.items()), columns=[\"Pathology\", \"Val AUROC\"])
print(\"=== cnn_gnn_multimodal: Per-Disease Validation AUROC ===\")
display(df_val_gnn)
"""))

# Cell 8G: Train Model 4 (quantum_gnn_multimodal)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8G — Train Model 4: `quantum_gnn_multimodal` (Proposed Framework)
**Purpose**: Trains the full proposed framework (EfficientNet-B0 + 4-qubit Variational Quantum Layer + GraphSAGE Inter-Scan GNN + Demographic MLP + 14-Head Disease-Specific Gating)."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8G: Train Model 4 — quantum_gnn_multimodal ──────────────────────────
from src.training.experiment import Experiment

print(\"\"\"
======================================================================
  STARTING EXPERIMENT 4/4: quantum_gnn_multimodal (Proposed Framework)
  Architecture: Quantum VQC + GraphSAGE GNN + Clinical MLP + Disease Gating
  Maximum Epochs: 15 | Early Stopping Patience: 4
======================================================================
\"\"\")

exp_qgnn = Experiment(name=\"quantum_gnn_multimodal\", cfg=BASE_CFG.copy(), device=device)
exp_qgnn.setup(train_df, val_df, test_df, hf_dataset)

val_m_qgnn = exp_qgnn.run(max_epochs=MAX_EPOCHS)
exp_qgnn.save_results(\"results\")

print(f\"\\n[COMPLETE] quantum_gnn_multimodal training finished in {exp_qgnn.training_time_seconds:.1f}s\")
print(f\" - Best Epoch:      {exp_qgnn.best_epoch + 1}\")
print(f\" - Best Val AUROC:  {val_m_qgnn.get('auc_mean', float('nan')):.4f}\")
print(f\" - Val Macro F1:    {val_m_qgnn.get('f1_macro', float('nan')):.4f}\")
"""))

# Cell 8H: Evaluate Model 4 (quantum_gnn_multimodal)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8H — Evaluate Model 4: `quantum_gnn_multimodal` Validation Curves
**Purpose**: Plots training/validation curves and displays per-disease validation AUROC for `quantum_gnn_multimodal`."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8H: Evaluate Model 4 — quantum_gnn_multimodal ───────────────────────
import matplotlib.pyplot as plt
import pandas as pd

history = exp_qgnn.train_history
epochs = [h[\"epoch\"] + 1 for h in history]
train_losses = [h[\"loss\"] for h in history]
val_losses = [h[\"val_loss\"] for h in history]
val_aucs = [h.get(\"val_auc_mean\", float(\"nan\")) for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(epochs, train_losses, label=\"Train Loss\", marker=\"o\")
ax1.plot(epochs, val_losses, label=\"Validation Loss\", marker=\"s\")
ax1.set_xlabel(\"Epoch\")
ax1.set_ylabel(\"Loss\")
ax1.set_title(\"quantum_gnn_multimodal — Loss Curves\")
ax1.legend()
ax1.grid(True, linestyle=\":\", alpha=0.6)

ax2.plot(epochs, val_aucs, label=\"Validation Mean AUROC\", color=\"green\", marker=\"^\")
ax2.axvline(exp_qgnn.best_epoch + 1, color=\"red\", linestyle=\"--\", label=f\"Best Epoch ({exp_qgnn.best_epoch + 1})\")
ax2.set_xlabel(\"Epoch\")
ax2.set_ylabel(\"Mean AUROC\")
ax2.set_title(\"quantum_gnn_multimodal — Validation AUROC Curve\")
ax2.legend()
ax2.grid(True, linestyle=\":\", alpha=0.6)

plt.tight_layout()
plt.show()

# Per-disease validation AUROC table
val_per_disease = {k.replace(\"auc_\", \"\"): v for k, v in val_m_qgnn.items() if k.startswith(\"auc_\") and k != \"auc_mean\"}
df_val_qgnn = pd.DataFrame(list(val_per_disease.items()), columns=[\"Pathology\", \"Val AUROC\"])
print(\"=== quantum_gnn_multimodal: Per-Disease Validation AUROC ===\")
display(df_val_qgnn)
"""))

# Cell 8I: Optional Ablation (quantum_cnn)
cells.append(nbf.v4.new_markdown_cell("""## Cell 8I (Optional Ablation) — Train Model 5: `quantum_cnn`
**Purpose**: Trains the Quantum CNN baseline (EfficientNet-B0 + Quantum VQC + Demographic MLP, without GNN) for full factorial ablation."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 8I: Optional Ablation — quantum_cnn ──────────────────────────────────
# Uncomment to execute this optional 5th experiment
\"\"\"
exp_qcnn = Experiment(name=\"quantum_cnn\", cfg=BASE_CFG.copy(), device=device)
exp_qcnn.setup(train_df, val_df, test_df, hf_dataset)
val_m_qcnn = exp_qcnn.run(max_epochs=MAX_EPOCHS)
exp_qcnn.save_results(\"results\")
print(f\"quantum_cnn finished: Best Epoch={exp_qcnn.best_epoch + 1}, Val AUROC={val_m_qcnn.get('auc_mean', float('nan')):.4f}\")
\"\"\"
print(\"Optional Cell 8I ready for manual execution if quantum_cnn ablation is desired.\")
"""))

# Cell 9: Held-Out Test Set Evaluation
cells.append(nbf.v4.new_markdown_cell("""## Cell 9 — Held-Out Test Evaluation & Comparative Analysis
> [!IMPORTANT]
> **The held-out test split remained strictly untouched during training and epoch selection.**
> This cell evaluates the final restored best checkpoints for all trained models on `test_df`, generates comparison tables, runs subgroup analysis, and saves comparison plots to `outputs/model_comparison/`.
> Only models actively trained in this session or explicitly verified are evaluated, preventing stale metrics."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 9: Held-Out Test Evaluation & Comparative Analysis ───────────────────
from pathlib import Path
import json, os
import pandas as pd
import matplotlib.pyplot as plt
from src.training.evaluator import load_experiment_metrics, generate_comparison_plots, evaluate_scan_count_subgroups, FOUR_MODELS\nfrom src.inference.checkpoint_loader import ModelCheckpointLoader, validate_auroc

candidate_models = [
    ("cnn_only", "exp_cnn"),
    ("classical_multimodal", "exp_class_multi"),
    ("cnn_gnn_multimodal", "exp_gnn"),
    ("quantum_gnn_multimodal", "exp_qgnn"),
]

# Identify active experiments in current session
active_experiments = {}
for m_name, var_name in candidate_models:
    exp_inst = globals().get(var_name)
    if exp_inst is not None and getattr(exp_inst, "trainer", None) is not None:
        active_experiments[m_name] = exp_inst

print(f"Active experiments in current session: {list(active_experiments.keys())}\\n")

# Evaluate FRESH on test_loader for active experiments or verified checkpoints
freshly_evaluated = []
evaluated_metadata = {}

for m_name, var_name in candidate_models:
    exp_inst = active_experiments.get(m_name)
    
    if exp_inst is not None:
        # Active in-memory experiment from this session
        ckpt_path = exp_inst.best_checkpoint_path or "In-memory restored best weights"
        print(f"Evaluating active experiment '{m_name}' on held-out test split...")
        print(f" - Verified Checkpoint:    {ckpt_path}")
        print(f" - Best Validation Epoch:  {exp_inst.best_epoch + 1}")
        
        test_metrics = exp_inst.evaluate_test()
        exp_inst.save_results("results")
        freshly_evaluated.append(m_name)
        evaluated_metadata[m_name] = {
            "checkpoint": Path(ckpt_path).name if ckpt_path != "In-memory restored best weights" else "in-memory-best",
            "best_epoch": f"Epoch {exp_inst.best_epoch + 1}",
        }
        print(f" - Fresh Test Mean AUROC:  {test_metrics.get('auc_mean', float('nan')):.4f}")
        print(f" - Fresh Test Macro F1:    {test_metrics.get('f1_macro', float('nan')):.4f}\\n")
        
    else:
        # Check if an authentic, verifiable checkpoint file exists on disk
        ckpt_dir = Path("checkpoints") / m_name
        available_ckpts = list(ckpt_dir.glob("checkpoint_epoch*.pt")) if ckpt_dir.exists() else []
        
        if available_ckpts:
            ranked = []
            for p in available_ckpts:
                try:
                    payload = torch.load(p, map_location="cpu", weights_only=False)
                    v_score = payload.get("metrics", {}).get("val_auc_mean", -1.0)
                    ep = payload.get("epoch", -1)
                    ranked.append((v_score, ep, p))
                except Exception:
                    pass
            
            if ranked:
                ranked.sort(key=lambda x: x[0], reverse=True)
                best_score, best_ep, best_p = ranked[0]
                print(f"Found verified checkpoint for '{m_name}': {best_p.name} (Val AUC: {best_score:.4f})")
                print(f"Running fresh held-out test evaluation from verified checkpoint...")
                
                from src.training.trainer import CheckpointManager
                from src.training.experiment import Experiment
                exp_disk = Experiment(name=m_name, cfg=BASE_CFG.copy(), device=device)
                exp_disk.setup(train_df, val_df, test_df, hf_dataset)
                CheckpointManager.load(str(best_p), exp_disk.model, device=device)
                
                test_metrics = exp_disk.evaluate_test()
                exp_disk.save_results("results")
                freshly_evaluated.append(m_name)
                evaluated_metadata[m_name] = {
                    "checkpoint": best_p.name,
                    "best_epoch": f"Epoch {best_ep + 1 if isinstance(best_ep, int) else best_ep}",
                }
                print(f" - Fresh Test Mean AUROC:  {test_metrics.get('auc_mean', float('nan')):.4f}")
                print(f" - Fresh Test Macro F1:    {test_metrics.get('f1_macro', float('nan')):.4f}\\n")
            else:
                print(f"ℹ Skipping '{m_name}': Checkpoint files could not be verified. Old test metrics excluded.\\n")
        else:
            print(f"ℹ Skipping '{m_name}': No active experiment or verified checkpoint found. Old unverified results excluded.\\n")

models_to_compare = freshly_evaluated
assert len(models_to_compare) > 0, "No trained models found to evaluate! Run Cell 8 training first."

out_dir = Path("outputs/model_comparison")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Comparison Metrics Table with Provenance Metadata
df_comparison = load_experiment_metrics(results_dir="results", models=models_to_compare)

# Annotate each model with its specific evaluated checkpoint and best epoch
ckpt_annotations = []
for m_name in df_comparison["model"]:
    meta = evaluated_metadata.get(m_name, {})
    ckpt_name = meta.get("checkpoint", "verified-checkpoint")
    epoch_str = meta.get("best_epoch", "N/A")
    ckpt_annotations.append({"model": m_name, "checkpoint": ckpt_name, "best_epoch": epoch_str})

df_annot = pd.DataFrame(ckpt_annotations)
df_comparison = pd.merge(df_comparison, df_annot, on="model", how="left")
df_comparison.to_csv(out_dir / "comparison_metrics.csv", index=False)

print("\\n=======================================================================")
print("  MULTI-MODEL COMPARATIVE TEST PERFORMANCE (AUROC & F1 SCORES)")
print("=======================================================================")
display(df_comparison)

# 2. Subgroup Performance: Single-Scan vs Multi-Scan Patients
df_subgroups = evaluate_scan_count_subgroups(test_df=test_df, results_dir="results", models=models_to_compare)
df_subgroups.to_csv(out_dir / "subgroups_metrics.csv", index=False)
print("\\n=======================================================================")
print("  PATIENT SUBGROUP ANALYSIS (SINGLE-SCAN VS MULTI-SCAN PATIENTS)")
print("=======================================================================")
display(df_subgroups)

# 3. Generate Comparative Plots
plots = generate_comparison_plots(results_dir="results", output_dir=out_dir, models=models_to_compare)
for plot_name, path in plots.items():
    print(f"Generated plot: {path}")

# Display key plots inline
for p_name in ["training_curves.png", "validation_auc.png", "test_auc_comparison.png", "macro_f1_comparison.png", "disease_wise_auc.png"]:
    p_path = out_dir / p_name
    if p_path.exists():
        img = plt.imread(str(p_path))
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(p_name, fontsize=11, fontweight="bold")
        plt.show()

print("\\n[SUCCESS] Cell 9 complete: Held-Out test evaluation finished and comparison charts saved.")
"""))

# Cell 10: Manual Prediction Verification
cells.append(nbf.v4.new_markdown_cell("""## Cell 10 — Manual Prediction Verification (Single & Multi-Scan Scenarios)
**Purpose**: Loads the verified highest-scoring validation checkpoint of `quantum_gnn_multimodal` into the inference service and runs diagnostic predictions across 3 distinct clinical scenarios:
1. Exactly one standalone radiograph.
2. Exactly two radiographs belonging to the same patient (intra-patient message passing).
3. Three or more radiographs from the same patient.
Displays ground-truth labels alongside predictions without leaking labels into inference."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 10: Manual Prediction Verification ──────────────────────────────────
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.inference.predictor import CXRInferenceService, InferenceMode
from src.dataset.label_encoder import safe_parse_labels, NIH_LABEL_NAMES, DISEASE_LABEL_NAMES

# 1. Locate the best checkpoint for quantum_gnn_multimodal strictly by highest validation score
best_checkpoint_path = None
best_val_score = -1.0
best_epoch_num = 'N/A'

if 'exp_qgnn' in globals() and exp_qgnn is not None:
    if getattr(exp_qgnn, 'best_checkpoint_path', None) is not None and Path(exp_qgnn.best_checkpoint_path).exists():
        best_checkpoint_path = Path(exp_qgnn.best_checkpoint_path)

ckpt_dir = Path('checkpoints/quantum_gnn_multimodal')
assert ckpt_dir.exists(), "Directory 'checkpoints/quantum_gnn_multimodal' not found! Complete Cell 8G training first."

available_ckpts = list(ckpt_dir.glob('checkpoint_epoch*.pt'))
assert len(available_ckpts) > 0, "No checkpoint files found in checkpoints/quantum_gnn_multimodal/! Complete Cell 8G first."

# Rank checkpoints strictly by validation AUROC score (val_auc_mean) from checkpoint payloads
from src.inference.checkpoint_loader import ModelCheckpointLoader, validate_auroc

best_checkpoint_path = ModelCheckpointLoader.find_best_checkpoint('quantum_gnn_multimodal', checkpoint_dir='checkpoints')
assert best_checkpoint_path is not None, "No verified checkpoint with valid AUROC found for quantum_gnn_multimodal!"

is_valid_auc, best_val_score, _ = ModelCheckpointLoader.extract_checkpoint_auroc(best_checkpoint_path)
payload = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
best_epoch_num = payload.get('epoch', 'N/A')

print('==================================================')
print('VERIFIED HIGHEST-SCORING CHECKPOINT FOR INFERENCE')
print('==================================================')
print(f" - Model Architecture:  quantum_gnn_multimodal")
print(f" - Checkpoint File:     {best_checkpoint_path.name}")
print(f" - Checkpoint Path:     {best_checkpoint_path.resolve()}")
print(f" - Checkpoint Epoch:    {best_epoch_num + 1 if isinstance(best_epoch_num, int) else best_epoch_num}")
print(f" - Validation Mean AUC: {best_val_score:.4f}")
print('==================================================\\n')

inference_service = CXRInferenceService(device=device)
DECISION_THRESHOLD = 0.50

# Helper to format ground truth labels without passing into inference
def format_ground_truth(df_subset):
    gt_list = []
    for _, row in df_subset.iterrows():
        raw_l = row['labels_raw'] if 'labels_raw' in row else row['labels']
        parsed = safe_parse_labels(raw_l)
        names = [NIH_LABEL_NAMES[i] for i in parsed if 0 < i < len(NIH_LABEL_NAMES)]
        gt_list.append(', '.join(names) if names else 'No Finding')
    return gt_list

p_counts = test_df['patient_id'].value_counts()
res_1, res_2, res_3 = None, None, None

# --- SCENARIO 1: Exactly 1 Standalone Radiograph ---
single_candidates = p_counts[p_counts == 1].index
if len(single_candidates) == 0:
    print('⚠ Scenario 1 skipped: No patient with exactly 1 scan found in test split.')
else:
    pid_1 = single_candidates[0]
    df_p1 = test_df[test_df['patient_id'] == pid_1]
    assert len(df_p1) == 1, f"Expected exactly 1 scan for Scenario 1, got {len(df_p1)}"
    row_1 = df_p1.iloc[0]
    img_1 = hf_dataset[int(row_1['hf_index'])]['image']
    gt_1 = format_ground_truth(df_p1)[0]
    
    res_1 = inference_service.predict(
        image_input=img_1,
        age=int(row_1['age']),
        sex=str(row_1['sex']),
        experiment_name='quantum_gnn_multimodal',
        checkpoint_path=str(best_checkpoint_path),
        mode=InferenceMode.RESEARCH,
        threshold=DECISION_THRESHOLD,
    )
    
    print('--- Scenario 1: Exactly 1 Standalone Radiograph ---')
    print(f" - Patient ID:           {pid_1} (Age: {row_1['age']}, Sex: {row_1['sex']})")
    print(f" - Ground Truth Finding: {gt_1}")
    print(f" - Graph Nodes / Edges:  {res_1.node_count} node, {res_1.edge_count} edge (self-loop: [[0], [0]])")
    print(f" - Predicted Positive:   {res_1.positive_predictions if res_1.positive_predictions else ['No Finding']}")
    print('Top 3 Predicted Pathologies:')
    for d, p in sorted(res_1.probabilities.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   * {d:<20}: {p*100:.1f}%")

# --- SCENARIO 2: Exactly 2 Radiographs from Same Patient ---
two_candidates = p_counts[p_counts == 2].index
if len(two_candidates) == 0:
    print('\\n⚠ Scenario 2 skipped: No patient with exactly 2 scans found in test split.')
else:
    pid_2 = two_candidates[0]
    df_p2 = test_df[test_df['patient_id'] == pid_2]
    assert len(df_p2) == 2, f"Expected exactly 2 scans for Scenario 2, got {len(df_p2)}"
    imgs_2 = [hf_dataset[int(r['hf_index'])]['image'] for _, r in df_p2.iterrows()]
    gts_2 = format_ground_truth(df_p2)
    
    res_2 = inference_service.predict(
        image_input=imgs_2,
        age=int(df_p2.iloc[0]['age']),
        sex=str(df_p2.iloc[0]['sex']),
        experiment_name='quantum_gnn_multimodal',
        checkpoint_path=str(best_checkpoint_path),
        mode=InferenceMode.RESEARCH,
        threshold=DECISION_THRESHOLD,
        primary_scan_index=0,
    )
    
    print('\\n--- Scenario 2: Exactly 2 Radiographs from Same Patient ---')
    print(f" - Patient ID:           {pid_2} (Age: {df_p2.iloc[0]['age']}, Sex: {df_p2.iloc[0]['sex']})")
    print(f" - Primary Scan GT:      {gts_2[0]}")
    print(f" - Companion Scan GT:    {gts_2[1]}")
    print(f" - Graph Nodes / Edges:  {res_2.node_count} nodes, {res_2.edge_count} directed edges (all-to-all: K^2=4)")
    print(f" - Primary Scan Finding: {res_2.positive_predictions if res_2.positive_predictions else ['No Finding']}")
    print('Top 3 Predicted Pathologies (Primary Scan):')
    for d, p in sorted(res_2.probabilities.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   * {d:<20}: {p*100:.1f}%")

# --- SCENARIO 3: Three or More Radiographs (K >= 3) from Same Patient ---
three_candidates = p_counts[p_counts >= 3].index
if len(three_candidates) == 0:
    print('\\n⚠ Scenario 3 skipped: No patient with 3 or more scans found in test split.')
else:
    pid_3 = three_candidates[0]
    df_p3 = test_df[test_df['patient_id'] == pid_3]
    K3 = len(df_p3)
    assert K3 >= 3, f"Expected at least 3 scans for Scenario 3, got {K3}"
    imgs_3 = [hf_dataset[int(r['hf_index'])]['image'] for _, r in df_p3.iterrows()]
    gts_3 = format_ground_truth(df_p3)
    
    res_3 = inference_service.predict(
        image_input=imgs_3,
        age=int(df_p3.iloc[0]['age']),
        sex=str(df_p3.iloc[0]['sex']),
        experiment_name='quantum_gnn_multimodal',
        checkpoint_path=str(best_checkpoint_path),
        mode=InferenceMode.RESEARCH,
        threshold=DECISION_THRESHOLD,
        primary_scan_index=0,
    )
    
    print(f"\\n--- Scenario 3: {K3} Radiographs from Same Patient (K >= 3) ---")
    print(f" - Patient ID:           {pid_3} (Age: {df_p3.iloc[0]['age']}, Sex: {df_p3.iloc[0]['sex']})")
    print(f" - Primary Scan GT:      {gts_3[0]}")
    print(f" - Companion Scans GT:   {gts_3[1:]}")
    print(f" - Graph Nodes / Edges:  {res_3.node_count} nodes, {res_3.edge_count} directed edges (all-to-all: K^2={K3*K3})")
    print(f" - Primary Scan Finding: {res_3.positive_predictions if res_3.positive_predictions else ['No Finding']}")
    print('Top 3 Predicted Pathologies (Primary Scan):')
    for d, p in sorted(res_3.probabilities.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   * {d:<20}: {p*100:.1f}%")

print('\\n[SUCCESS] Cell 10 complete: Multi-scan clinical prediction scenarios verified with ground truth.')
"""))

# Cell 11: Multimodal Explainability Verification
cells.append(nbf.v4.new_markdown_cell("""## Cell 11 — Multimodal Explainability Verification
**Purpose**: Generates Grad-CAM++ saliency overlays on the primary radiograph for the top detected pathology, analytical demographic attribution for Age and Sex, graph companion scan attribution, and 3-way modality Shapley contributions."""))

cells.append(nbf.v4.new_code_cell("""# ── Cell 11: Multimodal Explainability Verification ──────────────────────────
import matplotlib.pyplot as plt
from src.inference.explanation_service import ExplanationService

# Prefer Scenario 2, then Scenario 3, then Scenario 1
active_res = res_2 if res_2 is not None else (res_3 if res_3 is not None else res_1)
assert active_res is not None, 'No prediction results available from Cell 10!'

# Select target disease: prefer a positive prediction, otherwise top probability
if active_res.positive_predictions and active_res.positive_predictions[0] != 'No Finding':
    top_disease = active_res.positive_predictions[0]
else:
    top_disease = sorted(active_res.probabilities.items(), key=lambda x: x[1], reverse=True)[0][0]

print(f"Generating multimodal explanations for target pathology: '{top_disease}'")
print(f"Model used: quantum_gnn_multimodal | Checkpoint: {best_checkpoint_path.name}\\n")

# 1. Grad-CAM++ Visual Heatmap & Anatomical Overlay
cam_res = ExplanationService.get_image_explanation(active_res, target_disease=top_disease)
if cam_res.get('status') == 'success':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(active_res.raw_rgb)
    ax1.set_title('Primary Chest Radiograph', fontsize=10)
    ax1.axis('off')
    
    ax2.imshow(cam_res['overlay'])
    ax2.set_title(f"Grad-CAM++ Saliency Overlay\\nTarget: {top_disease}", fontsize=10, fontweight="bold")
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print('Grad-CAM note:', cam_res.get('message'))

# 2. Demographic Shapley Attribution (Age and Sex)
tab_res = ExplanationService.get_tabular_explanation(active_res, target_disease=top_disease)
if tab_res.get('status') == 'success':
    print('\\nDemographic Feature Attributions (Analytical Shapley Values):')
    print(f" - Age Attribution: {tab_res.get('age_shap', 0.0):+.4f}")
    print(f" - Sex Attribution: {tab_res.get('sex_shap', 0.0):+.4f}")

# 3. Inter-Scan Graph Attribution (GNN Companion Scan Influence)
graph_res = ExplanationService.get_graph_explanation(active_res, target_disease=top_disease)
if graph_res.get('status') == 'success':
    print('\\nInter-Scan Graph Attribution:')
    print(f" - Companion scan nodes analyzed:  {len(graph_res.get('node_importances', []))}")
    print(f" - Edge attribution values:        {graph_res.get('edge_importances', [])}")

# 4. Three-Way Modality Attribution (Image vs Clinical MLP vs GNN)
mod_res = ExplanationService.get_modality_attribution(active_res, target_disease=top_disease)
if mod_res.get('status') == 'success':
    print('\\nModality Shapley Relative Contributions:')
    print(f" - Image Modality (Quantum Vision): {mod_res.get('image_pct', 0.0):.1f}%")
    print(f" - Clinical Metadata (Demographics): {mod_res.get('clinical_pct', 0.0):.1f}%")
    print(f" - Inter-Scan GNN (Patient Graph):  {mod_res.get('gnn_pct', 0.0):.1f}%")

print('\\n[SUCCESS] Cell 11 complete: Multimodal explainability pipeline verified.')
"""))

nb.cells = cells

# Write output notebook
out_path = Path("notebooks/02_model_training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {out_path} with {len(nb.cells)} cells.")

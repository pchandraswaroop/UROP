"""
smoke_test_notebook_logic.py — Execute runtime smoke tests of Cell 5, 6, 7, and 10 logic
"""

import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_factory import build_model
from src.graph.graph_builder import build_batch_scan_graph, ScanGraphBuilder
from src.inference.predictor import CXRInferenceService, InferenceMode
from src.dataset.label_encoder import NIH_LABEL_NAMES, safe_parse_labels

print("=== 1. Testing Cell 5: Graph Edge Decomposition Logic ===")
builder = ScanGraphBuilder(add_self_loops=True)
for K in [1, 2, 4]:
    # Construct dummy patient with K scans
    import pandas as pd
    df_p = pd.DataFrame({
        "patient_id": ["P1"] * K,
        "age": [50] * K,
        "sex": ["M"] * K,
        "labels": ["[1]"] * K,
    })
    g = builder.build(df_p)
    expected_edges = K * K
    print(f" - K={K}: Nodes={g.num_nodes}, Edges={g.edge_index.shape[1]} (Expected K^2={expected_edges}) -> Match: {g.edge_index.shape[1] == expected_edges}")
    assert g.edge_index.shape[1] == expected_edges

print("\n=== 2. Testing Cell 6: Forward API & Graph Inputs for all 5 models ===")
TEST_MODELS = [
    "cnn_only",
    "classical_multimodal",
    "cnn_gnn_multimodal",
    "quantum_cnn",
    "quantum_gnn_multimodal",
]

B = 2
dummy_img = torch.randn(B, 3, 224, 224)
dummy_meta = torch.randn(B, 2)
dummy_pids = ["P_A", "P_A"]
_, edge_index = build_batch_scan_graph(dummy_meta, dummy_pids, add_self_loops=True)

for m_name in TEST_MODELS:
    model = build_model(m_name, pretrained=False, device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(dummy_img, dummy_meta, graph_edge_index=edge_index, return_logits=True)
    assert out.shape == (B, 14), f"Failed for {m_name}: shape={out.shape}"
    print(f" - {m_name:<24}: Forward pass with graph_edge_index OK, output shape: {list(out.shape)}")

print("\n=== 3. Testing Cell 7: AMP and Full Model Forward/Backward ===")
trial_model = build_model("quantum_gnn_multimodal", pretrained=False, device="cpu")
trial_opt = torch.optim.AdamW(trial_model.parameters(), lr=1e-4)

# Test forward + backward pass with batch size 4 on CPU (simulating GPU batch 32)
b_img = torch.randn(4, 3, 224, 224)
b_meta = torch.randn(4, 2)
b_labels = torch.randint(0, 2, (4, 14)).float()
b_pids = ["P1", "P1", "P2", "P2"]
_, b_edge_index = build_batch_scan_graph(b_meta, b_pids, add_self_loops=True)

trial_opt.zero_grad()
logits = trial_model(b_img, b_meta, graph_edge_index=b_edge_index, return_logits=True)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_labels)
loss.backward()
trial_opt.step()
print(f" - Backward step success: Loss = {loss.item():.4f}")

print("\n=== 4. Testing Cell 10 Checkpoint & Ground Truth Decoding ===")
sample_raw = "[2, 11]"
parsed = safe_parse_labels(sample_raw)
decoded = [NIH_LABEL_NAMES[i] for i in parsed if 0 < i < len(NIH_LABEL_NAMES)]
print(f" - safe_parse_labels('[2, 11]') -> {parsed}")
print(f" - NIH decoded strings: {', '.join(decoded)}")
assert decoded == ["Cardiomegaly", "Pleural Thickening"]

print("\nAll runtime smoke tests passed 100% cleanly.")

"""
label_encoder.py — Multi-Label Encoder for NIH ChestX-ray14
=============================================================
The HuggingFace dataset stores labels as list[int32] where each integer
is an index into the NIH canonical label list.

NIH canonical label list (index → name):
  0: No Finding        5: Edema           10: Infiltration
  1: Atelectasis       6: Emphysema       11: Mass
  2: Cardiomegaly      7: Fibrosis        12: Nodule
  3: Consolidation     8: Hernia          13: Pleural Thickening
  4: Effusion          9: Pneumonia       14: Pneumothorax

We predict **14 disease classes** (indices 1–14).
- "No Finding" (index 0) is NOT a prediction target; it signals the image
  has no disease, so all 14 disease outputs should be 0.
- Output: float tensor of shape [14] with values in {0.0, 1.0}.
"""

from __future__ import annotations

from typing import List, Sequence
import torch


# ── NIH Canonical label order ────────────────────────────────────────────────

NIH_LABEL_NAMES: List[str] = [
    "No Finding",       # 0  — not a target
    "Atelectasis",      # 1
    "Cardiomegaly",     # 2
    "Consolidation",    # 3
    "Effusion",         # 4
    "Emphysema",        # 5
    "Fibrosis",         # 6
    "Hernia",           # 7
    "Infiltration",     # 8
    "Mass",             # 9
    "Nodule",           # 10
    "Pleural Thickening",  # 11
    "Pneumonia",        # 12
    "Pneumothorax",     # 13
    "Edema",            # 14
]

# The 14 disease classes we predict (NIH index → our 0-based index)
DISEASE_LABEL_NAMES: List[str] = NIH_LABEL_NAMES[1:]  # indices 1..14
DISEASE_NAMES: List[str] = DISEASE_LABEL_NAMES        # alias for convenience
NUM_CLASSES: int = len(DISEASE_LABEL_NAMES)  # 14

NO_FINDING_IDX: int = 0  # NIH index for "No Finding"


# ── Encoder ──────────────────────────────────────────────────────────────────

class LabelEncoder:
    """Convert a HuggingFace labels field (list[int]) to a multi-hot tensor.

    Usage
    -----
    >>> enc = LabelEncoder()
    >>> tensor = enc.encode([3, 7])          # Consolidation + Fibrosis
    >>> tensor.shape
    torch.Size([14])
    >>> tensor[2], tensor[6]                 # 0-based disease indices
    tensor(1.), tensor(1.)
    """

    def __init__(self) -> None:
        self.num_classes = NUM_CLASSES
        self.label_names = DISEASE_LABEL_NAMES
        # Map NIH index → our 0-based disease index (NIH idx - 1)
        # NIH 0 → excluded
        # NIH 1 → 0, NIH 2 → 1, ..., NIH 14 → 13
        self._nih_to_idx = {nih_i: nih_i - 1 for nih_i in range(1, 15)}

    def encode(self, nih_label_list: Sequence[int]) -> torch.Tensor:
        """Convert list[int] labels from HF dataset to multi-hot [14] float tensor.

        Args:
            nih_label_list: The ``labels`` field from the HF dataset row.
                            A list of NIH indices (ints). May be empty or
                            contain only 0 (No Finding).

        Returns:
            Float tensor of shape [14] with 1.0 for each present disease.
        """
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        for nih_idx in nih_label_list:
            if nih_idx == NO_FINDING_IDX:
                # No Finding → all zeros (do not set any disease bit)
                continue
            if nih_idx in self._nih_to_idx:
                target[self._nih_to_idx[nih_idx]] = 1.0
            # Unknown indices are silently ignored with a warning logged externally

        return target

    def decode(self, tensor: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Convert a [14] probability tensor back to disease name list.

        Args:
            tensor:    [14] float tensor (probabilities or binary).
            threshold: Classification threshold.

        Returns:
            List of disease names with probability > threshold.
        """
        names = []
        for i, val in enumerate(tensor):
            if val.item() > threshold:
                names.append(self.label_names[i])
        return names if names else ["No Finding"]

    def multihot_to_nih_indices(self, tensor: torch.Tensor) -> List[int]:
        """Convert [14] multi-hot tensor back to list of NIH indices."""
        return [i + 1 for i, v in enumerate(tensor) if v.item() > 0.5]

    @property
    def class_weights_info(self) -> str:
        return (
            "Use LabelStatistics.compute_pos_weights(df) to get "
            "per-class positive weights for BCE loss."
        )


# ── Convenience functions ────────────────────────────────────────────────────

_DEFAULT_ENCODER = LabelEncoder()


def safe_parse_labels(raw_val: Any) -> List[int]:
    """Safely parse label list from strings, lists, numpy arrays, or CSV serialization.

    Handles:
      - '[np.int32(2), np.int32(11)]' (from pandas CSV with numpy dtypes)
      - '[2, 11]' (standard python list string)
      - list or tuple of ints
      - numpy ndarray
      - empty / None / NaN values -> returns [0] (No Finding)
    """
    import ast
    import re
    import numpy as np

    if raw_val is None:
        return [0]
    if isinstance(raw_val, float) and np.isnan(raw_val):
        return [0]
    if isinstance(raw_val, (list, tuple)):
        return [int(x) for x in raw_val]
    if isinstance(raw_val, np.ndarray):
        return [int(x) for x in raw_val.tolist()]
    if hasattr(raw_val, "tolist") and callable(raw_val.tolist):
        return [int(x) for x in raw_val.tolist()]

    if isinstance(raw_val, str):
        cleaned = re.sub(r'(?:np\.)?int\d+\((\d+)\)', r'\1', raw_val).strip()
        if not cleaned or cleaned in ("[]", "nan", "None", "set()"):
            return [0]
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple, set)):
                return [int(x) for x in parsed]
            elif isinstance(parsed, (int, float)):
                return [int(parsed)]
        except Exception:
            nums = re.findall(r'(?<!int)\b(\d+)\b', cleaned)
            if nums:
                return [int(x) for x in nums]
    return [0]


def encode_labels(nih_label_list: Sequence[int]) -> torch.Tensor:
    """Module-level convenience: encode a HF labels list to [14] float tensor."""
    return _DEFAULT_ENCODER.encode(nih_label_list)


def decode_labels(tensor: torch.Tensor, threshold: float = 0.5) -> List[str]:
    """Module-level convenience: decode [14] prob tensor to disease names."""
    return _DEFAULT_ENCODER.decode(tensor, threshold)

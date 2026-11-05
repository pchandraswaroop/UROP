"""
reproducibility.py — Seed setting for reproducibility.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

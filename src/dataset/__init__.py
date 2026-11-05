# src/dataset/__init__.py
# NOTE: Imports are explicit to allow individual module imports without
# requiring all optional dependencies (e.g., `datasets` for hf_loader).

from src.dataset.label_encoder import (
    LabelEncoder, encode_labels, decode_labels, DISEASE_LABEL_NAMES, NUM_CLASSES,
)
from src.dataset.preprocessor import get_transforms, MetadataEncoder

__all__ = [
    "LabelEncoder", "encode_labels", "decode_labels",
    "DISEASE_LABEL_NAMES", "NUM_CLASSES",
    "get_transforms", "MetadataEncoder",
]

# Lazy imports for modules that require optional dependencies
def __getattr__(name):
    if name in ("CXRDataset", "get_dataloader", "get_all_dataloaders"):
        from src.dataset.dataloader import CXRDataset, get_dataloader, get_all_dataloaders
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


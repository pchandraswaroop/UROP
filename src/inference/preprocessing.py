"""
preprocessing.py — Inference Preprocessing & Input Validation
=============================================================
Prepares chest X-ray images, clinical metadata, and patient scan graphs
for model inference using exact transformations matched to training.
"""

from __future__ import annotations

import io
import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.dataset.preprocessor import MetadataEncoder, get_val_transforms
from src.graph.graph_builder import build_batch_scan_graph

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Raised when user-provided inference inputs violate validity constraints."""
    pass


class InferencePreprocessor:
    """Preprocesses images, metadata, and optional scan graphs for inference."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.val_transform = get_val_transforms(image_size=image_size)
        self.meta_encoder = MetadataEncoder()

    @staticmethod
    def validate_inputs(
        image: Any,
        age: Union[int, float, str],
        sex: str,
    ) -> None:
        """Validate input parameters. Accepts single image or list of images."""
        # 1. Image validation
        if image is None:
            raise InputValidationError("No image provided. Please upload a chest X-ray.")

        if isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise InputValidationError("No image provided. Image list is empty.")
            for idx, img in enumerate(image):
                if img is None:
                    raise InputValidationError(f"Scan at index {idx} is None.")
        else:
            if isinstance(image, (bytes, bytearray)) and len(image) == 0:
                raise InputValidationError("Uploaded image byte stream is empty.")

        # 2. Age validation
        try:
            age_int = int(age)
        except (ValueError, TypeError):
            raise InputValidationError(f"Invalid age value: '{age}'. Age must be an integer between 0 and 120.")

        if age_int < 0 or age_int > 120:
            raise InputValidationError(f"Age {age_int} out of valid clinical range [0, 120].")

        # 3. Sex validation
        sex_clean = str(sex).strip().upper()
        if sex_clean not in {"M", "F", "MALE", "FEMALE"}:
            raise InputValidationError(f"Invalid sex value: '{sex}'. Must be Male ('M') or Female ('F').")

    def preprocess_image(self, image_input: Union[Image.Image, bytes, str, io.BytesIO]) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert input to preprocessed tensor and normalized display numpy array.

        Returns:
            tensor: [1, 3, image_size, image_size] normalized float tensor.
            raw_rgb: [image_size, image_size, 3] uint8 numpy array for visual display.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif hasattr(image_input, "read"):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            raise InputValidationError(f"Unsupported image input type: {type(image_input)}")

        # Create display version resized to target resolution
        display_img = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        raw_rgb = np.array(display_img, dtype=np.uint8)

        # Apply deterministic validation transforms matched to training
        img_tensor = self.val_transform(pil_img).unsqueeze(0)  # [1, 3, H, W]
        return img_tensor, raw_rgb

    def preprocess_images(
        self,
        image_inputs: Union[Any, List[Any]],
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Convert one or more image inputs into a batched tensor and display numpy arrays.

        Returns:
            stacked_tensors: [K, 3, image_size, image_size]
            raw_rgbs: List of K [image_size, image_size, 3] uint8 arrays
        """
        if not isinstance(image_inputs, (list, tuple)):
            image_inputs = [image_inputs]

        tensors = []
        raw_rgbs = []
        for img in image_inputs:
            t, rgb = self.preprocess_image(img)
            tensors.append(t)
            raw_rgbs.append(rgb)

        stacked = torch.cat(tensors, dim=0)  # [K, 3, H, W]
        return stacked, raw_rgbs

    def preprocess_metadata(self, age: Union[int, float], sex: str) -> torch.Tensor:
        """Encode clinical metadata into [1, 2] normalized tensor [age_norm, sex_binary]."""
        vec = self.meta_encoder.encode_row(int(age), str(sex))  # [2]
        return vec.unsqueeze(0)  # [1, 2]

    @staticmethod
    def build_single_scan_graph(meta_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a valid single-node scan graph with a self-loop.

        Returns:
            gx: [1, 2] node features
            edge_index: [2, 1] directed edge index (self-loop (0, 0))
        """
        gx = meta_tensor.clone()
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return gx, edge_index

    @staticmethod
    def build_multi_scan_patient_graph(
        num_scans: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Construct a scan graph connecting all uploaded scans of the same patient.

        Requirements:
          - Connects scans belonging to the same patient (all-to-all).
          - Never connects scans across different patients.
          - Includes self-loops for each scan.

        Returns:
            edge_index: [2, num_scans * num_scans] long tensor
        """
        K = max(1, num_scans)
        src = [i for i in range(K) for j in range(K)]
        dst = [j for i in range(K) for j in range(K)]
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    @staticmethod
    def build_patient_history_graph(
        meta_history: List[torch.Tensor],
        patient_id: str = "PATIENT_QUERY",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a scan-to-scan graph for multiple scans belonging to the same patient."""
        N = len(meta_history)
        stacked_meta = torch.cat(meta_history, dim=0)  # [N, 2]
        patient_ids = [patient_id] * N
        return build_batch_scan_graph(stacked_meta, patient_ids, add_self_loops=True)

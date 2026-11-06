"""
predictor.py — Multi-Modal CXR Inference Service
=================================================
Executes multi-label disease predictions across the 14 thoracic disease targets
with strict separation between Research Mode (requires verified checkpoint)
and Demo Mode (explicitly labeled interface preview).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.checkpoint_loader import CheckpointError, ModelCheckpointLoader
from src.inference.preprocessing import InferencePreprocessor, InputValidationError
from src.models.model_factory import build_model

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = (
    "This application is a research prototype for educational and experimental purposes. "
    "It is not a clinically validated diagnostic system and must not be used for medical "
    "diagnosis or treatment decisions."
)


class InferenceMode(str, Enum):
    RESEARCH = "RESEARCH"
    DEMO = "DEMO"


@dataclass
class PredictionResult:
    """Holds output of inference with metadata and safety disclaimers."""
    mode: InferenceMode
    model_name: str
    probabilities: Dict[str, float]  # 14 disease probabilities for primary scan
    positive_predictions: List[str]  # diseases exceeding threshold
    is_no_finding: bool              # True if no disease exceeds threshold
    threshold: float
    checkpoint_info: Optional[Dict[str, Any]]
    device: str
    disclaimer: str = MEDICAL_DISCLAIMER
    raw_logits: Optional[np.ndarray] = None
    node_count: int = 1
    edge_count: int = 1
    is_multi_scan: bool = False
    all_probabilities: Optional[List[Dict[str, float]]] = None
    all_raw_rgbs: Optional[List[np.ndarray]] = None
    primary_scan_index: int = 0
    model_instance: Optional[nn.Module] = None
    image_tensor: Optional[torch.Tensor] = None
    meta_tensor: Optional[torch.Tensor] = None
    graph_x: Optional[torch.Tensor] = None
    graph_edge_index: Optional[torch.Tensor] = None
    raw_rgb: Optional[np.ndarray] = None
    all_image_tensors: Optional[torch.Tensor] = None
    all_meta_tensors: Optional[torch.Tensor] = None

    @property
    def is_demo(self) -> bool:
        return self.mode == InferenceMode.DEMO


class CXRInferenceService:
    """Coordinates input preprocessing, checkpoint verification, and model prediction."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = InferencePreprocessor(image_size=224)
        self._model_cache: Dict[str, nn.Module] = {}

    def get_or_build_model(
        self,
        experiment_name: str,
        checkpoint_path: Optional[str | Path] = None,
        require_checkpoint: bool = True,
    ) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
        """Instantiate model and load checkpoint if in Research Mode."""
        exp = experiment_name.lower().strip()
        model = build_model(exp, pretrained=False, device=self.device)
        model = model.to(self.device)
        model.eval()

        ckpt_info: Optional[Dict[str, Any]] = None

        if require_checkpoint:
            if not checkpoint_path:
                # Attempt to find best available checkpoint for this experiment
                available = ModelCheckpointLoader.list_available_checkpoints()
                exp_ckpts = available.get(exp, [])
                if not exp_ckpts:
                    raise CheckpointError(
                        f"No trained checkpoint found for experiment '{experiment_name}'. "
                        "Real inference requires a valid checkpoint. Please train the model "
                        "on Google Colab or provide a valid .pt file path."
                    )
                checkpoint_path = exp_ckpts[0]

            ckpt_info = ModelCheckpointLoader.load_weights(
                checkpoint_path, model, device=self.device, strict=True
            )
            ckpt_info["path"] = str(checkpoint_path)

        return model, ckpt_info

    def predict(
        self,
        image_input: Any,
        age: Union[int, float, str],
        sex: str,
        experiment_name: str,
        mode: InferenceMode = InferenceMode.RESEARCH,
        checkpoint_path: Optional[str | Path] = None,
        threshold: float = 0.5,
        patient_history: Optional[List[Tuple[Any, int, str]]] = None,
        primary_scan_index: int = 0,
    ) -> PredictionResult:
        """Execute full prediction pipeline.

        Supports both single-scan and multi-scan patient evaluations:
          - If image_input is a single image, evaluates a 1-node graph with self-loop.
          - If image_input is a list of images confirmed to belong to the same patient,
            constructs a temporary patient scan graph with all-to-all edges.

        Args:
            image_input: Single PIL image/path/bytes, or a list of images for one patient.
            age: Patient age (0-120), provided once for the patient.
            sex: Patient sex ('M' or 'F'), provided once for the patient.
            experiment_name: Model architecture experiment variant.
            mode: RESEARCH (requires trained checkpoint) or DEMO (preview mode).
            checkpoint_path: Optional explicit path to checkpoint.
            threshold: Probability threshold for positive classification (default 0.5).
            patient_history: Optional prior scans for longitudinal GNN graphs.
            primary_scan_index: Index of the primary scan to report in main probabilities.

        Returns:
            PredictionResult with probabilities and positive labels.
        """
        # 1. Input Validation
        self.preprocessor.validate_inputs(image_input, age, sex)

        # 2. Image and Metadata Preprocessing
        if isinstance(image_input, (list, tuple)):
            img_tensor, raw_rgbs = self.preprocessor.preprocess_images(image_input)
            num_scans = len(image_input)
            primary_idx = min(max(0, primary_scan_index), num_scans - 1)
            raw_rgb = raw_rgbs[primary_idx]
        else:
            img_tensor, raw_rgb = self.preprocessor.preprocess_image(image_input)
            raw_rgbs = [raw_rgb]
            num_scans = 1
            primary_idx = 0

        img_tensor = img_tensor.to(self.device)  # [K, 3, H, W]
        base_meta = self.preprocessor.preprocess_metadata(age, sex).to(self.device)  # [1, 2]
        meta_tensor = base_meta.repeat(num_scans, 1)  # [K, 2]

        # 3. Patient Scan Graph Preparation
        if num_scans > 1:
            edge_index = self.preprocessor.build_multi_scan_patient_graph(num_scans, device=self.device)
            gx = meta_tensor
        elif patient_history and len(patient_history) > 0:
            hist_tensors = [meta_tensor.cpu()]
            for _, h_age, h_sex in patient_history:
                hist_tensors.append(self.preprocessor.preprocess_metadata(h_age, h_sex).cpu())
            gx, edge_index = self.preprocessor.build_patient_history_graph(hist_tensors)
            gx = gx.to(self.device)
            edge_index = edge_index.to(self.device)
        else:
            gx, edge_index = self.preprocessor.build_single_scan_graph(meta_tensor.cpu())
            gx = gx.to(self.device)
            edge_index = edge_index.to(self.device)

        # 4. Mode-Specific Execution
        if mode == InferenceMode.RESEARCH:
            return self._predict_research(
                img_tensor=img_tensor,
                meta_tensor=meta_tensor,
                gx=gx,
                edge_index=edge_index,
                raw_rgb=raw_rgb,
                raw_rgbs=raw_rgbs,
                num_scans=num_scans,
                primary_idx=primary_idx,
                experiment_name=experiment_name,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
            )
        else:
            return self._predict_demo(
                img_tensor=img_tensor,
                meta_tensor=meta_tensor,
                gx=gx,
                edge_index=edge_index,
                raw_rgb=raw_rgb,
                raw_rgbs=raw_rgbs,
                num_scans=num_scans,
                primary_idx=primary_idx,
                experiment_name=experiment_name,
                threshold=threshold,
            )

    def _predict_research(
        self,
        img_tensor: torch.Tensor,
        meta_tensor: torch.Tensor,
        gx: torch.Tensor,
        edge_index: torch.Tensor,
        raw_rgb: np.ndarray,
        raw_rgbs: List[np.ndarray],
        num_scans: int,
        primary_idx: int,
        experiment_name: str,
        checkpoint_path: Optional[str | Path],
        threshold: float,
    ) -> PredictionResult:
        """Strict research prediction with actual checkpoint weights."""
        model, ckpt_info = self.get_or_build_model(
            experiment_name, checkpoint_path=checkpoint_path, require_checkpoint=True
        )

        with torch.no_grad():
            try:
                logits = model(
                    img_tensor,
                    meta_tensor,
                    graph_x=gx,
                    graph_edge_index=edge_index,
                    return_logits=True,
                )
            except TypeError:
                logits = model(img_tensor, meta_tensor, return_logits=True)

            probs_all = torch.sigmoid(logits).cpu().numpy()  # [K, 14]
            logits_np = logits.cpu().numpy()                 # [K, 14]

        # Primary scan probabilities
        primary_probs = probs_all[primary_idx]
        primary_logits = logits_np[primary_idx]

        prob_dict = {name: float(primary_probs[i]) for i, name in enumerate(DISEASE_LABEL_NAMES)}
        positives = [name for name, p in prob_dict.items() if p >= threshold]
        is_no_finding = len(positives) == 0

        # All scans probabilities list
        all_prob_dicts = [
            {name: float(probs_all[k, i]) for i, name in enumerate(DISEASE_LABEL_NAMES)}
            for k in range(num_scans)
        ]

        return PredictionResult(
            mode=InferenceMode.RESEARCH,
            model_name=experiment_name,
            probabilities=prob_dict,
            positive_predictions=positives,
            is_no_finding=is_no_finding,
            threshold=threshold,
            checkpoint_info=ckpt_info,
            device=self.device,
            raw_logits=primary_logits,
            node_count=num_scans,
            edge_count=edge_index.shape[1],
            is_multi_scan=(num_scans > 1),
            all_probabilities=all_prob_dicts,
            all_raw_rgbs=raw_rgbs,
            primary_scan_index=primary_idx,
            model_instance=model,
            image_tensor=img_tensor[primary_idx:primary_idx+1],
            meta_tensor=meta_tensor[primary_idx:primary_idx+1],
            graph_x=gx,
            graph_edge_index=edge_index,
            raw_rgb=raw_rgb,
            all_image_tensors=img_tensor,
            all_meta_tensors=meta_tensor,
        )

    def _predict_demo(
        self,
        img_tensor: torch.Tensor,
        meta_tensor: torch.Tensor,
        gx: torch.Tensor,
        edge_index: torch.Tensor,
        raw_rgb: np.ndarray,
        raw_rgbs: List[np.ndarray],
        num_scans: int,
        primary_idx: int,
        experiment_name: str,
        threshold: float,
    ) -> PredictionResult:
        """Demo mode with deterministically generated synthetic preview probabilities."""
        model, _ = self.get_or_build_model(
            experiment_name, checkpoint_path=None, require_checkpoint=False
        )

        all_prob_dicts = []
        for k in range(num_scans):
            img_hash = int(hashlib.sha256(raw_rgbs[k].tobytes()).hexdigest()[:8], 16)
            rng = np.random.default_rng(img_hash)
            base_probs = np.array([
                0.12, 0.08, 0.06, 0.14, 0.03, 0.02, 0.01, 0.18, 0.05, 0.07, 0.04, 0.02, 0.05, 0.03
            ])
            noise = rng.uniform(-0.04, 0.05, size=len(base_probs))
            probs_k = np.clip(base_probs + noise, 0.005, 0.95)
            all_prob_dicts.append({name: float(probs_k[i]) for i, name in enumerate(DISEASE_LABEL_NAMES)})

        prob_dict = all_prob_dicts[primary_idx]
        positives = [name for name, p in prob_dict.items() if p >= threshold]
        is_no_finding = len(positives) == 0

        probs_np = np.array([prob_dict[name] for name in DISEASE_LABEL_NAMES])
        return PredictionResult(
            mode=InferenceMode.DEMO,
            model_name=f"{experiment_name} (DEMO PREVIEW)",
            probabilities=prob_dict,
            positive_predictions=positives,
            is_no_finding=is_no_finding,
            threshold=threshold,
            checkpoint_info={"status": "DEMO MODE (No checkpoint loaded)"},
            device=self.device,
            raw_logits=np.log(probs_np / (1.0 - probs_np)),
            node_count=num_scans,
            edge_count=edge_index.shape[1],
            is_multi_scan=(num_scans > 1),
            all_probabilities=all_prob_dicts,
            all_raw_rgbs=raw_rgbs,
            primary_scan_index=primary_idx,
            model_instance=model,
            image_tensor=img_tensor[primary_idx:primary_idx+1],
            meta_tensor=meta_tensor[primary_idx:primary_idx+1],
            graph_x=gx,
            graph_edge_index=edge_index,
            raw_rgb=raw_rgb,
            all_image_tensors=img_tensor,
            all_meta_tensors=meta_tensor,
        )

"""
test_label_encoder.py — Unit tests for LabelEncoder
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.label_encoder import (
    LabelEncoder, encode_labels, decode_labels,
    DISEASE_LABEL_NAMES, NUM_CLASSES, NIH_LABEL_NAMES
)


@pytest.fixture
def encoder():
    return LabelEncoder()


class TestLabelEncoder:
    def test_num_classes(self, encoder):
        assert encoder.num_classes == 14

    def test_no_finding_all_zeros(self, encoder):
        t = encoder.encode([0])
        assert t.sum() == 0, "No Finding should produce all-zero tensor"
        assert t.shape == torch.Size([14])

    def test_empty_labels_all_zeros(self, encoder):
        t = encoder.encode([])
        assert t.sum() == 0
        assert t.shape == torch.Size([14])

    def test_single_disease(self, encoder):
        # NIH idx 1 = Atelectasis → our idx 0
        t = encoder.encode([1])
        assert t[0] == 1.0
        assert t.sum() == 1.0

    def test_multiple_diseases(self, encoder):
        # NIH 1 (Atelectasis) + NIH 14 (Pneumothorax)
        t = encoder.encode([1, 14])
        assert t[0] == 1.0   # Atelectasis → idx 0
        assert t[13] == 1.0  # Pneumothorax → idx 13
        assert t.sum() == 2.0

    def test_all_diseases(self, encoder):
        t = encoder.encode(list(range(1, 15)))
        assert t.sum() == 14.0, "All 14 disease labels should be set"

    def test_mixed_no_finding_and_disease(self, encoder):
        # No Finding (0) + Atelectasis (1)
        t = encoder.encode([0, 1])
        assert t[0] == 1.0
        assert t.sum() == 1.0  # Only disease, No Finding doesn't add any bit

    def test_decode_round_trip(self, encoder):
        labels = [1, 4, 7]
        t = encoder.encode(labels)
        names = encoder.decode(t, threshold=0.5)
        assert "Atelectasis" in names  # NIH 1 → our 0
        assert "No Finding" not in names

    def test_decode_all_zero(self, encoder):
        t = torch.zeros(14)
        names = encoder.decode(t)
        assert names == ["No Finding"]

    def test_dtype_float32(self, encoder):
        t = encoder.encode([1])
        assert t.dtype == torch.float32


class TestConvenienceFunctions:
    def test_encode_labels_function(self):
        t = encode_labels([2])  # Cardiomegaly
        assert t.shape == torch.Size([14])
        assert t[1] == 1.0  # NIH 2 → idx 1

    def test_disease_label_names_count(self):
        assert len(DISEASE_LABEL_NAMES) == 14

    def test_nih_label_names_count(self):
        assert len(NIH_LABEL_NAMES) == 15  # 14 diseases + No Finding

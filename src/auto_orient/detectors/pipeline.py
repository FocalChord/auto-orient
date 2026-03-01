"""Detection pipeline using Deep-OAD model."""

from __future__ import annotations

from typing import List

from . import DetectionResult
from .oad import detect_oad, detect_oad_batch


def detect_orientation(image_path: str, confidence_threshold: float = 0.6) -> DetectionResult:
    """Run OAD orientation detection on a single image."""
    return detect_oad(image_path)


def detect_orientation_batch(image_paths: List[str], confidence_threshold: float = 0.6) -> List[DetectionResult]:
    """Run OAD orientation detection on a batch of images."""
    return detect_oad_batch(image_paths)

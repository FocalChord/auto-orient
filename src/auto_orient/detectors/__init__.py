"""Shared types for orientation detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Orientation(IntEnum):
    """Detected content orientation as clockwise rotation needed to correct."""
    CORRECT = 0      # No rotation needed
    CW_90 = 90       # Content is rotated 90° CCW → rotate 90° CW to fix
    CW_180 = 180     # Content is upside down → rotate 180° to fix
    CCW_90 = 270     # Content is rotated 90° CW → rotate 90° CCW to fix

    @property
    def exif_orientation(self) -> int:
        """Map to EXIF Orientation tag value."""
        return {0: 1, 90: 6, 180: 3, 270: 8}[self.value]

    @property
    def label(self) -> str:
        return {
            0: "Correct",
            90: "90° CW",
            180: "180°",
            270: "90° CCW",
        }[self.value]


@dataclass
class DetectionResult:
    orientation: Orientation
    confidence: float  # 0.0 to 1.0
    method: str        # "face", "edge", "pipeline"
    details: str = ""  # human-readable explanation

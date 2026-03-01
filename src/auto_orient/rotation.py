"""Lossless rotation via exiftool or jpegtran."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .detectors import Orientation


def _check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def rotate_exiftool(image_path: Path, orientation: Orientation) -> bool:
    """Set EXIF Orientation tag using exiftool. Truly lossless."""
    if not _check_tool("exiftool"):
        raise RuntimeError("exiftool not found. Install with: brew install exiftool")

    if orientation == Orientation.CORRECT:
        return False

    exif_val = orientation.exif_orientation
    result = subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            f"-Orientation={exif_val}",
            "-n",  # numeric tag values
            str(image_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def rotate_jpegtran(image_path: Path, orientation: Orientation) -> bool:
    """Lossless DCT rotation using jpegtran. Modifies pixel data."""
    if not _check_tool("jpegtran"):
        raise RuntimeError(
            "jpegtran not found. Install with: brew install libjpeg-turbo"
        )

    if orientation == Orientation.CORRECT:
        return False

    rotation_flag = {
        Orientation.CW_90: "90",
        Orientation.CW_180: "180",
        Orientation.CCW_90: "270",
    }[orientation]

    tmp_path = image_path.with_suffix(".tmp.jpg")

    result = subprocess.run(
        [
            "jpegtran",
            "-rotate", rotation_flag,
            "-copy", "all",    # preserve metadata
            "-outfile", str(tmp_path),
            str(image_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and tmp_path.exists():
        tmp_path.replace(image_path)
        return True
    else:
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def apply_rotation(
    image_path: Path,
    orientation: Orientation,
    method: str = "exiftool",
) -> bool:
    """Apply lossless rotation using the specified method."""
    if method == "exiftool":
        return rotate_exiftool(image_path, orientation)
    elif method == "jpegtran":
        return rotate_jpegtran(image_path, orientation)
    else:
        raise ValueError(f"Unknown rotation method: {method}")

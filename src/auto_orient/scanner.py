"""Recursive JPEG discovery with filtering for scanner output."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ScanFile:
    path: Path
    width: int
    height: int
    roll_folder: str


def discover_jpegs(
    root: Path,
    skip_contact_sheets: bool = True,
) -> List[ScanFile]:
    """Recursively find JPEG files under root, filtering as needed.

    Returns ScanFile objects with basic image dimensions read via Pillow.
    """
    from PIL import Image

    jpeg_extensions = {".jpg", ".jpeg"}
    results: List[ScanFile] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        roll_folder = dirpath_p.name if dirpath_p != root else ""

        for fname in sorted(filenames):
            if Path(fname).suffix.lower() not in jpeg_extensions:
                continue

            if skip_contact_sheets and fname.lower().startswith("contact_sheet"):
                continue

            fpath = dirpath_p / fname

            try:
                with Image.open(fpath) as img:
                    width, height = img.size
            except Exception:
                continue

            results.append(ScanFile(
                path=fpath,
                width=width,
                height=height,
                roll_folder=roll_folder,
            ))

    return results

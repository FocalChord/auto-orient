"""Click CLI for image orientation detection and correction."""

from __future__ import annotations

import os

# Must be set before any TF import
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import List

import click
from rich.console import Console

from .detectors import DetectionResult, Orientation
from .detectors.pipeline import detect_orientation_batch
from .report import write_output
from .rotation import apply_rotation
from .scanner import ScanFile, discover_jpegs

console = Console()

_BATCH_SIZE = 32


@click.command()
@click.argument("path", default="./photos", type=click.Path(exists=True))
@click.option(
    "--apply/--dry-run",
    default=False,
    help="Apply rotation or just report (default: dry-run).",
)
@click.option(
    "--method",
    type=click.Choice(["exiftool", "jpegtran"]),
    default="exiftool",
    help="Rotation method (default: exiftool).",
)
@click.option(
    "--confidence",
    type=float,
    default=0.6,
    help="Min confidence to auto-classify (default: 0.6).",
)
@click.option(
    "--skip-contact-sheets/--include-contact-sheets",
    default=True,
    help="Skip contact_sheet*.jpg files (default: yes).",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format (default: table).",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default=None,
    help="Write results to file.",
)
@click.option(
    "--batch-size",
    type=int,
    default=_BATCH_SIZE,
    help=f"Inference batch size (default: {_BATCH_SIZE}).",
)
def main(
    path: str,
    apply: bool,
    method: str,
    confidence: float,
    skip_contact_sheets: bool,
    output_format: str,
    output_file: str | None,
    batch_size: int,
) -> None:
    """Detect and fix orientation of JPEG images.

    Scans PATH for JPEG files, detects which are rotated, and optionally
    applies lossless rotation to correct them.
    """
    root = Path(path)

    # Discover files
    console.print(f"[bold]Scanning[/bold] {root} for JPEG files...")
    scan_files = discover_jpegs(root, skip_contact_sheets=skip_contact_sheets)

    if not scan_files:
        console.print("[yellow]No JPEG files found.[/yellow]")
        return

    subdirs = set(s.roll_folder for s in scan_files) - {""}
    dir_msg = f" across [bold]{len(subdirs)}[/bold] directories" if subdirs else ""
    console.print(f"Found [bold]{len(scan_files)}[/bold] images{dir_msg}.")

    # Detect orientations via batched OAD inference
    paths: List[Path] = [s.path for s in scan_files]
    rolls: List[str] = [s.roll_folder for s in scan_files]
    str_paths = [str(p) for p in paths]

    console.print("[bold]Loading model...[/bold]")
    detections: List[DetectionResult] = []

    for i in range(0, len(str_paths), batch_size):
        batch = str_paths[i:i + batch_size]
        batch_results = detect_orientation_batch(batch, confidence)
        detections.extend(batch_results)
        console.print(f"  Processed {min(i + batch_size, len(str_paths))}/{len(str_paths)} images")

    # Report
    write_output(
        paths=paths,
        rolls=rolls,
        detections=detections,
        output_format=output_format,
        output_file=output_file,
        confidence_threshold=confidence,
    )

    # Apply rotations if requested
    if apply:
        to_rotate = [
            (paths[i], detections[i])
            for i in range(len(paths))
            if detections[i].orientation != Orientation.CORRECT
            and detections[i].confidence >= confidence
        ]

        if not to_rotate:
            console.print("\n[green]No images need rotation.[/green]")
            return

        console.print(f"\n[bold]Applying {method} rotation to {len(to_rotate)} images...[/bold]")

        success = 0
        for img_path, det in to_rotate:
            try:
                if apply_rotation(img_path, det.orientation, method):
                    success += 1
                    console.print(f"  [green]Rotated[/green] {img_path.name} → {det.orientation.label}")
                else:
                    console.print(f"  [red]Failed[/red] {img_path.name}")
            except Exception as e:
                console.print(f"  [red]Error[/red] {img_path.name}: {e}")

        console.print(f"\n[bold]Done:[/bold] {success}/{len(to_rotate)} images rotated.")
    else:
        would_rotate = sum(
            1 for d in detections
            if d.orientation != Orientation.CORRECT and d.confidence >= confidence
        )
        if would_rotate:
            console.print(f"\n[dim]Run with --apply to rotate {would_rotate} images.[/dim]")

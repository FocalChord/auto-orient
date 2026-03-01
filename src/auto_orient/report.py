"""Output formatting: rich table, JSON, CSV."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .detectors import DetectionResult, Orientation


@dataclass
class ImageResult:
    path: str
    roll: str
    orientation: str
    confidence: float
    method: str
    needs_rotation: bool
    details: str


def _build_results(
    paths: List[Path],
    rolls: List[str],
    detections: List[DetectionResult],
) -> List[ImageResult]:
    results = []
    for path, roll, det in zip(paths, rolls, detections):
        results.append(ImageResult(
            path=str(path),
            roll=roll,
            orientation=det.orientation.label,
            confidence=det.confidence,
            method=det.method,
            needs_rotation=det.orientation != Orientation.CORRECT,
            details=det.details,
        ))
    return results


def format_table(
    paths: List[Path],
    rolls: List[str],
    detections: List[DetectionResult],
    confidence_threshold: float = 0.6,
) -> str:
    """Format results as a rich table printed to console."""
    from rich.console import Console
    from rich.table import Table

    results = _build_results(paths, rolls, detections)

    table = Table(title="Orientation Analysis", show_lines=False)
    table.add_column("File", style="cyan", no_wrap=True, max_width=50)
    table.add_column("Roll", style="blue")
    table.add_column("Orientation", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Method", style="dim")
    table.add_column("Action", style="bold")

    needs_rotation = 0
    uncertain = 0

    for r in results:
        if r.needs_rotation and r.confidence >= confidence_threshold:
            action = f"[red]Rotate {r.orientation}[/red]"
            needs_rotation += 1
        elif r.confidence < confidence_threshold and r.needs_rotation:
            action = "[yellow]Review[/yellow]"
            uncertain += 1
        else:
            action = "[green]OK[/green]"

        conf_color = "green" if r.confidence >= confidence_threshold else "yellow" if r.confidence > 0 else "red"
        conf_str = f"[{conf_color}]{r.confidence:.0%}[/{conf_color}]"

        table.add_row(
            Path(r.path).name,
            r.roll,
            r.orientation,
            conf_str,
            r.method,
            action,
        )

    console = Console(record=True)
    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[bold]Summary:[/bold] {len(results)} images scanned, "
        f"[red]{needs_rotation}[/red] need rotation, "
        f"[yellow]{uncertain}[/yellow] need review, "
        f"[green]{len(results) - needs_rotation - uncertain}[/green] OK"
    )

    return console.export_text()


def format_json(
    paths: List[Path],
    rolls: List[str],
    detections: List[DetectionResult],
) -> str:
    results = _build_results(paths, rolls, detections)
    return json.dumps([asdict(r) for r in results], indent=2)


def format_csv(
    paths: List[Path],
    rolls: List[str],
    detections: List[DetectionResult],
) -> str:
    results = _build_results(paths, rolls, detections)
    output = io.StringIO()
    if results:
        writer = csv.DictWriter(output, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    return output.getvalue()


def write_output(
    paths: List[Path],
    rolls: List[str],
    detections: List[DetectionResult],
    output_format: str = "table",
    output_file: Optional[str] = None,
    confidence_threshold: float = 0.6,
) -> None:
    """Format and optionally write results."""
    if output_format == "table":
        text = format_table(paths, rolls, detections, confidence_threshold)
        if output_file:
            Path(output_file).write_text(text)
    elif output_format == "json":
        text = format_json(paths, rolls, detections)
        if output_file:
            Path(output_file).write_text(text)
        else:
            print(text)
    elif output_format == "csv":
        text = format_csv(paths, rolls, detections)
        if output_file:
            Path(output_file).write_text(text)
        else:
            print(text)

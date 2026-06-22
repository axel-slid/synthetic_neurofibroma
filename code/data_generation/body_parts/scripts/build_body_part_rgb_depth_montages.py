#!/usr/bin/env python3
"""Build full RGB/depth-pair montages for each synthetic body part."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "synthetic"
    / "_regen_manual_body_parts"
    / "multiple_physics_aug_growth"
    / "data"
    / "camera_depth_manifest.csv"
)
DEFAULT_VISUALIZATION_ROOT = (
    ROOT
    / "data"
    / "synthetic"
    / "_regen_manual_body_parts"
    / "multiple_physics_aug_growth_visualizations"
)
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_root_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def resolve_manifest_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def rows_by_body_part(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {body_part: [] for body_part in BODY_PARTS}
    for row in rows:
        body_part = row.get("body_part")
        if body_part in grouped:
            grouped[body_part].append(row)
    missing = [body_part for body_part, part_rows in grouped.items() if not part_rows]
    if missing:
        raise ValueError(f"Missing body parts in manifest: {missing}")
    return grouped


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> str:
    if draw.textlength(text) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix) > max_width:
        text = text[:-1]
    return text + suffix if text else suffix


def load_square(path: Path, tile_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)


def make_pair_tile(row: dict[str, str], tile_size: int) -> Image.Image:
    rgb = load_square(resolve_manifest_path(row["image_path"]), tile_size)
    depth = load_square(resolve_manifest_path(row["depth_vis_path"]), tile_size)
    tile = Image.new("RGB", (tile_size * 2, tile_size), (255, 255, 255))
    tile.paste(rgb, (0, 0))
    tile.paste(depth, (tile_size, 0))
    return tile


def make_montage(
    body_part: str,
    rows: list[dict[str, str]],
    tile_size: int,
    columns: int,
    include_labels: bool,
) -> Image.Image:
    pair_width = tile_size * 2
    label_height = 18 if include_labels else 0
    title_height = 42
    tile_height = tile_size + label_height
    rows_count = int(math.ceil(len(rows) / columns))
    montage = Image.new(
        "RGB",
        (columns * pair_width, title_height + rows_count * tile_height),
        (248, 249, 251),
    )
    draw = ImageDraw.Draw(montage)
    draw.rectangle((0, 0, montage.width, title_height), fill=(236, 239, 243))
    title = f"{body_part}: all {len(rows)} RGB / GT depth pairs"
    draw.text((10, 10), title, fill=(18, 24, 32))
    draw.text((montage.width - 180, 10), "RGB | GT depth", fill=(76, 86, 100))

    for index, row in enumerate(rows):
        x = (index % columns) * pair_width
        y = title_height + (index // columns) * tile_height
        montage.paste(make_pair_tile(row, tile_size), (x, y + label_height))
        if include_labels:
            label = fit_text(draw, row["sample_id"], pair_width - 4)
            draw.text((x + 2, y + 2), label, fill=(76, 86, 100))
    return montage


def image_to_png_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def plotly_notebook(body_part: str, row_count: int, image: Image.Image) -> nbformat.NotebookNode:
    source = image_to_png_data_uri(image)
    fig = go.Figure(go.Image(source=source))
    fig.update_layout(
        title=f"{body_part}: all {row_count} RGB / GT depth pairs",
        width=min(1800, image.width),
        height=min(1800, image.height),
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        dragmode="pan",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {body_part} RGB / GT depth montage\n\nAll {row_count} pairs."),
        nbformat.v4.new_code_cell(
            source="",
            outputs=[
                nbformat.v4.new_output(
                    output_type="display_data",
                    data={
                        "application/vnd.plotly.v1+json": fig.to_plotly_json(),
                        "text/plain": f"<Plotly RGB/depth montage for {body_part}>",
                    },
                    metadata={},
                )
            ],
            execution_count=None,
        ),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return notebook


def write_body_part_outputs(
    visualization_root: Path,
    body_part: str,
    rows: list[dict[str, str]],
    tile_size: int,
    columns: int,
    include_labels: bool,
) -> dict[str, Any]:
    print(f"building {body_part}: {len(rows)} pairs", flush=True)
    montage = make_montage(body_part, rows, tile_size, columns, include_labels)

    gif_dir = visualization_root / "montages" / "gifs"
    plotly_dir = visualization_root / "montages" / "plotly"
    gif_dir.mkdir(parents=True, exist_ok=True)
    plotly_dir.mkdir(parents=True, exist_ok=True)

    gif_path = gif_dir / f"{body_part}_all_rgb_depth_pairs_montage.gif"
    notebook_path = plotly_dir / f"{body_part}_all_rgb_depth_pairs_montage.ipynb"
    montage.save(gif_path, format="GIF")
    nbformat.write(plotly_notebook(body_part, len(rows), montage), notebook_path)

    return {
        "body_part": body_part,
        "pair_count": len(rows),
        "gif": repo_relative(gif_path),
        "notebook": repo_relative(notebook_path),
        "montage_width": montage.width,
        "montage_height": montage.height,
        "tile_size": tile_size,
        "columns": columns,
    }


def build_montages(
    manifest: Path,
    visualization_root: Path,
    tile_size: int,
    columns: int,
    include_labels: bool,
) -> dict[str, Any]:
    rows = read_manifest(manifest)
    grouped = rows_by_body_part(rows)
    outputs = [
        write_body_part_outputs(visualization_root, body_part, grouped[body_part], tile_size, columns, include_labels)
        for body_part in BODY_PARTS
    ]
    summary = {
        "source_manifest": repo_relative(manifest),
        "visualization_root": repo_relative(visualization_root),
        "total_pair_count": sum(item["pair_count"] for item in outputs),
        "body_parts": outputs,
    }
    summary_path = visualization_root / "montages" / "rgb_depth_pair_montage_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["manifest"] = repo_relative(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=repo_relative(DEFAULT_MANIFEST))
    parser.add_argument("--visualization-root", default=repo_relative(DEFAULT_VISUALIZATION_ROOT))
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--columns", type=int, default=25)
    parser.add_argument("--include-labels", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_montages(
        manifest=resolve_root_path(args.manifest),
        visualization_root=resolve_root_path(args.visualization_root),
        tile_size=args.tile_size,
        columns=args.columns,
        include_labels=args.include_labels,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build real RGB vs ground-truth depth visualizations for body-part pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat
import numpy as np
from nbclient import NotebookClient
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = ROOT / "data" / "depth_maps" / "body_parts"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_root_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def resolve_pair_path(dataset_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = [
        dataset_root / path,
        ROOT / path,
        dataset_root.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def body_part_rows(dataset_root: Path, body_part: str) -> tuple[Path, list[dict[str, str]]]:
    manifest_path = dataset_root / "data" / body_part / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing body-part manifest: {manifest_path}")

    rows = read_rows(manifest_path)
    if not rows:
        raise ValueError(f"No rows in manifest: {manifest_path}")
    return manifest_path, rows


def evenly_spaced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indices]


def text_width(draw: ImageDraw.ImageDraw, text: str) -> int:
    bbox = draw.textbbox((0, 0), text)
    return bbox[2] - bbox[0]


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> str:
    if text_width(draw, text) <= max_width:
        return text
    suffix = "..."
    while text and text_width(draw, text + suffix) > max_width:
        text = text[:-1]
    return text + suffix if text else suffix


def compose_real_vs_gt_frame(
    dataset_root: Path,
    row: dict[str, str],
    body_part: str,
    tile_size: int,
) -> np.ndarray:
    label_height = 30
    footer_height = 24
    width = tile_size * 2
    height = tile_size + label_height + footer_height

    rgb = (
        Image.open(resolve_pair_path(dataset_root, row["image_path"]))
        .convert("RGB")
        .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    )
    depth = (
        Image.open(resolve_pair_path(dataset_root, row["depth_vis_path"]))
        .convert("RGB")
        .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    )

    frame = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(frame)
    frame.paste(rgb, (0, label_height))
    frame.paste(depth, (tile_size, label_height))

    draw.rectangle((0, 0, width, label_height - 1), fill=(239, 241, 244))
    draw.rectangle((0, tile_size + label_height, width, height), fill=(239, 241, 244))
    draw.line((tile_size, 0, tile_size, tile_size + label_height), fill=(210, 214, 220), width=1)
    draw.text((10, 8), "Real RGB", fill=(20, 24, 31))
    draw.text((tile_size + 10, 8), "GT depth", fill=(20, 24, 31))

    footer = f"{body_part} | {row['pair_id']} | {row.get('scan_id', '')}"
    footer = fit_text(draw, footer, width - 20)
    draw.text((10, tile_size + label_height + 6), footer, fill=(20, 24, 31))
    return np.asarray(frame)


def save_real_vs_gt_gif(
    dataset_root: Path,
    body_part: str,
    rows: list[dict[str, str]],
    frame_count: int,
    tile_size: int,
) -> Path:
    output_dir = dataset_root / "visualizations" / body_part / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{body_part}_real_vs_gt_preview.gif"

    frames = [
        compose_real_vs_gt_frame(dataset_root, row, body_part, tile_size)
        for row in evenly_spaced_rows(rows, frame_count)
    ]
    imageio.mimsave(output_path, frames, duration=0.22, loop=0)
    return output_path


def notebook_cells(
    dataset_root: Path,
    manifest_path: Path,
    body_part: str,
    notebook_sample_count: int,
) -> list[nbformat.NotebookNode]:
    dataset_root_text = str(dataset_root)
    manifest_path_text = str(manifest_path)
    return [
        nbformat.v4.new_markdown_cell(
            f"# {body_part}: real RGB vs GT depth\n\n"
            "Executed Plotly notebook for interactively checking rendered RGB images against "
            "their ground-truth depth maps."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import base64\n"
            "import csv\n"
            "from io import BytesIO\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"DATASET_ROOT = Path({dataset_root_text!r})\n"
            f"MANIFEST_PATH = Path({manifest_path_text!r})\n"
            f"BODY_PART = {body_part!r}\n"
            f"NOTEBOOK_SAMPLE_COUNT = {notebook_sample_count!r}\n\n"
            "def resolve_pair_path(path_value):\n"
            "    path = Path(path_value)\n"
            "    if path.is_absolute():\n"
            "        return path\n"
            "    candidates = [DATASET_ROOT / path, DATASET_ROOT.parent / path]\n"
            "    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])\n\n"
            "with MANIFEST_PATH.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n\n"
            "indices = np.linspace(0, len(rows) - 1, min(NOTEBOOK_SAMPLE_COUNT, len(rows)), dtype=int)\n"
            "samples = [rows[int(index)] for index in indices]\n"
            "{'body_part': BODY_PART, 'pair_count': len(rows), 'previewed_pair_count': len(samples)}"
        ),
        nbformat.v4.new_code_cell(
            "def load_image_source(row, field):\n"
            "    image = Image.open(resolve_pair_path(row[field])).convert('RGB')\n"
            "    buffer = BytesIO()\n"
            "    image.save(buffer, format='PNG')\n"
            "    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')\n"
            "    return 'data:image/png;base64,' + encoded\n\n"
            "def load_depth_surface(row, stride=4):\n"
            "    depth = np.load(resolve_pair_path(row['depth_npy_path'])).astype(float)\n"
            "    z = depth[::stride, ::stride]\n"
            "    z[~np.isfinite(z) | (z <= 0.0)] = np.nan\n"
            "    yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]\n"
            "    return xx, -yy, z\n\n"
            "fig = make_subplots(\n"
            "    rows=1,\n"
            "    cols=3,\n"
            "    specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'surface'}]],\n"
            "    subplot_titles=('Real RGB', 'GT depth image', 'Interactive GT depth surface'),\n"
            "    column_widths=[0.28, 0.28, 0.44],\n"
            ")\n\n"
            "for index, row in enumerate(samples):\n"
            "    visible = index == 0\n"
            "    sample_id = row.get('pair_id') or row.get('sample_id') or f'sample_{index}'\n"
            "    fig.add_trace(go.Image(source=load_image_source(row, 'image_path'), name=f'{sample_id} real RGB', visible=visible), row=1, col=1)\n"
            "    fig.add_trace(go.Image(source=load_image_source(row, 'depth_vis_path'), name=f'{sample_id} GT depth', visible=visible), row=1, col=2)\n"
            "    xx, yy, z = load_depth_surface(row)\n"
            "    fig.add_trace(\n"
            "        go.Surface(\n"
            "            x=xx,\n"
            "            y=yy,\n"
            "            z=z,\n"
            "            surfacecolor=z,\n"
            "            colorscale='Viridis',\n"
            "            colorbar={'title': 'm'},\n"
            "            connectgaps=False,\n"
            "            name=f'{sample_id} GT surface',\n"
            "            visible=visible,\n"
            "        ),\n"
            "        row=1,\n"
            "        col=3,\n"
            "    )\n\n"
            "buttons = []\n"
            "for index, row in enumerate(samples):\n"
            "    visible = [False] * (len(samples) * 3)\n"
            "    visible[index * 3:index * 3 + 3] = [True, True, True]\n"
            "    sample_id = row.get('pair_id') or row.get('sample_id') or f'sample_{index}'\n"
            "    buttons.append({\n"
            "        'label': sample_id,\n"
            "        'method': 'update',\n"
            "        'args': [\n"
            "            {'visible': visible},\n"
            "            {'title': f'{BODY_PART}: real RGB vs GT depth ({sample_id})'},\n"
            "        ],\n"
            "    })\n\n"
            "fig.update_layout(\n"
            "    title=f\"{BODY_PART}: real RGB vs GT depth ({samples[0]['pair_id']})\",\n"
            "    width=1320,\n"
            "    height=620,\n"
            "    margin={'l': 10, 'r': 10, 't': 90, 'b': 10},\n"
            "    updatemenus=[{\n"
            "        'buttons': buttons,\n"
            "        'direction': 'down',\n"
            "        'showactive': True,\n"
            "        'x': 0.0,\n"
            "        'xanchor': 'left',\n"
            "        'y': 1.13,\n"
            "        'yanchor': 'top',\n"
            "    }],\n"
            "    scene={\n"
            "        'xaxis_title': 'x pixels / stride',\n"
            "        'yaxis_title': 'y pixels / stride',\n"
            "        'zaxis_title': 'camera z distance (m)',\n"
            "        'aspectmode': 'data',\n"
            "    },\n"
            ")\n"
            "fig.update_xaxes(visible=False)\n"
            "fig.update_yaxes(visible=False)\n"
            "fig"
        ),
    ]


def save_plotly_notebook(
    dataset_root: Path,
    manifest_path: Path,
    body_part: str,
    notebook_sample_count: int,
) -> Path:
    output_dir = dataset_root / "visualizations" / body_part / "plotly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{body_part}_real_vs_gt_viewer.ipynb"

    notebook = nbformat.v4.new_notebook(
        cells=notebook_cells(dataset_root, manifest_path, body_part, notebook_sample_count)
    )
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    NotebookClient(notebook, timeout=180, kernel_name="python3").execute()
    nbformat.write(notebook, output_path)
    return output_path


def build_body_part_visualization(
    dataset_root: Path,
    body_part: str,
    frame_count: int,
    tile_size: int,
    notebook_sample_count: int,
) -> dict[str, Any]:
    manifest_path, rows = body_part_rows(dataset_root, body_part)
    gif_path = save_real_vs_gt_gif(dataset_root, body_part, rows, frame_count, tile_size)
    notebook_path = save_plotly_notebook(dataset_root, manifest_path, body_part, notebook_sample_count)
    return {
        "body_part": body_part,
        "pair_count": len(rows),
        "manifest": root_relative(manifest_path),
        "gif": root_relative(gif_path),
        "notebook": root_relative(notebook_path),
        "gif_frame_count": min(frame_count, len(rows)),
        "notebook_sample_count": min(notebook_sample_count, len(rows)),
    }


def build_visualizations(
    dataset_root: Path,
    body_parts: list[str],
    frame_count: int,
    tile_size: int,
    notebook_sample_count: int,
) -> dict[str, Any]:
    visualization_root = dataset_root / "visualizations"
    visualization_root.mkdir(parents=True, exist_ok=True)
    results = [
        build_body_part_visualization(dataset_root, body_part, frame_count, tile_size, notebook_sample_count)
        for body_part in body_parts
    ]
    summary = {
        "dataset_root": root_relative(dataset_root),
        "visualization_root": root_relative(visualization_root),
        "body_parts": results,
        "total_pair_count": sum(result["pair_count"] for result in results),
    }
    manifest_path = visualization_root / "real_vs_gt_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["manifest"] = root_relative(manifest_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=root_relative(DEFAULT_DATASET_ROOT))
    parser.add_argument("--body-parts", nargs="+", default=BODY_PARTS)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--notebook-sample-count", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = resolve_root_path(args.dataset_root)
    summary = build_visualizations(
        dataset_root=dataset_root,
        body_parts=args.body_parts,
        frame_count=args.frame_count,
        tile_size=args.tile_size,
        notebook_sample_count=args.notebook_sample_count,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

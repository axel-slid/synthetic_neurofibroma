#!/usr/bin/env python3
"""Build body-part RGB vs GT depth vs DepthPro prediction visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat
import numpy as np
import torch
from nbclient import NotebookClient
from PIL import Image, ImageDraw
from transformers import pipeline


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = ROOT / "data" / "depth_maps" / "body_parts"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]
MODEL_ID = "apple/DepthPro-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--body-parts", nargs="+", default=BODY_PARTS, choices=BODY_PARTS)
    parser.add_argument("--samples-per-part", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--tile-size", type=int, default=180)
    parser.add_argument("--montage-columns", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    return parser.parse_args()


def root_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_dataset_path(dataset_root: Path, path_value: str) -> Path:
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
    raise FileNotFoundError(f"Could not resolve path from manifest value: {path_value}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return rows


def evenly_spaced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0:
        raise ValueError("--samples-per-part must be positive")
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indices]


def device_index(device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    return -1


def load_image(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def predict_depth(pipe: Any, image: Image.Image) -> np.ndarray:
    with torch.inference_mode():
        result = pipe(image)
    depth = result["predicted_depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().float().cpu().numpy()
    else:
        depth = np.asarray(depth, dtype=np.float32)
    return np.squeeze(depth).astype(np.float32)


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if depth.shape == (height, width):
        return depth.astype(np.float32, copy=False)
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def near_bright_depth_visual(depth: np.ndarray) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 0.0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(mask):
        return vis

    near = float(np.percentile(depth[mask], 1))
    far = float(np.percentile(depth[mask], 99))
    if far <= near:
        far = near + 1e-6

    normalized = np.clip((far - depth) / (far - near), 0.0, 1.0)
    vis[mask] = np.rint(normalized[mask] * 255.0).astype(np.uint8)
    return vis


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


def panel_image(path: Path, tile_size: int, grayscale: bool = False) -> Image.Image:
    mode = "L" if grayscale else "RGB"
    return Image.open(path).convert(mode).resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert("RGB")


def compose_triplet_frame(row: dict[str, Any], tile_size: int) -> np.ndarray:
    label_height = 34
    footer_height = 30
    width = tile_size * 3
    height = label_height + tile_size + footer_height
    frame = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(frame)

    panels = [
        ("Original image", Path(row["resolved_image_path"]), False),
        ("GT depth map", Path(row["resolved_gt_depth_vis_path"]), True),
        ("Predicted depth map", Path(row["resolved_pred_depth_vis_path"]), True),
    ]

    draw.rectangle((0, 0, width, label_height - 1), fill=(238, 240, 244))
    draw.rectangle((0, label_height + tile_size, width, height), fill=(238, 240, 244))
    for idx, (label, path, grayscale) in enumerate(panels):
        x0 = idx * tile_size
        frame.paste(panel_image(path, tile_size, grayscale=grayscale), (x0, label_height))
        draw.text((x0 + 10, 10), label, fill=(18, 22, 30))
        if idx:
            draw.line((x0, 0, x0, label_height + tile_size), fill=(207, 211, 218), width=1)

    footer = f"{row['body_part']} | {row['sample_id']} | {row.get('scan_id', '')}"
    draw.text((10, label_height + tile_size + 8), fit_text(draw, footer, width - 20), fill=(18, 22, 30))
    return np.asarray(frame)


def build_gif(rows: list[dict[str, Any]], output_path: Path, tile_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [compose_triplet_frame(row, tile_size) for row in rows]
    imageio.mimsave(output_path, frames, duration=0.34, loop=0)


def build_montage(rows: list[dict[str, Any]], output_path: Path, tile_size: int, columns: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(compose_triplet_frame(row, tile_size)) for row in rows]
    frame_w, frame_h = frames[0].size
    rows_count = int(math.ceil(len(frames) / columns))
    montage = Image.new("RGB", (columns * frame_w, rows_count * frame_h), (248, 248, 248))
    for idx, frame in enumerate(frames):
        x = (idx % columns) * frame_w
        y = (idx // columns) * frame_h
        montage.paste(frame, (x, y))
    montage.save(output_path)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "pair_id",
        "body_part",
        "scan_id",
        "patient_volume_index",
        "source_image_path",
        "source_gt_depth_npy_path",
        "source_gt_depth_vis_path",
        "pred_depth_npy_path",
        "pred_depth_vis_path",
        "metadata_path",
        "model",
        "width",
        "height",
        "pred_depth_min",
        "pred_depth_median",
        "pred_depth_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def notebook_cells(dataset_root: Path, manifest_path: Path, sample_limit: int) -> list[nbformat.NotebookNode]:
    return [
        nbformat.v4.new_markdown_cell(
            "# Body-part DepthPro comparison\n\n"
            "Executed Plotly notebook for inspecting original images, ground-truth depth, "
            "and predicted depth maps for the body-part depth-map dataset."
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
            f"DATASET_ROOT = Path({str(dataset_root)!r})\n"
            f"MANIFEST_PATH = Path({str(manifest_path)!r})\n"
            f"SAMPLE_LIMIT = {sample_limit!r}\n\n"
            "def resolve_path(path_value):\n"
            "    path = Path(path_value)\n"
            "    if path.is_absolute():\n"
            "        return path\n"
            "    candidates = [Path.cwd() / path, DATASET_ROOT / path, DATASET_ROOT.parent / path]\n"
            "    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])\n\n"
            "with MANIFEST_PATH.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n"
            "rows = rows[:SAMPLE_LIMIT]\n"
            "{'sample_count': len(rows), 'body_parts': sorted({row['body_part'] for row in rows})}"
        ),
        nbformat.v4.new_code_cell(
            "def image_source(path_value):\n"
            "    image = Image.open(resolve_path(path_value)).convert('RGB')\n"
            "    buffer = BytesIO()\n"
            "    image.save(buffer, format='PNG')\n"
            "    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')\n"
            "    return 'data:image/png;base64,' + encoded\n\n"
            "def surface_values(path_value, max_points=96):\n"
            "    depth = np.load(resolve_path(path_value)).astype(float)\n"
            "    mask = np.isfinite(depth) & (depth > 0.0)\n"
            "    z = np.where(mask, depth, np.nan)\n"
            "    stride = max(1, int(np.ceil(max(z.shape) / max_points)))\n"
            "    z = z[::stride, ::stride]\n"
            "    yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]\n"
            "    return xx, -yy, z\n\n"
            "fig = make_subplots(\n"
            "    rows=1,\n"
            "    cols=3,\n"
            "    specs=[[{'type': 'xy'}, {'type': 'surface'}, {'type': 'surface'}]],\n"
            "    subplot_titles=('Original image', 'GT depth surface', 'Predicted depth surface'),\n"
            "    column_widths=[0.28, 0.36, 0.36],\n"
            ")\n\n"
            "for index, row in enumerate(rows):\n"
            "    visible = index == 0\n"
            "    sample_id = row.get('sample_id', f'sample_{index}')\n"
            "    fig.add_trace(go.Image(source=image_source(row['source_image_path']), visible=visible, name=f'{sample_id} image'), row=1, col=1)\n"
            "    for col, field, title in [(2, 'source_gt_depth_npy_path', 'GT'), (3, 'pred_depth_npy_path', 'Predicted')]:\n"
            "        x, y, z = surface_values(row[field])\n"
            "        fig.add_trace(\n"
            "            go.Surface(\n"
            "                x=x,\n"
            "                y=y,\n"
            "                z=z,\n"
            "                surfacecolor=z,\n"
            "                colorscale='Viridis',\n"
            "                colorbar={'title': 'depth'},\n"
            "                connectgaps=False,\n"
            "                showscale=(index == 0 and col == 3),\n"
            "                visible=visible,\n"
            "                name=f'{sample_id} {title}',\n"
            "            ),\n"
            "            row=1,\n"
            "            col=col,\n"
            "        )\n\n"
            "buttons = []\n"
            "for index, row in enumerate(rows):\n"
            "    visible = [False] * (len(rows) * 3)\n"
            "    visible[index * 3:index * 3 + 3] = [True, True, True]\n"
            "    label = f\"{row['body_part']} | {row['sample_id']}\"\n"
            "    buttons.append({\n"
            "        'label': label,\n"
            "        'method': 'update',\n"
            "        'args': [\n"
            "            {'visible': visible},\n"
            "            {'title': f\"Body-part DepthPro comparison: {label}\"},\n"
            "        ],\n"
            "    })\n\n"
            "first_label = f\"{rows[0]['body_part']} | {rows[0]['sample_id']}\"\n"
            "fig.update_layout(\n"
            "    title=f'Body-part DepthPro comparison: {first_label}',\n"
            "    width=1380,\n"
            "    height=660,\n"
            "    margin={'l': 10, 'r': 10, 't': 95, 'b': 10},\n"
            "    updatemenus=[{\n"
            "        'buttons': buttons,\n"
            "        'direction': 'down',\n"
            "        'showactive': True,\n"
            "        'x': 0.0,\n"
            "        'xanchor': 'left',\n"
            "        'y': 1.12,\n"
            "        'yanchor': 'top',\n"
            "    }],\n"
            ")\n"
            "fig.update_xaxes(visible=False)\n"
            "fig.update_yaxes(visible=False)\n"
            "fig.update_scenes(xaxis_title='x', yaxis_title='y', zaxis_title='depth', aspectmode='data')\n"
            "fig"
        ),
    ]


def save_plotly_notebook(dataset_root: Path, manifest_path: Path, output_path: Path, sample_limit: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(cells=notebook_cells(dataset_root, manifest_path, sample_limit))
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    client = NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    nbformat.write(executed, output_path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    data_root = dataset_root / "data" / "predictions" / "depth_pro"
    pred_depth_root = data_root / "predicted_depth"
    pred_vis_root = data_root / "predicted_depth_vis"
    metadata_root = data_root / "metadata"
    visual_root = dataset_root / "visualizations" / "depth_pro_comparison"
    gif_root = visual_root / "gifs"
    preview_root = visual_root / "previews"
    plotly_root = visual_root / "plotly"

    if args.overwrite:
        for path in [data_root, visual_root]:
            if path.exists():
                shutil.rmtree(path)

    for path in [pred_depth_root, pred_vis_root, metadata_root, gif_root, preview_root, plotly_root]:
        path.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, str]] = []
    for body_part in args.body_parts:
        manifest_path = dataset_root / "data" / body_part / "manifest.csv"
        rows = read_manifest(manifest_path)
        selected_rows.extend(evenly_spaced_rows(rows, args.samples_per_part))

    pipe = pipeline("depth-estimation", model=MODEL_ID, device=device_index(args.device))

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        sample_id = row["sample_id"]
        body_part = row["body_part"]
        image_path = resolve_dataset_path(dataset_root, row["image_path"])
        gt_depth_path = resolve_dataset_path(dataset_root, row["depth_npy_path"])
        gt_depth_vis_path = resolve_dataset_path(dataset_root, row["depth_vis_path"])
        pred_depth_path = pred_depth_root / f"{sample_id}_depthpro_depth.npy"
        pred_vis_path = pred_vis_root / f"{sample_id}_depthpro_depth_vis.png"
        metadata_path = metadata_root / f"{sample_id}_depthpro.json"

        rgb_full = Image.open(image_path).convert("RGB")
        if args.overwrite or not pred_depth_path.exists():
            image = load_image(image_path, args.max_side)
            pred_depth = predict_depth(pipe, image)
            pred_depth = resize_depth(pred_depth, rgb_full.size)
            np.save(pred_depth_path, pred_depth.astype(np.float32))
            imageio.imwrite(pred_vis_path, near_bright_depth_visual(pred_depth))
        else:
            pred_depth = np.load(pred_depth_path)
            if args.overwrite or not pred_vis_path.exists():
                imageio.imwrite(pred_vis_path, near_bright_depth_visual(pred_depth))

        valid = np.isfinite(pred_depth)
        values = pred_depth[valid] if np.any(valid) else np.array([np.nan], dtype=np.float32)
        metadata = {
            "sample_id": sample_id,
            "pair_id": row.get("pair_id", sample_id),
            "body_part": body_part,
            "scan_id": row.get("scan_id", ""),
            "model": MODEL_ID,
            "source_image_path": root_relative(image_path),
            "source_gt_depth_npy_path": root_relative(gt_depth_path),
            "source_gt_depth_vis_path": root_relative(gt_depth_vis_path),
            "pred_depth_npy_path": root_relative(pred_depth_path),
            "pred_depth_vis_path": root_relative(pred_vis_path),
            "width": rgb_full.width,
            "height": rgb_full.height,
            "pred_depth_min": float(np.nanmin(values)),
            "pred_depth_median": float(np.nanmedian(values)),
            "pred_depth_max": float(np.nanmax(values)),
            "depth_visualization": "near_bright_far_dark_1_to_99_percentile",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        output_row = {
            "sample_id": sample_id,
            "pair_id": row.get("pair_id", sample_id),
            "body_part": body_part,
            "scan_id": row.get("scan_id", ""),
            "patient_volume_index": row.get("patient_volume_index", ""),
            "source_image_path": root_relative(image_path),
            "source_gt_depth_npy_path": root_relative(gt_depth_path),
            "source_gt_depth_vis_path": root_relative(gt_depth_vis_path),
            "pred_depth_npy_path": root_relative(pred_depth_path),
            "pred_depth_vis_path": root_relative(pred_vis_path),
            "metadata_path": root_relative(metadata_path),
            "model": MODEL_ID,
            "width": rgb_full.width,
            "height": rgb_full.height,
            "pred_depth_min": metadata["pred_depth_min"],
            "pred_depth_median": metadata["pred_depth_median"],
            "pred_depth_max": metadata["pred_depth_max"],
            "resolved_image_path": str(image_path),
            "resolved_gt_depth_npy_path": str(gt_depth_path),
            "resolved_gt_depth_vis_path": str(gt_depth_vis_path),
            "resolved_pred_depth_npy_path": str(pred_depth_path),
            "resolved_pred_depth_vis_path": str(pred_vis_path),
        }
        output_rows.append(output_row)
        print(f"[{index:03d}/{len(selected_rows):03d}] {body_part}: {sample_id}", flush=True)

    manifest_path = data_root / "manifest.csv"
    write_manifest(manifest_path, output_rows)

    by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        by_part[row["body_part"]].append(row)

    gif_outputs: dict[str, str] = {}
    for body_part in args.body_parts:
        part_gif_path = gif_root / f"{body_part}_original_gt_predicted_depth.gif"
        build_gif(by_part[body_part], part_gif_path, args.tile_size)
        gif_outputs[body_part] = root_relative(part_gif_path)

    combined_gif_path = gif_root / "body_parts_original_gt_predicted_depth.gif"
    montage_path = preview_root / "body_parts_original_gt_predicted_depth_montage.png"
    build_gif(output_rows, combined_gif_path, args.tile_size)
    build_montage(output_rows, montage_path, max(128, args.tile_size // 2), args.montage_columns)

    notebook_path = plotly_root / "body_part_depth_pro_comparison_viewer.ipynb"
    if not args.skip_notebook:
        save_plotly_notebook(dataset_root, manifest_path, notebook_path, sample_limit=len(output_rows))

    summary = {
        "dataset": "body_parts",
        "sample_count": len(output_rows),
        "samples_per_part": args.samples_per_part,
        "body_parts": args.body_parts,
        "model": MODEL_ID,
        "folders": {
            "data": root_relative(data_root),
            "visualizations": root_relative(visual_root),
        },
        "outputs": {
            "manifest": root_relative(manifest_path),
            "combined_gif": root_relative(combined_gif_path),
            "per_part_gifs": gif_outputs,
            "montage": root_relative(montage_path),
            "notebook": root_relative(notebook_path) if notebook_path.exists() else None,
        },
    }
    summary_path = data_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

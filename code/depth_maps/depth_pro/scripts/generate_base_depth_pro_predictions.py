#!/usr/bin/env python3
"""Run Depth Pro on the base RGB/depth examples and build comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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
DEPTH_ROOT = ROOT / "data" / "depth_maps"
BASE_ROOT = DEPTH_ROOT / "base"
DEFAULT_MANIFEST = BASE_ROOT / "manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "predictions" / "depth_pro_base"
MODEL_ID = "apple/DepthPro-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=96)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_base_path(relative_path: str) -> Path:
    path = Path(relative_path)
    candidates = [
        BASE_ROOT / path,
        DEPTH_ROOT / path,
    ]
    parts = path.parts
    if parts[:1] == ("base",):
        candidates.append(BASE_ROOT.joinpath(*parts[1:]))
        if len(parts) > 2 and parts[1] in {"images", "depth", "depth_vis", "metadata"}:
            candidates.append(BASE_ROOT / "images" / Path(*parts[1:]))
    elif parts[:1] in {("images",), ("depth",), ("depth_vis",), ("metadata",)}:
        candidates.append(BASE_ROOT / "images" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve base path from manifest value: {relative_path}")


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def output_relative(path: Path, output_root: Path) -> str:
    return str(path.resolve().relative_to(output_root.resolve()))


def load_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows found in manifest: {path}")
    return rows


def load_image(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def predict_depth(pipe: Any, image: Image.Image) -> np.ndarray:
    with torch.inference_mode():
        depth = pipe(image)["predicted_depth"]
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


def paste_panel(tile: Image.Image, source: Image.Image, x: int, y: int, tile_size: int) -> None:
    image = source.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    tile.paste(image, (x, y))


def make_triplet_tile(
    rgb_path: Path,
    gt_depth_vis_path: Path,
    pred_depth_vis_path: Path,
    tile_size: int,
    label_height: int,
) -> Image.Image:
    width = tile_size * 3
    tile = Image.new("RGB", (width, tile_size + label_height), "white")
    draw = ImageDraw.Draw(tile)
    for idx, label in enumerate(["RGB", "GT depth", "Depth Pro"]):
        draw.text((idx * tile_size + 4, 3), label, fill=(0, 0, 0))
    paste_panel(tile, Image.open(rgb_path), 0, label_height, tile_size)
    paste_panel(tile, Image.open(gt_depth_vis_path).convert("L"), tile_size, label_height, tile_size)
    paste_panel(tile, Image.open(pred_depth_vis_path).convert("L"), tile_size * 2, label_height, tile_size)
    return tile


def build_montage(rows: list[dict[str, Any]], output_path: Path, tile_size: int, columns: int) -> None:
    label_height = max(16, tile_size // 6)
    tiles = [
        make_triplet_tile(
            Path(row["resolved_rgb_path"]),
            Path(row["resolved_gt_depth_vis_path"]),
            Path(row["resolved_pred_depth_vis_path"]),
            tile_size,
            label_height,
        )
        for row in rows
    ]
    rows_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new(
        "RGB",
        (columns * tile_size * 3, rows_count * (tile_size + label_height)),
        "white",
    )
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile.width
        y = (idx // columns) * tile.height
        montage.paste(tile, (x, y))
    montage.save(output_path)


def build_triplet_gif(rows: list[dict[str, Any]], output_path: Path, tile_size: int = 160) -> None:
    label_height = 36
    frames = []
    for row in rows:
        frame = make_triplet_tile(
            Path(row["resolved_rgb_path"]),
            Path(row["resolved_gt_depth_vis_path"]),
            Path(row["resolved_pred_depth_vis_path"]),
            tile_size,
            label_height,
        )
        draw = ImageDraw.Draw(frame)
        draw.text((4, 18), row["sample_id"], fill=(0, 0, 0))
        frames.append(np.asarray(frame))
    imageio.mimsave(output_path, frames, duration=0.12, loop=0)


def write_notebook(output_root: Path, rows: list[dict[str, Any]], notebook_path: Path) -> None:
    rel_output = repo_relative(output_root)

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Depth Pro base RGB/depth comparison\n\n"
            "This executed notebook loads the generated manifest and displays interactive Plotly 3D surfaces "
            "for the rendered ground-truth depth and the Depth Pro prediction."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import numpy as np\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"ROOT = Path({str(ROOT)!r})\n"
            f"OUTPUT_ROOT = ROOT / {rel_output!r}\n"
            "MANIFEST = OUTPUT_ROOT / 'data' / 'manifest.csv'\n"
            "with MANIFEST.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n"
            "print(f'Loaded {len(rows)} Depth Pro predictions from {MANIFEST}')"
        ),
        nbformat.v4.new_code_cell(
            "def load_depth(path_value):\n"
            "    return np.load(OUTPUT_ROOT / path_value)\n\n"
            "def surface_values(depth, max_points=96):\n"
            "    mask = np.isfinite(depth) & (depth > 0.0)\n"
            "    z = np.where(mask, depth, np.nan).astype(np.float32)\n"
            "    stride = max(1, int(np.ceil(max(z.shape) / max_points)))\n"
            "    z = z[::stride, ::stride]\n"
            "    if np.isfinite(z).any():\n"
            "        median = float(np.nanmedian(z))\n"
            "        scale = float(np.nanpercentile(np.abs(z - median), 95)) or 1.0\n"
            "        z = np.clip((z - median) / scale, -1.5, 1.5)\n"
            "        z = np.nan_to_num(z, nan=1.5)\n"
            "    else:\n"
            "        z = np.zeros_like(z)\n"
            "    h, w = z.shape\n"
            "    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))\n"
            "    return x, y, -z\n\n"
            "def make_depth_surface_comparison(row):\n"
            "    gt_depth = np.load(ROOT / row['source_gt_depth_npy_path'])\n"
            "    pred_depth = load_depth(row['pred_depth_npy_path'])\n"
            "    fig = make_subplots(\n"
            "        rows=1,\n"
            "        cols=2,\n"
            "        specs=[[{'type': 'surface'}, {'type': 'surface'}]],\n"
            "        subplot_titles=['GT depth', 'Depth Pro'],\n"
            "    )\n"
            "    for col, depth in enumerate([gt_depth, pred_depth], start=1):\n"
            "        x, y, z = surface_values(depth)\n"
            "        fig.add_trace(\n"
            "            go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale='Viridis', showscale=False, hoverinfo='skip'),\n"
            "            row=1,\n"
            "            col=col,\n"
            "        )\n"
            "    fig.update_layout(title=f\"{row['sample_id']}: depth surface comparison\", height=560, margin=dict(l=0, r=0, t=52, b=0))\n"
            "    fig.update_scenes(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data')\n"
            "    return fig"
        ),
        nbformat.v4.new_code_cell(
            "# Change sample_index and rerun this cell to inspect another RGB/depth pair.\n"
            "sample_index = 0\n"
            "fig = make_depth_surface_comparison(rows[sample_index])\n"
            "fig"
        ),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    nbformat.write(executed, notebook_path)


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = output_root / "data" / "manifest.csv"
    fieldnames = [
        "sample_id",
        "subject_id",
        "view_index",
        "source_rgb_path",
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
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    data_root = output_root / "data"
    pred_depth_root = data_root / "predicted_depth"
    pred_depth_vis_root = data_root / "predicted_depth_vis"
    metadata_root = data_root / "metadata"
    visualizations_root = output_root / "visualizations"
    for path in [pred_depth_root, pred_depth_vis_root, metadata_root, visualizations_root]:
        path.mkdir(parents=True, exist_ok=True)

    input_rows = load_manifest(args.manifest, args.limit)
    device_arg = 0 if args.device.startswith("cuda") and torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=MODEL_ID, device=device_arg)

    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows, start=1):
        sample_id = row["sample_id"]
        rgb_path = resolve_base_path(row["image_path"])
        gt_depth_npy_path = resolve_base_path(row["depth_npy_path"])
        gt_depth_vis_path = resolve_base_path(row["depth_vis_path"])
        pred_depth_path = pred_depth_root / f"{sample_id}_depthpro_depth.npy"
        pred_vis_path = pred_depth_vis_root / f"{sample_id}_depthpro_depth_vis.png"
        metadata_path = metadata_root / f"{sample_id}_depthpro.json"

        rgb_full = Image.open(rgb_path).convert("RGB")
        if not pred_depth_path.exists() or args.overwrite:
            image = load_image(rgb_path, args.max_side)
            pred_depth = predict_depth(pipe, image)
            pred_depth = resize_depth(pred_depth, rgb_full.size)
            np.save(pred_depth_path, pred_depth.astype(np.float32))
            imageio.imwrite(pred_vis_path, near_bright_depth_visual(pred_depth))
        else:
            pred_depth = np.load(pred_depth_path)
            if not pred_vis_path.exists():
                imageio.imwrite(pred_vis_path, near_bright_depth_visual(pred_depth))

        stats_mask = np.isfinite(pred_depth)
        stats_values = pred_depth[stats_mask] if np.any(stats_mask) else np.array([np.nan], dtype=np.float32)
        metadata = {
            "sample_id": sample_id,
            "model": MODEL_ID,
            "source_rgb_path": repo_relative(rgb_path),
            "source_gt_depth_npy_path": repo_relative(gt_depth_npy_path),
            "source_gt_depth_vis_path": repo_relative(gt_depth_vis_path),
            "pred_depth_npy_path": output_relative(pred_depth_path, output_root),
            "pred_depth_vis_path": output_relative(pred_vis_path, output_root),
            "width": rgb_full.width,
            "height": rgb_full.height,
            "pred_depth_min": float(np.nanmin(stats_values)),
            "pred_depth_median": float(np.nanmedian(stats_values)),
            "pred_depth_max": float(np.nanmax(stats_values)),
            "depth_visualization": "near_bright_far_dark_1_to_99_percentile",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        output_row = {
            "sample_id": sample_id,
            "subject_id": row["subject_id"],
            "view_index": row["view_index"],
            "source_rgb_path": repo_relative(rgb_path),
            "source_gt_depth_npy_path": repo_relative(gt_depth_npy_path),
            "source_gt_depth_vis_path": repo_relative(gt_depth_vis_path),
            "pred_depth_npy_path": output_relative(pred_depth_path, output_root),
            "pred_depth_vis_path": output_relative(pred_vis_path, output_root),
            "metadata_path": output_relative(metadata_path, output_root),
            "model": MODEL_ID,
            "width": rgb_full.width,
            "height": rgb_full.height,
            "pred_depth_min": metadata["pred_depth_min"],
            "pred_depth_median": metadata["pred_depth_median"],
            "pred_depth_max": metadata["pred_depth_max"],
            "resolved_rgb_path": str(rgb_path),
            "resolved_gt_depth_npy_path": str(gt_depth_npy_path),
            "resolved_gt_depth_vis_path": str(gt_depth_vis_path),
            "resolved_pred_depth_npy_path": str(pred_depth_path),
            "resolved_pred_depth_vis_path": str(pred_vis_path),
        }
        output_rows.append(output_row)
        print(f"[{idx:03d}/{len(input_rows):03d}] wrote Depth Pro prediction for {sample_id}", flush=True)

    write_manifest(output_root, output_rows)

    montage_path = visualizations_root / "montage_rgb_gt_depthpro.png"
    gif_path = visualizations_root / "rgb_gt_depthpro_triplets.gif"
    notebook_path = visualizations_root / "plot_depth_pro_base_surfaces.ipynb"
    build_montage(output_rows, montage_path, args.tile_size, args.columns)
    build_triplet_gif(output_rows, gif_path)
    write_notebook(output_root, output_rows, notebook_path)

    summary = {
        "sample_count": len(output_rows),
        "source_manifest": repo_relative(args.manifest.resolve()),
        "model": MODEL_ID,
        "folders": {
            "data": repo_relative(data_root),
            "visualizations": repo_relative(visualizations_root),
        },
        "outputs": {
            "manifest": repo_relative(data_root / "manifest.csv"),
            "montage": repo_relative(montage_path),
            "gif": repo_relative(gif_path),
            "notebook": repo_relative(notebook_path),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

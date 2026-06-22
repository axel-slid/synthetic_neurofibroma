#!/usr/bin/env python3
"""Build polarity-corrected Depth Pro comparisons from saved predictions."""

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
from nbclient import NotebookClient
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_ROOT = ROOT / "data" / "predictions" / "depth_pro_base"
DEFAULT_MANIFEST = DEFAULT_SOURCE_ROOT / "data" / "manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "predictions" / "depth_pro_base_polarity_corrected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root used to resolve output-relative paths from the source prediction manifest.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--conversion",
        choices=["auto", "raw", "reciprocal"],
        default="auto",
        help="Use raw Depth Pro, reciprocal Depth Pro, or choose the polarity with the stronger GT correlation.",
    )
    parser.add_argument("--tile-size", type=int, default=96)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_manifest_path(path_value: str, source_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        source_root / path,
        ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve manifest path: {path_value}")


def load_rows(manifest_path: Path, limit: int | None) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows found in {manifest_path}")
    return rows


def reciprocal_depth(raw_depth: np.ndarray) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    valid = np.isfinite(raw_depth) & (raw_depth > 1e-6)
    corrected = np.zeros(raw_depth.shape, dtype=np.float32)
    corrected[valid] = 1.0 / raw_depth[valid]
    return corrected


def convert_depth(raw_depth: np.ndarray, conversion: str) -> np.ndarray:
    if conversion == "raw":
        return np.asarray(raw_depth, dtype=np.float32)
    if conversion == "reciprocal":
        return reciprocal_depth(raw_depth)
    raise ValueError(f"Unsupported conversion: {conversion}")


def depth_visual(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0.0)
    else:
        mask = mask & np.isfinite(depth) & (depth > 0.0)
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


def safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return float("nan")
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    if float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def median_scaled_abs_rel(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(pred_depth) & (pred_depth > 0.0) & np.isfinite(gt_depth) & (gt_depth > 0.0)
    if not np.any(valid):
        return float("nan")
    pred_values = pred_depth[valid]
    gt_values = gt_depth[valid]
    scale = float(np.median(gt_values) / max(float(np.median(pred_values)), 1e-6))
    scaled = pred_values * scale
    return float(np.mean(np.abs(scaled - gt_values) / np.maximum(gt_values, 1e-6)))


def paste_panel(tile: Image.Image, image: Image.Image, x: int, y: int, size: int) -> None:
    tile.paste(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), (x, y))


def make_triplet_tile(
    rgb_path: Path,
    gt_depth_vis_path: Path,
    pred_depth_vis_path: Path,
    tile_size: int,
    label_height: int,
    sample_id: str | None = None,
) -> Image.Image:
    labels = ["RGB", "GT depth", "Depth Pro checked"]
    tile = Image.new("RGB", (tile_size * 3, tile_size + label_height), "white")
    draw = ImageDraw.Draw(tile)
    for idx, label in enumerate(labels):
        draw.text((idx * tile_size + 4, 3), label, fill=(0, 0, 0))
    if sample_id:
        draw.text((4, max(14, label_height - 14)), sample_id, fill=(0, 0, 0))
    paste_panel(tile, Image.open(rgb_path), 0, label_height, tile_size)
    paste_panel(tile, Image.open(gt_depth_vis_path).convert("L"), tile_size, label_height, tile_size)
    paste_panel(tile, Image.open(pred_depth_vis_path).convert("L"), tile_size * 2, label_height, tile_size)
    return tile


def build_montage(rows: list[dict[str, Any]], output_path: Path, tile_size: int, columns: int) -> None:
    label_height = max(22, tile_size // 4)
    tiles = [
        make_triplet_tile(
            Path(row["resolved_rgb_path"]),
            Path(row["resolved_gt_depth_vis_path"]),
            Path(row["resolved_corrected_depth_vis_path"]),
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


def build_gif(rows: list[dict[str, Any]], output_path: Path, tile_size: int = 160) -> None:
    frames = []
    for row in rows:
        frames.append(
            np.asarray(
                make_triplet_tile(
                    Path(row["resolved_rgb_path"]),
                    Path(row["resolved_gt_depth_vis_path"]),
                    Path(row["resolved_corrected_depth_vis_path"]),
                    tile_size,
                    40,
                    row["sample_id"],
                )
            )
        )
    imageio.mimsave(output_path, frames, duration=0.14, loop=0)


def write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "subject_id",
        "view_index",
        "source_rgb_path",
        "source_gt_depth_npy_path",
        "source_gt_depth_vis_path",
        "raw_depthpro_depth_path",
        "selected_conversion",
        "selected_depth_path",
        "selected_depth_vis_path",
        "raw_depthpro_model",
        "width",
        "height",
        "raw_depth_min",
        "raw_depth_median",
        "raw_depth_max",
        "corrected_depth_min",
        "corrected_depth_median",
        "corrected_depth_max",
        "raw_metric_corr",
        "reciprocal_metric_corr",
        "median_scaled_abs_rel",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_diagnostics(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "raw_metric_corr",
        "reciprocal_metric_corr",
        "median_scaled_abs_rel",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_notebook(output_root: Path, manifest_path: Path, notebook_path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Polarity-corrected Depth Pro surfaces\n\n"
            "This executed notebook loads the corrected prediction manifest and displays interactive Plotly 3D "
            "surfaces for ground-truth depth and reciprocal Depth Pro depth."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import numpy as np\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"ROOT = Path({str(ROOT)!r})\n"
            f"OUTPUT_ROOT = Path({str(output_root)!r})\n"
            f"MANIFEST = Path({str(manifest_path)!r})\n"
            "with MANIFEST.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n"
            "print(f'Loaded {len(rows)} polarity-corrected Depth Pro rows from {MANIFEST}')"
        ),
        nbformat.v4.new_code_cell(
            "def resolve(path_value):\n"
            "    path = Path(path_value)\n"
            "    return path if path.is_absolute() else ROOT / path\n\n"
            "def surface_values(depth, mask=None, max_points=96):\n"
            "    if mask is None:\n"
            "        mask = np.isfinite(depth) & (depth > 0.0)\n"
            "    else:\n"
            "        mask = mask & np.isfinite(depth) & (depth > 0.0)\n"
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
            "def make_surface_comparison(row):\n"
            "    gt_depth = np.load(resolve(row['source_gt_depth_npy_path']))\n"
            "    checked_depth = np.load(resolve(row['selected_depth_path']))\n"
            "    gt_mask = np.isfinite(gt_depth) & (gt_depth > 0.0)\n"
            "    fig = make_subplots(\n"
            "        rows=1,\n"
            "        cols=2,\n"
            "        specs=[[{'type': 'surface'}, {'type': 'surface'}]],\n"
            "        subplot_titles=['GT depth', 'Depth Pro checked'],\n"
            "    )\n"
            "    for col, depth in enumerate([gt_depth, checked_depth], start=1):\n"
            "        x, y, z = surface_values(depth, gt_mask)\n"
            "        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale='Viridis', showscale=False, hoverinfo='skip'), row=1, col=col)\n"
            "    fig.update_layout(title=f\"{row['sample_id']}: polarity-checked Depth Pro\", height=560, margin=dict(l=0, r=0, t=52, b=0))\n"
            "    fig.update_scenes(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data')\n"
            "    return fig"
        ),
        nbformat.v4.new_code_cell(
            "sample_index = 0\n"
            "fig = make_surface_comparison(rows[sample_index])\n"
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


def load_raw_and_gt(row: dict[str, str], source_root: Path) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    rgb_path = resolve_manifest_path(row["source_rgb_path"], source_root)
    gt_depth_path = resolve_manifest_path(row["source_gt_depth_npy_path"], source_root)
    raw_depth_path = resolve_manifest_path(row["pred_depth_npy_path"], source_root)
    gt_depth = np.load(gt_depth_path).astype(np.float32)
    raw_depth = np.load(raw_depth_path).astype(np.float32)
    if raw_depth.shape != gt_depth.shape:
        raw_image = Image.fromarray(raw_depth, mode="F")
        raw_depth = np.asarray(
            raw_image.resize((gt_depth.shape[1], gt_depth.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    return rgb_path, gt_depth_path, raw_depth_path, gt_depth, raw_depth


def choose_conversion(source_rows: list[dict[str, str]], source_root: Path, requested: str) -> tuple[str, dict[str, float]]:
    raw_corrs: list[float] = []
    reciprocal_corrs: list[float] = []
    for row in source_rows:
        _rgb_path, _gt_depth_path, _raw_depth_path, gt_depth, raw_depth = load_raw_and_gt(row, source_root)
        gt_mask = np.isfinite(gt_depth) & (gt_depth > 0.0)
        raw_corrs.append(safe_corr(gt_depth, raw_depth, gt_mask))
        reciprocal_corrs.append(safe_corr(gt_depth, reciprocal_depth(raw_depth), gt_mask))

    raw_mean = float(np.nanmean(np.asarray(raw_corrs, dtype=np.float64)))
    reciprocal_mean = float(np.nanmean(np.asarray(reciprocal_corrs, dtype=np.float64)))
    if requested == "auto":
        selected = "raw" if raw_mean >= reciprocal_mean else "reciprocal"
    else:
        selected = requested
    return selected, {
        "raw_metric_corr_mean": raw_mean,
        "reciprocal_metric_corr_mean": reciprocal_mean,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_rows = load_rows(args.manifest, args.limit)
    selected_conversion, polarity_diagnostics = choose_conversion(source_rows, source_root, args.conversion)
    print(
        "selected conversion="
        f"{selected_conversion} "
        f"(raw corr={polarity_diagnostics['raw_metric_corr_mean']:.4f}, "
        f"reciprocal corr={polarity_diagnostics['reciprocal_metric_corr_mean']:.4f})",
        flush=True,
    )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    data_root = output_root / "data"
    checked_depth_root = data_root / "polarity_checked_depth"
    checked_vis_root = data_root / "polarity_checked_depth_vis"
    gt_vis_root = data_root / "gt_depth_vis"
    visualizations_root = output_root / "visualizations"
    for path in [checked_depth_root, checked_vis_root, gt_vis_root, visualizations_root]:
        path.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows, start=1):
        sample_id = row["sample_id"]
        rgb_path, gt_depth_path, raw_depth_path, gt_depth, raw_depth = load_raw_and_gt(row, source_root)
        checked_depth = convert_depth(raw_depth, selected_conversion)
        gt_mask = np.isfinite(gt_depth) & (gt_depth > 0.0)

        checked_depth_path = checked_depth_root / f"{sample_id}_depthpro_{selected_conversion}_depth.npy"
        checked_vis_path = checked_vis_root / f"{sample_id}_depthpro_{selected_conversion}_depth_vis.png"
        gt_vis_path = gt_vis_root / f"{sample_id}_gt_depth_vis.png"
        np.save(checked_depth_path, checked_depth)
        imageio.imwrite(gt_vis_path, depth_visual(gt_depth, gt_mask))
        imageio.imwrite(checked_vis_path, depth_visual(checked_depth, gt_mask))

        checked_valid = checked_depth[np.isfinite(checked_depth) & (checked_depth > 0.0)]
        raw_valid = raw_depth[np.isfinite(raw_depth) & (raw_depth > 0.0)]
        raw_metric_corr = safe_corr(gt_depth, raw_depth, gt_mask)
        reciprocal_metric_corr = safe_corr(gt_depth, reciprocal_depth(raw_depth), gt_mask)
        scaled_abs_rel = median_scaled_abs_rel(checked_depth, gt_depth, gt_mask)

        output_row = {
            "sample_id": sample_id,
            "subject_id": row.get("subject_id", ""),
            "view_index": row.get("view_index", ""),
            "source_rgb_path": repo_relative(rgb_path),
            "source_gt_depth_npy_path": repo_relative(gt_depth_path),
            "source_gt_depth_vis_path": repo_relative(gt_vis_path),
            "raw_depthpro_depth_path": repo_relative(raw_depth_path),
            "selected_conversion": selected_conversion,
            "selected_depth_path": repo_relative(checked_depth_path),
            "selected_depth_vis_path": repo_relative(checked_vis_path),
            "raw_depthpro_model": row.get("model", "apple/DepthPro-hf"),
            "width": int(gt_depth.shape[1]),
            "height": int(gt_depth.shape[0]),
            "raw_depth_min": float(np.min(raw_valid)) if raw_valid.size else float("nan"),
            "raw_depth_median": float(np.median(raw_valid)) if raw_valid.size else float("nan"),
            "raw_depth_max": float(np.max(raw_valid)) if raw_valid.size else float("nan"),
            "corrected_depth_min": float(np.min(checked_valid)) if checked_valid.size else float("nan"),
            "corrected_depth_median": float(np.median(checked_valid)) if checked_valid.size else float("nan"),
            "corrected_depth_max": float(np.max(checked_valid)) if checked_valid.size else float("nan"),
            "raw_metric_corr": raw_metric_corr,
            "reciprocal_metric_corr": reciprocal_metric_corr,
            "median_scaled_abs_rel": scaled_abs_rel,
            "resolved_rgb_path": str(rgb_path),
            "resolved_gt_depth_vis_path": str(gt_vis_path),
            "resolved_corrected_depth_vis_path": str(checked_vis_path),
        }
        output_rows.append(output_row)
        print(f"[{idx:03d}/{len(source_rows):03d}] wrote polarity-checked Depth Pro for {sample_id}", flush=True)

    manifest_path = data_root / "manifest.csv"
    diagnostics_path = data_root / "diagnostics.csv"
    montage_path = visualizations_root / "montage_rgb_gt_depthpro_polarity_corrected.png"
    gif_path = visualizations_root / "rgb_gt_depthpro_polarity_corrected.gif"
    notebook_path = visualizations_root / "plot_depth_pro_polarity_corrected_surfaces.ipynb"
    write_manifest(output_rows, manifest_path)
    write_diagnostics(output_rows, diagnostics_path)
    build_montage(output_rows, montage_path, args.tile_size, args.columns)
    build_gif(output_rows, gif_path)
    write_notebook(output_root, manifest_path, notebook_path)

    raw_corr_values = np.array([row["raw_metric_corr"] for row in output_rows], dtype=np.float64)
    corrected_corr_values = np.array([row["reciprocal_metric_corr"] for row in output_rows], dtype=np.float64)
    scaled_abs_rel_values = np.array([row["median_scaled_abs_rel"] for row in output_rows], dtype=np.float64)
    summary = {
        "sample_count": len(output_rows),
        "source_manifest": repo_relative(args.manifest.resolve()),
        "requested_conversion": args.conversion,
        "selected_conversion": selected_conversion,
        "correction": "polarity checked against GT depth; selected output is visualized as near_bright_far_dark",
        "diagnostics": {
            "raw_metric_corr_mean": polarity_diagnostics["raw_metric_corr_mean"],
            "reciprocal_metric_corr_mean": polarity_diagnostics["reciprocal_metric_corr_mean"],
            "median_scaled_abs_rel_mean": float(np.nanmean(scaled_abs_rel_values)),
        },
        "folders": {
            "data": repo_relative(data_root),
            "visualizations": repo_relative(visualizations_root),
        },
        "outputs": {
            "manifest": repo_relative(manifest_path),
            "diagnostics": repo_relative(diagnostics_path),
            "montage": repo_relative(montage_path),
            "gif": repo_relative(gif_path),
            "notebook": repo_relative(notebook_path),
        },
    }
    (data_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

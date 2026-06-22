#!/usr/bin/env python3
"""Export synthetic multi-lesion body-part RGB/depth pairs into data/depth_maps/body_parts."""

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
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_MANIFEST = (
    ROOT
    / "data"
    / "synthetic"
    / "body_parts_multi_lesion"
    / "data"
    / "camera_depth_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "depth_maps" / "body_parts"

BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_relative(dataset_root: Path, path: Path) -> str:
    return str(path.relative_to(dataset_root))


def resolve_root_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_rows(rows: list[dict[str, str]], limit_per_part: int | None) -> list[dict[str, str]]:
    if limit_per_part is None:
        return rows
    output = []
    for body_part in BODY_PARTS:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        output.extend(part_rows[:limit_per_part])
    return output


def copy_file(source_path: Path, destination_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def save_pair_figure(rgb_path: Path, depth_vis_path: Path, output_path: Path) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    depth_vis = Image.open(depth_vis_path).convert("L").convert("RGB")
    if depth_vis.size != rgb.size:
        depth_vis = depth_vis.resize(rgb.size, Image.Resampling.LANCZOS)
    figure = Image.new("RGB", (rgb.width * 2, rgb.height), "white")
    figure.paste(rgb, (0, 0))
    figure.paste(depth_vis, (rgb.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path)


def source_root_for_manifest(source_manifest: Path) -> Path:
    if source_manifest.parent.name == "data":
        return source_manifest.parent.parent
    return source_manifest.parent


def row_value(row: dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else value


def export_row(dataset_root: Path, row: dict[str, str], source_generation_root: Path) -> dict[str, str]:
    sample_id = row["sample_id"]
    body_part = row["body_part"]
    part_root = dataset_root / "data" / body_part
    image_path = part_root / "2d_images" / f"{sample_id}_2d.png"
    depth_npy_path = part_root / "2d_gt_depth_maps" / f"{sample_id}_gt_depth_m.npy"
    depth_png_path = part_root / "2d_gt_depth_maps" / f"{sample_id}_gt_depth_mm.png"
    depth_vis_path = part_root / "2d_gt_depth_maps" / f"{sample_id}_gt_depth_vis.png"
    figure_path = part_root / "2d_rgb_gt_depth_figures" / f"{sample_id}_2d_gt_depth.png"

    source_image_path = resolve_root_path(row["image_path"])
    source_depth_npy_path = resolve_root_path(row["depth_npy_path"])
    source_depth_png_path = resolve_root_path(row["depth_png_path"])
    source_depth_vis_path = resolve_root_path(row["depth_vis_path"])

    copy_file(source_image_path, image_path)
    copy_file(source_depth_npy_path, depth_npy_path)
    copy_file(source_depth_png_path, depth_png_path)
    copy_file(source_depth_vis_path, depth_vis_path)
    save_pair_figure(image_path, depth_vis_path, figure_path)

    out_row = {
        "sample_id": sample_id,
        "body_part": body_part,
        "source_segmentation_body_part": row_value(row, "source_segmentation_body_part", body_part),
        "scan_id": row["scan_id"],
        "patient_volume_index": row["patient_volume_index"],
        "seed": row["seed"],
        "image_path": dataset_relative(dataset_root, image_path),
        "depth_npy_path": dataset_relative(dataset_root, depth_npy_path),
        "depth_png_path": dataset_relative(dataset_root, depth_png_path),
        "depth_vis_path": dataset_relative(dataset_root, depth_vis_path),
        "figure_path": dataset_relative(dataset_root, figure_path),
        "depth_type": row["depth_type"],
        "camera_mode": row["camera_mode"],
        "body_region": row_value(row, "body_region"),
        "width": row["width"],
        "height": row["height"],
        "valid_depth_pixels": row["valid_depth_pixels"],
        "lesion_count": row_value(row, "lesion_count"),
        "lesion_count_min": row_value(row, "lesion_count_min"),
        "lesion_count_max": row_value(row, "lesion_count_max"),
        "radius_m": row_value(row, "radius_m", row_value(row, "radius_mean_m")),
        "radius_min_m": row_value(row, "radius_min_m"),
        "radius_mean_m": row_value(row, "radius_mean_m", row_value(row, "radius_m")),
        "radius_max_m": row_value(row, "radius_max_m"),
        "lesion_height_m": row_value(row, "lesion_height_m", row_value(row, "lesion_height_mean_m")),
        "lesion_height_min_m": row_value(row, "lesion_height_min_m"),
        "lesion_height_mean_m": row_value(row, "lesion_height_mean_m", row_value(row, "lesion_height_m")),
        "lesion_height_max_m": row_value(row, "lesion_height_max_m"),
        "support_radius_m": row_value(row, "support_radius_m", row_value(row, "support_radius_mean_m")),
        "support_radius_min_m": row_value(row, "support_radius_min_m"),
        "support_radius_mean_m": row_value(row, "support_radius_mean_m", row_value(row, "support_radius_m")),
        "support_radius_max_m": row_value(row, "support_radius_max_m"),
        "spherical_cap_volume_ml": row_value(row, "spherical_cap_volume_ml", row_value(row, "total_spherical_cap_volume_ml")),
        "total_spherical_cap_volume_ml": row_value(row, "total_spherical_cap_volume_ml", row_value(row, "spherical_cap_volume_ml")),
        "fov_deg": row_value(row, "fov_deg"),
        "camera_distance_m": row_value(row, "camera_distance_m"),
        "frame_half_height_m": row_value(row, "frame_half_height_m"),
        "frame_half_width_m": row_value(row, "frame_half_width_m"),
        "source_image_path": row["image_path"],
        "source_depth_npy_path": row["depth_npy_path"],
        "source_depth_png_path": row["depth_png_path"],
        "source_depth_vis_path": row["depth_vis_path"],
        "source_metadata_path": row["metadata_path"],
        "source_volume_mesh_path": row["volume_mesh_path"],
        "source_generation": root_relative(source_generation_root),
        "lesion_pattern_source": row_value(
            row,
            "lesion_pattern_source",
            "10-100 random spherical-cap NF-like lesions with interpolated skin-color texture",
        ),
    }
    return out_row


def balanced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    selected = []
    per_part = max(1, count // len(BODY_PARTS))
    for body_part in BODY_PARTS:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        if not part_rows:
            continue
        indices = np.linspace(0, len(part_rows) - 1, min(per_part, len(part_rows)), dtype=int)
        selected.extend(part_rows[int(index)] for index in indices)
    if len(selected) < count:
        selected_ids = {row["sample_id"] for row in selected}
        selected.extend(row for row in rows if row["sample_id"] not in selected_ids)
    return selected[:count]


def build_montage(dataset_root: Path, rows: list[dict[str, str]], output_path: Path, count: int, tile_height: int = 96, columns: int = 10) -> None:
    tiles = []
    for row in balanced_rows(rows, count):
        figure = Image.open(dataset_root / row["figure_path"]).convert("RGB")
        tile = figure.resize((tile_height * 2, tile_height), Image.Resampling.LANCZOS)
        tiles.append(tile)
    row_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * tile_height * 2, row_count * tile_height), "white")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def build_preview_gif(dataset_root: Path, rows: list[dict[str, str]], output_path: Path, frame_count: int = 32, tile_size: int = 192) -> None:
    frames = []
    for row in balanced_rows(rows, frame_count):
        rgb = Image.open(dataset_root / row["image_path"]).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        depth = (
            Image.open(dataset_root / row["depth_vis_path"])
            .convert("L")
            .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            .convert("RGB")
        )
        frame = Image.new("RGB", (tile_size * 2, tile_size), "white")
        frame.paste(rgb, (0, 0))
        frame.paste(depth, (tile_size, 0))
        frames.append(np.asarray(frame))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.18, loop=0)


def build_plotly_notebook(dataset_root: Path, rows: list[dict[str, str]], output_path: Path) -> None:
    row = rows[len(rows) // 2]
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Body-part synthetic lesion RGB / GT depth surface\n\n"
            "Executed Plotly notebook for inspecting one representative exported body-part sample."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"DATASET_ROOT = Path({str(dataset_root)!r})\n"
            f"SAMPLE_ID = {row['sample_id']!r}\n"
            "rows = list(csv.DictReader((DATASET_ROOT / 'data' / 'manifest.csv').open(newline='', encoding='utf-8')))\n"
            "row = next(item for item in rows if item['sample_id'] == SAMPLE_ID)\n"
            "row"
        ),
        nbformat.v4.new_code_cell(
            "depth = np.load(DATASET_ROOT / row['depth_npy_path']).astype(float)\n"
            "rgb = np.array(Image.open(DATASET_ROOT / row['image_path']).convert('RGB'))\n"
            "stride = 4\n"
            "z = depth[::stride, ::stride]\n"
            "z[~np.isfinite(z) | (z <= 0)] = np.nan\n"
            "yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]\n"
            "fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'xy'}]], column_widths=[0.62, 0.38])\n"
            "fig.add_trace(go.Surface(x=xx, y=-yy, z=z, surfacecolor=z, colorscale='Viridis', colorbar={'title': 'meters'}, connectgaps=False), row=1, col=1)\n"
            "fig.add_trace(go.Image(z=rgb), row=1, col=2)\n"
            "fig.update_layout(width=1050, height=560, title=f\"{row['body_part']} / {row['scan_id']} synthetic lesion RGB and GT depth\")\n"
            "fig.update_layout(scene={'xaxis_title': 'x pixels / stride', 'yaxis_title': 'y pixels / stride', 'zaxis_title': 'camera z distance (m)', 'aspectmode': 'data'})\n"
            "fig.update_xaxes(showticklabels=False, row=1, col=2)\n"
            "fig.update_yaxes(showticklabels=False, row=1, col=2)\n"
            "fig"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    NotebookClient(notebook, timeout=180, kernel_name="python3").execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


def write_manifests(dataset_root: Path, rows: list[dict[str, str]], source_manifest: Path, montage_path: Path, gif_path: Path, notebook_path: Path) -> dict[str, Any]:
    data_dir = dataset_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    csv_path = data_dir / "manifest.csv"
    jsonl_path = data_dir / "manifest.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    by_part = {body_part: sum(row["body_part"] == body_part for row in rows) for body_part in BODY_PARTS}
    by_scan: dict[str, int] = {}
    part_manifests = {}
    for body_part in BODY_PARTS:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        part_dir = data_dir / body_part
        part_csv_path = part_dir / "manifest.csv"
        part_jsonl_path = part_dir / "manifest.jsonl"
        with part_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(part_rows)
        with part_jsonl_path.open("w", encoding="utf-8") as handle:
            for row in part_rows:
                handle.write(json.dumps(row) + "\n")
        part_manifests[body_part] = {
            "manifest_csv": root_relative(part_csv_path),
            "manifest_jsonl": root_relative(part_jsonl_path),
            "folders": [
                root_relative(part_dir / "2d_images"),
                root_relative(part_dir / "2d_gt_depth_maps"),
                root_relative(part_dir / "2d_rgb_gt_depth_figures"),
            ],
        }
    for row in rows:
        by_scan[row["scan_id"]] = by_scan.get(row["scan_id"], 0) + 1
    lesion_counts = [int(row["lesion_count"]) for row in rows if row.get("lesion_count")]
    camera_modes = sorted({row["camera_mode"] for row in rows if row.get("camera_mode")})
    lesion_pattern_sources = sorted({row["lesion_pattern_source"] for row in rows if row.get("lesion_pattern_source")})

    summary = {
        "dataset": "body_parts",
        "output_root": root_relative(dataset_root),
        "source_manifest": root_relative(source_manifest),
        "source_generation_script": "code/data_generation/body_parts/scripts/build_body_part_multi_lesion_depth_dataset.py",
        "source_generation": root_relative(source_root_for_manifest(source_manifest)),
        "source_volume_shape": "multi_spherical_cap_nf_like",
        "lesion_pattern_source": lesion_pattern_sources[0] if len(lesion_pattern_sources) == 1 else lesion_pattern_sources,
        "camera_mode": camera_modes[0] if len(camera_modes) == 1 else camera_modes,
        "framing": "random close-up camera centered near a sampled visible lesion",
        "lesion_count_min": min(lesion_counts) if lesion_counts else None,
        "lesion_count_max": max(lesion_counts) if lesion_counts else None,
        "lesion_count_mean": float(np.mean(lesion_counts)) if lesion_counts else None,
        "pair_count": len(rows),
        "image_count": sum(1 for row in rows if (dataset_root / row["image_path"]).exists()),
        "depth_npy_count": sum(1 for row in rows if (dataset_root / row["depth_npy_path"]).exists()),
        "depth_png_count": sum(1 for row in rows if (dataset_root / row["depth_png_path"]).exists()),
        "depth_vis_count": sum(1 for row in rows if (dataset_root / row["depth_vis_path"]).exists()),
        "figure_count": sum(1 for row in rows if (dataset_root / row["figure_path"]).exists()),
        "by_part": by_part,
        "by_scan": by_scan,
        "manifest_csv": root_relative(csv_path),
        "manifest_jsonl": root_relative(jsonl_path),
        "part_manifests": part_manifests,
        "visualizations": {
            "montage_100": root_relative(montage_path),
            "gif": root_relative(gif_path),
            "plotly_notebook": root_relative(notebook_path),
        },
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (dataset_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def export_pairs(source_manifest: Path, output_root: Path, overwrite: bool, limit_per_part: int | None, montage_count: int) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "visualizations").mkdir(parents=True, exist_ok=True)

    source_rows = selected_rows(read_manifest(source_manifest), limit_per_part)
    if not source_rows:
        raise ValueError(f"No rows found in source manifest: {source_manifest}")

    source_generation_root = source_root_for_manifest(source_manifest)
    rows = [export_row(output_root, row, source_generation_root) for row in source_rows]
    montage_path = output_root / "visualizations" / f"montage_{montage_count}_rgb_gt_depth.png"
    gif_path = output_root / "visualizations" / "body_parts_rgb_gt_depth_preview.gif"
    notebook_path = output_root / "visualizations" / "body_parts_depth_surface.ipynb"
    build_montage(output_root, rows, montage_path, montage_count)
    build_preview_gif(output_root, rows, gif_path)
    write_manifests(output_root, rows, source_manifest, montage_path, gif_path, notebook_path)
    build_plotly_notebook(output_root, rows, notebook_path)
    summary = write_manifests(output_root, rows, source_manifest, montage_path, gif_path, notebook_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=root_relative(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-root", default=root_relative(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--limit-per-part", type=int, default=None, help="Optional test limit per body part.")
    parser.add_argument("--montage-count", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_manifest = resolve_root_path(args.source_manifest)
    output_root = resolve_root_path(args.output_root)
    summary = export_pairs(source_manifest, output_root, args.overwrite, args.limit_per_part, args.montage_count)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

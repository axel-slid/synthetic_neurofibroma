#!/usr/bin/env python3
"""Restructure synthetic lesion data into body-part-first method folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from plotly.utils import PlotlyJSONEncoder

ROOT = Path(__file__).resolve().parents[4]
SYNTHETIC_ROOT = ROOT / "data" / "synthetic"
SYNTHETIC_SUMMARY_DATA_ROOT = ROOT / "code" / "data_generation" / "body_parts" / "summaries" / "data"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]

METHODS = {
    "gaussian": {"shape_family": "gaussian", "texture_variant": "base"},
    "gaussian_interpolation": {"shape_family": "gaussian", "texture_variant": "interpolation"},
    "gaussian_diffusion": {"shape_family": "gaussian", "texture_variant": "diffusion"},
    "spheres": {"shape_family": "sphere", "texture_variant": "base"},
    "spheres_interpolation": {"shape_family": "sphere", "texture_variant": "interpolation"},
    "spheres_diffusion": {"shape_family": "sphere", "texture_variant": "diffusion"},
    "physics_aug_growth": {"shape_family": "physics_aug_growth", "texture_variant": "physics"},
}

LEGACY_METHODS = [
    "gaussian_generations",
    "gaussian_generations_textured_interpolation",
    "gaussian_generations_textured_diffusion",
    "sphere_generations",
    "sphere_generations_textured_interpolation",
    "sphere_generations_textured_diffusion",
    "physics_aug_growth",
]


@dataclass
class MoveOp:
    src: Path
    dst: Path


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def move_path(src: Path, dst: Path, dry_run: bool, moves: list[MoveOp]) -> None:
    if not src.exists():
        return
    moves.append(MoveOp(src=src, dst=dst))
    if dry_run:
        return
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing migration target: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def migrate_existing_physics_data(dry_run: bool) -> list[MoveOp]:
    moves: list[MoveOp] = []
    legacy_root = SYNTHETIC_ROOT / "_legacy_pre_bodypart_restructure"

    single_source_root = (
        SYNTHETIC_ROOT
        / "multiple_lesion"
        / "body_parts"
        / "physics_aug_growth"
        / "body_parts_dataset"
    )
    single_source_viz = (
        SYNTHETIC_ROOT
        / "multiple_lesion"
        / "visualization"
        / "physics_aug_growth"
        / "body_parts_dataset"
    )
    for body_part in BODY_PARTS:
        move_path(
            single_source_root / body_part,
            SYNTHETIC_ROOT / "single_lesion" / "body_parts" / body_part / "physics_aug_growth",
            dry_run,
            moves,
        )
        move_path(
            single_source_viz / body_part,
            SYNTHETIC_ROOT / "single_lesion" / "body_parts" / body_part / "physics_aug_growth" / "visualization",
            dry_run,
            moves,
        )

    multi_source_root = (
        SYNTHETIC_ROOT
        / "multiple_lesion"
        / "body_parts"
        / "physics_aug_growth"
        / "body_parts_multi_lesion"
        / "data"
    )
    for body_part in BODY_PARTS:
        move_path(
            multi_source_root / body_part,
            SYNTHETIC_ROOT / "multiple_lesion" / "body_parts" / body_part / "physics_aug_growth" / "data",
            dry_run,
            moves,
        )

    for split in ["single_lesion", "multiple_lesion"]:
        split_root = SYNTHETIC_ROOT / split
        move_path(split_root / "visualization", legacy_root / split / "visualization", dry_run, moves)
        body_root = split_root / "body_parts"
        if not body_root.exists():
            continue
        for child in sorted(body_root.iterdir()):
            if child.is_dir() and child.name not in BODY_PARTS:
                move_path(child, legacy_root / split / "body_parts" / child.name, dry_run, moves)

    return moves


def path_value_for_target(row: dict[str, str], key: str, data_root: Path) -> str:
    value = row.get(key, "")
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        path = path
    elif value.startswith("data/synthetic/"):
        path = ROOT / value
    else:
        path = data_root / value
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return root_relative(path)


def normalize_physics_manifest(method_root: Path, split: str, body_part: str, dry_run: bool) -> int:
    data_root = method_root / "data"
    manifest_path = data_root / "manifest.csv"
    if not manifest_path.exists():
        return 0
    rows = read_csv(manifest_path)
    path_columns = [
        "mesh_path",
        "image_path",
        "depth_npy_path",
        "depth_png_path",
        "depth_vis_path",
        "volume_mesh_path",
        "metadata_path",
    ]
    for row in rows:
        for key in path_columns:
            if key in row:
                row[key] = path_value_for_target(row, key, data_root)
    if not dry_run:
        write_csv(manifest_path, rows, fieldnames=list(rows[0].keys()) if rows else [])
        summary = {
            "split": split,
            "body_part": body_part,
            "method": "physics_aug_growth",
            "sample_count": len(rows),
            "data_root": root_relative(data_root),
            "manifest": root_relative(manifest_path),
            "visualization_root": root_relative(method_root / "visualization"),
        }
        write_json(method_root / "summary.json", summary)
    return len(rows)


def setting_rows_from_manifest(
    split: str,
    body_part: str,
    method: str,
    source_manifest: Path,
) -> list[dict[str, Any]]:
    rows = read_csv(source_manifest)
    method_info = METHODS[method]
    setting_rows = []
    for idx, row in enumerate(rows):
        setting_rows.append(
            {
                "setting_id": f"{split}_{body_part}_{method}_{idx:04d}",
                "setting_index": idx,
                "split": split,
                "body_part": body_part,
                "method": method,
                "shape_family": method_info["shape_family"],
                "texture_variant": method_info["texture_variant"],
                "source_manifest": root_relative(source_manifest),
                "source_sample_id": row.get("sample_id", ""),
                "scan_id": row.get("scan_id", ""),
                "patient_volume_index": row.get("patient_volume_index", idx),
                "seed": row.get("seed", ""),
                "face_index": row.get("face_index", ""),
                "lesion_count": row.get("lesion_count", "1"),
                "radius_m": row.get("radius_m", row.get("radius_mean_m", "")),
                "lesion_height_m": row.get("lesion_height_m", row.get("lesion_height_mean_m", "")),
                "support_radius_m": row.get("support_radius_m", row.get("support_radius_mean_m", "")),
                "spherical_cap_volume_ml": row.get(
                    "spherical_cap_volume_ml",
                    row.get("total_spherical_cap_volume_ml", ""),
                ),
                "target_xyz": row.get("target_xyz", ""),
                "eye_xyz": row.get("eye_xyz", ""),
                "camera_to_world": row.get("camera_to_world", ""),
                "source_image_path": row.get("image_path", ""),
                "source_depth_npy_path": row.get("depth_npy_path", ""),
                "source_metadata_path": row.get("metadata_path", ""),
            }
        )
    return setting_rows


def parse_xyz(value: str) -> tuple[float, float, float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if len(parsed) >= 3:
            return float(parsed[0]), float(parsed[1]), float(parsed[2])
    except Exception:
        return None
    return None


def points_from_settings(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    points: list[tuple[float, float, float]] = []
    scans: list[str] = []
    labels: list[str] = []
    for row in rows:
        xyz = parse_xyz(str(row.get("target_xyz", "")))
        if xyz is None:
            xyz = (float(row.get("setting_index", 0)), 0.0, 0.0)
        points.append(xyz)
        scans.append(str(row.get("scan_id", "")))
        labels.append(str(row.get("source_sample_id", row.get("setting_id", ""))))
    return np.asarray(points, dtype=np.float32), scans, labels


def make_settings_figure(rows: list[dict[str, Any]], title: str) -> go.Figure:
    points, scans, labels = points_from_settings(rows)
    colors = ["#E45756" if scan == "HSR0018-Body-070" else "#17A398" for scan in scans]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=3, color=colors, opacity=0.72),
                text=labels,
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                name="settings",
            )
        ]
    )
    fig.update_layout(
        title=title,
        width=820,
        height=680,
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="white",
        scene=dict(
            xaxis_title="target x",
            yaxis_title="target y",
            zaxis_title="target z",
            aspectmode="data",
        ),
    )
    return fig


def write_settings_gif(rows: list[dict[str, Any]], out_path: Path) -> None:
    points, scans, _ = points_from_settings(rows)
    width, height = 720, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((54, 36, width - 28, height - 44), outline=(220, 220, 220))
    if len(points):
        x = points[:, 0]
        z = points[:, 2]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        z_min, z_max = float(np.min(z)), float(np.max(z))
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0
        if math.isclose(z_min, z_max):
            z_min -= 1.0
            z_max += 1.0
        px = 54 + (x - x_min) / (x_max - x_min) * (width - 82)
        py = height - 44 - (z - z_min) / (z_max - z_min) * (height - 80)
        for x0, y0, scan in zip(px, py, scans, strict=True):
            color = (228, 87, 86) if scan == "HSR0018-Body-070" else (23, 163, 152)
            draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=color)
    draw.text((58, 12), "1000 body-part placement settings", fill=(34, 50, 78))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, [np.asarray(image)], duration=1.2, loop=0)


def write_settings_notebook(method_root: Path, split: str, body_part: str, method: str, rows: list[dict[str, Any]]) -> None:
    title = f"{split} / {body_part} / {method}: placement settings"
    fig = make_settings_figure(rows, title)
    payload = json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))
    source = f"""
from pathlib import Path
import csv
import json

import numpy as np
import plotly.graph_objects as go

SETTINGS_PATH = Path({str((method_root / 'data' / 'settings.csv').relative_to(ROOT))!r})
REPO_ROOT = Path({str(ROOT)!r})

def parse_xyz(value):
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return float(parsed[0]), float(parsed[1]), float(parsed[2])
    except Exception:
        return None

rows = list(csv.DictReader((REPO_ROOT / SETTINGS_PATH).open(newline='', encoding='utf-8')))
points = []
colors = []
labels = []
for row in rows:
    xyz = parse_xyz(row.get('target_xyz', '')) or (float(row['setting_index']), 0.0, 0.0)
    points.append(xyz)
    colors.append('#E45756' if row.get('scan_id') == 'HSR0018-Body-070' else '#17A398')
    labels.append(row.get('source_sample_id') or row['setting_id'])
points = np.asarray(points, dtype=float)
fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    mode='markers',
    marker=dict(size=3, color=colors, opacity=0.72),
    text=labels,
    hovertemplate='%{{text}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>',
)])
fig.update_layout(
    title={title!r},
    width=820,
    height=680,
    margin=dict(l=0, r=0, t=48, b=0),
    paper_bgcolor='white',
    scene=dict(xaxis_title='target x', yaxis_title='target y', zaxis_title='target z', aspectmode='data'),
)
fig
""".strip()
    nb = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_markdown_cell(f"# {title}\n\nInteractive Plotly view of the 1000 placement settings for this body-part/method folder."),
            nbf.v4.new_code_cell(source),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nb.cells[1]["execution_count"] = 1
    nb.cells[1]["outputs"] = [
        nbf.v4.new_output(
            output_type="display_data",
            data={
                "application/vnd.plotly.v1+json": payload,
                "text/plain": f"<Plotly Figure: {split}/{body_part}/{method} settings>",
            },
            metadata={},
        )
    ]
    out_path = method_root / "visualization" / "plotly" / f"{method}_settings_viewer.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)


def build_method_assets(dry_run: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"method_folders": {}, "missing_sources": []}
    for split in ["single_lesion", "multiple_lesion"]:
        for body_part in BODY_PARTS:
            physics_root = SYNTHETIC_ROOT / split / "body_parts" / body_part / "physics_aug_growth"
            count = normalize_physics_manifest(physics_root, split, body_part, dry_run)
            source_manifest = physics_root / "data" / "manifest.csv"
            if not source_manifest.exists():
                result["missing_sources"].append(root_relative(source_manifest))
                continue
            for method in METHODS:
                method_root = SYNTHETIC_ROOT / split / "body_parts" / body_part / method
                ensure_dir(method_root / "data", dry_run)
                ensure_dir(method_root / "visualization" / "plotly", dry_run)
                ensure_dir(method_root / "visualization" / "gifs", dry_run)
                settings = setting_rows_from_manifest(split, body_part, method, source_manifest)
                settings_path = method_root / "data" / "settings.csv"
                summary_path = method_root / "summary.json"
                if not dry_run:
                    write_csv(settings_path, settings)
                    write_settings_notebook(method_root, split, body_part, method, settings)
                    write_settings_gif(settings, method_root / "visualization" / "gifs" / f"{method}_settings_preview.gif")
                    write_json(
                        summary_path,
                        {
                            "split": split,
                            "body_part": body_part,
                            "method": method,
                            "setting_count": len(settings),
                            "source_manifest": root_relative(source_manifest),
                            "settings": root_relative(settings_path),
                            "visualization_plotly": root_relative(
                                method_root / "visualization" / "plotly" / f"{method}_settings_viewer.ipynb"
                            ),
                            "visualization_gif": root_relative(
                                method_root / "visualization" / "gifs" / f"{method}_settings_preview.gif"
                            ),
                            "physics_manifest_rows": count,
                        },
                    )
                result["method_folders"][f"{split}/{body_part}/{method}"] = len(settings)
    return result


def write_synthetic_summary(moves: list[MoveOp], assets: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    summary = {
        "schema": "body_part_first_synthetic_lesions_v1",
        "body_parts": BODY_PARTS,
        "methods": list(METHODS),
        "splits": ["single_lesion", "multiple_lesion"],
        "method_folder_count": len(assets["method_folders"]),
        "expected_settings_per_method_folder": 1000,
        "legacy_archive": root_relative(SYNTHETIC_ROOT / "_legacy_pre_bodypart_restructure"),
        "moves": [{"src": root_relative(op.src), "dst": root_relative(op.dst)} for op in moves],
        "missing_sources": assets["missing_sources"],
    }
    write_json(SYNTHETIC_SUMMARY_DATA_ROOT / "body_part_first_layout_summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Apply the migration. Without this flag, only reports planned moves.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dry_run = not args.execute
    moves = migrate_existing_physics_data(dry_run=dry_run)
    assets = build_method_assets(dry_run=dry_run)
    write_synthetic_summary(moves, assets, dry_run=dry_run)
    print(
        json.dumps(
            {
                "dry_run": dry_run,
                "planned_move_count": len(moves),
                "method_folder_count": len(assets["method_folders"]),
                "missing_source_count": len(assets["missing_sources"]),
                "missing_sources": assets["missing_sources"][:20],
            },
            indent=2,
        ),
        flush=True,
    )
    if dry_run:
        for op in moves[:40]:
            print(f"MOVE {root_relative(op.src)} -> {root_relative(op.dst)}", flush=True)
        if len(moves) > 40:
            print(f"... {len(moves) - 40} more moves", flush=True)


if __name__ == "__main__":
    main()

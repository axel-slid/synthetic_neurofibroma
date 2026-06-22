#!/usr/bin/env python3
"""Repair body-part-first synthetic folders with RGB/depth pairs and mesh viewers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from nbclient import NotebookClient
from PIL import Image, ImageDraw
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[4]
SYNTHETIC_ROOT = ROOT / "data" / "synthetic"
SYNTHETIC_SUMMARY_DATA_ROOT = ROOT / "code" / "data_generation" / "body_parts" / "summaries" / "data"
HSR_MESH_ROOT = ROOT / "data" / "hsr" / "visualizations" / "meshes"
BODY_MESH_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

SPLITS = ["single_lesion", "multiple_lesion"]
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

PATH_COLUMNS = [
    "image_path",
    "depth_npy_path",
    "depth_png_path",
    "depth_vis_path",
    "metadata_path",
    "mesh_path",
    "volume_mesh_path",
]

SUBDIR_BY_COLUMN = {
    "image_path": "images",
    "depth_npy_path": "depth",
    "depth_png_path": "depth",
    "depth_vis_path": "depth_vis",
    "metadata_path": "metadata",
    "mesh_path": "volumes",
    "volume_mesh_path": "volumes",
}


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
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(value: str, data_root: Path, column: str) -> Path:
    if not value:
        raise ValueError(f"Empty path for {column} under {data_root}")
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([data_root / raw, ROOT / raw])
        subdir = SUBDIR_BY_COLUMN.get(column)
        if subdir:
            candidates.append(data_root / subdir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {column}={value!r}. Tried: {joined}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.samefile(dst):
                return
        except FileNotFoundError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def normalize_source_manifest(source_data_root: Path) -> list[dict[str, str]]:
    manifest_path = source_data_root / "manifest.csv"
    rows = read_csv(manifest_path)
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        fixed = dict(row)
        for column in PATH_COLUMNS:
            if column not in fixed or not fixed[column]:
                continue
            resolved = resolve_path(fixed[column], source_data_root, column)
            fixed[column] = str(resolved.relative_to(source_data_root))
        if "volume_mesh_path" not in fixed and fixed.get("mesh_path"):
            fixed["volume_mesh_path"] = fixed["mesh_path"]
        normalized_rows.append(fixed)
    write_csv(manifest_path, normalized_rows)
    write_csv(source_data_root / "camera_depth_manifest.csv", normalized_rows)
    return normalized_rows


def fieldnames_for(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def build_target_pair_rows(
    source_rows: list[dict[str, str]],
    source_data_root: Path,
    target_data_root: Path,
    split: str,
    body_part: str,
    method: str,
) -> list[dict[str, Any]]:
    method_info = METHODS[method]
    target_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows):
        target_row: dict[str, Any] = dict(source_row)
        target_row.update(
            {
                "pair_index": index,
                "split": split,
                "body_part": body_part,
                "method": method,
                "shape_family": method_info["shape_family"],
                "texture_variant": method_info["texture_variant"],
                "source_pair_manifest": root_relative(source_data_root / "manifest.csv"),
                "source_sample_id": source_row.get("sample_id", ""),
            }
        )
        for column in PATH_COLUMNS:
            if column not in source_row or not source_row[column]:
                continue
            source_path = resolve_path(source_row[column], source_data_root, column)
            subdir = SUBDIR_BY_COLUMN[column]
            target_path = target_data_root / subdir / source_path.name
            link_or_copy(source_path, target_path)
            relative = str(target_path.relative_to(target_data_root))
            target_row[column] = relative
            if column == "mesh_path":
                target_row["volume_mesh_path"] = relative
            if column == "volume_mesh_path":
                target_row["mesh_path"] = relative
        target_rows.append(target_row)
    return target_rows


def update_settings(
    method_root: Path,
    pair_rows: list[dict[str, Any]],
    split: str,
    body_part: str,
    method: str,
) -> list[dict[str, Any]]:
    settings_path = method_root / "data" / "settings.csv"
    existing = read_csv(settings_path) if settings_path.exists() else []
    method_info = METHODS[method]
    settings: list[dict[str, Any]] = []
    for index, pair_row in enumerate(pair_rows):
        row = dict(existing[index]) if index < len(existing) else {}
        row.update(
            {
                "setting_id": row.get("setting_id") or f"{split}_{body_part}_{method}_{index:04d}",
                "setting_index": index,
                "split": split,
                "body_part": body_part,
                "method": method,
                "shape_family": method_info["shape_family"],
                "texture_variant": method_info["texture_variant"],
                "sample_id": pair_row.get("sample_id", ""),
                "scan_id": pair_row.get("scan_id", ""),
                "lesion_count": pair_row.get("lesion_count", row.get("lesion_count", "1")),
                "radius_m": pair_row.get("radius_m", row.get("radius_m", "")),
                "lesion_height_m": pair_row.get("lesion_height_m", row.get("lesion_height_m", "")),
                "support_radius_m": pair_row.get("support_radius_m", row.get("support_radius_m", "")),
                "target_xyz": pair_row.get("target_xyz", row.get("target_xyz", "")),
                "eye_xyz": pair_row.get("eye_xyz", row.get("eye_xyz", "")),
                "image_path": pair_row.get("image_path", ""),
                "depth_npy_path": pair_row.get("depth_npy_path", ""),
                "depth_png_path": pair_row.get("depth_png_path", ""),
                "depth_vis_path": pair_row.get("depth_vis_path", ""),
                "metadata_path": pair_row.get("metadata_path", ""),
                "volume_mesh_path": pair_row.get("volume_mesh_path", pair_row.get("mesh_path", "")),
                "camera_depth_manifest": "camera_depth_manifest.csv",
            }
        )
        settings.append(row)
    write_csv(settings_path, settings)
    return settings


def read_ply_mesh(path: Path, max_faces: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    props = {prop.name for prop in vertex.properties}
    if {"red", "green", "blue"}.issubset(props):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 190, dtype=np.uint8)

    faces_raw = ply["face"].data["vertex_indices"]
    triangles: list[tuple[int, int, int]] = []
    for face in faces_raw:
        values = list(face)
        if len(values) < 3:
            continue
        for offset in range(1, len(values) - 1):
            triangles.append((values[0], values[offset], values[offset + 1]))
    faces = np.asarray(triangles, dtype=np.int32)

    if max_faces is not None and max_faces > 0 and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
        used = np.unique(faces.reshape(-1))
        remap = np.full(len(xyz), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        xyz = xyz[used]
        rgb = rgb[used]
        faces = remap[faces]
    return xyz, faces, rgb


def rgb_strings(rgb: np.ndarray) -> list[str]:
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]


def read_body_mesh(scan_id: str, max_body_faces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (scan_id, max_body_faces)
    if key not in BODY_MESH_CACHE:
        BODY_MESH_CACHE[key] = read_ply_mesh(
            HSR_MESH_ROOT / f"{scan_id}_closed_textured_mesh.ply",
            max_faces=max_body_faces,
        )
    return BODY_MESH_CACHE[key]


def mesh_trace_from_arrays(
    xyz: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    name: str,
    visible: bool,
    opacity: float = 1.0,
    lighting: dict[str, float] | None = None,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=rgb_strings(rgb),
        flatshading=False,
        name=name,
        visible=visible,
        opacity=opacity,
        lighting=lighting or dict(ambient=0.92, diffuse=0.6, roughness=0.9, specular=0.04),
        lightposition=dict(x=0, y=-2, z=2),
        hoverinfo="skip",
        showlegend=False,
    )


def plotly_sample_count_for_split(split: str, requested_sample_count: int) -> int:
    if split == "single_lesion":
        return 1
    return requested_sample_count


def sample_records_by_scan(pair_rows: list[dict[str, Any]], sample_count: int) -> dict[str, list[dict[str, Any]]]:
    by_scan: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        by_scan.setdefault(str(row.get("scan_id", "")), []).append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    for scan_id, rows in by_scan.items():
        rows.sort(key=lambda row: int(row.get("patient_volume_index") or row.get("pair_index") or 0))
        selected[scan_id] = rows[:sample_count]
    return selected


def combine_lesion_meshes(method_root: Path, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyzs: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    rgbs: list[np.ndarray] = []
    offset = 0
    for row in rows:
        mesh_rel = row.get("volume_mesh_path") or row.get("mesh_path")
        xyz, faces, rgb = read_ply_mesh(method_root / "data" / str(mesh_rel))
        xyzs.append(xyz)
        faces_list.append(faces + offset)
        rgbs.append(rgb)
        offset += len(xyz)
    if not xyzs:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(xyzs), np.vstack(faces_list), np.vstack(rgbs)


def normalize_to_body(body_xyz: np.ndarray, lesion_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = body_xyz.mean(axis=0)
    scale = float(np.max(np.ptp(body_xyz, axis=0)))
    if scale <= 0:
        scale = 1.0
    return (body_xyz - center) / scale, (lesion_xyz - center) / scale


def camera_for_body_part(body_part: str) -> dict[str, dict[str, float]]:
    if body_part == "back":
        return {"eye": {"x": 0.0, "y": -2.15, "z": 0.45}, "center": {"x": 0, "y": 0, "z": 0.04}}
    if body_part == "face":
        return {"eye": {"x": 0.0, "y": 2.05, "z": 0.70}, "center": {"x": 0, "y": 0, "z": 0.15}}
    if body_part == "hands":
        return {"eye": {"x": 0.55, "y": 2.05, "z": 0.35}, "center": {"x": 0, "y": 0, "z": -0.02}}
    if body_part == "feet":
        return {"eye": {"x": 0.25, "y": 2.10, "z": 0.18}, "center": {"x": 0, "y": 0, "z": -0.24}}
    return {"eye": {"x": 0.0, "y": 2.15, "z": 0.45}, "center": {"x": 0, "y": 0, "z": 0.04}}


def build_reconstruction_figure(
    method_root: Path,
    split: str,
    body_part: str,
    method: str,
    pair_rows: list[dict[str, Any]],
    sample_count: int,
    max_body_faces: int,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    records_by_scan = sample_records_by_scan(pair_rows, sample_count)
    if not records_by_scan:
        raise ValueError(f"No pair rows for {split}/{body_part}/{method}")

    traces: list[Any] = []
    records: list[dict[str, Any]] = []
    trace_ranges: list[tuple[str, int, int, int]] = []
    scans = sorted(records_by_scan)
    initial_scan = scans[0]
    for scan_id in scans:
        selected_rows = records_by_scan[scan_id]
        visible = scan_id == initial_scan
        start = len(traces)
        body_xyz, body_faces, body_rgb = read_body_mesh(scan_id, max_body_faces)
        lesion_xyz, lesion_faces, lesion_rgb = combine_lesion_meshes(method_root, selected_rows)
        body_plot_xyz, lesion_plot_xyz = normalize_to_body(body_xyz, lesion_xyz)
        traces.extend(
            [
                mesh_trace_from_arrays(
                    body_plot_xyz,
                    body_faces,
                    body_rgb,
                    name=f"{scan_id} colored mesh",
                    visible=visible,
                    opacity=1.0,
                    lighting=dict(ambient=0.92, diffuse=0.6, roughness=0.9, specular=0.04),
                ),
                mesh_trace_from_arrays(
                    lesion_plot_xyz,
                    lesion_faces,
                    lesion_rgb,
                    name=f"{len(selected_rows)} {body_part} lesion volumes",
                    visible=visible,
                    opacity=1.0,
                    lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                ),
            ]
        )
        trace_ranges.append((scan_id, start, 2, len(selected_rows)))
        records.extend(selected_rows)

    buttons = []
    for scan_id, start, count, volume_count in trace_ranges:
        visibility = [False] * len(traces)
        for trace_idx in range(start, start + count):
            visibility[trace_idx] = True
        title = f"{scan_id} baked textured closed mesh with {volume_count} {body_part} {method} lesion volumes"
        buttons.append(
            {
                "label": scan_id,
                "method": "update",
                "args": [{"visible": visibility}, {"title": title}],
            }
        )

    initial_count = trace_ranges[0][3]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{initial_scan} baked textured closed mesh with {initial_count} {body_part} {method} lesion volumes",
        width=980,
        height=780,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="white",
        showlegend=False,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.02,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
            }
        ],
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera_for_body_part(body_part),
            bgcolor="rgb(242, 244, 247)",
            aspectmode="data",
        ),
    )
    return fig, records


def write_reconstruction_notebook(
    method_root: Path,
    split: str,
    body_part: str,
    method: str,
    pair_rows: list[dict[str, Any]],
    sample_count: int,
    max_body_faces: int,
) -> tuple[Path, Path]:
    records_by_scan = sample_records_by_scan(pair_rows, sample_count)
    records = [row for scan_id in sorted(records_by_scan) for row in records_by_scan[scan_id]]
    manifest_payload = {
        "split": split,
        "body_part": body_part,
        "method": method,
        "pair_manifest": root_relative(method_root / "data" / "camera_depth_manifest.csv"),
        "record_count": len(records),
        "records": [
            {
                "sample_id": row.get("sample_id", ""),
                "scan_id": row.get("scan_id", ""),
                "volume_mesh_path": row.get("volume_mesh_path", row.get("mesh_path", "")),
                "image_path": row.get("image_path", ""),
                "depth_npy_path": row.get("depth_npy_path", ""),
                "depth_vis_path": row.get("depth_vis_path", ""),
            }
            for row in records
        ],
    }

    viewer_name = f"{method}_closed_body_lesion_viewer.ipynb"
    manifest_name = f"{method}_closed_body_lesion_manifest.json"
    out_dir = method_root / "visualization" / "plotly"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / manifest_name
    write_json(manifest_path, manifest_payload)

    records_literal = json.dumps(manifest_payload["records"], indent=2)
    source = f"""
from pathlib import Path
import json

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plyfile import PlyData

pio.renderers.default = 'notebook'

REPO_ROOT = Path({str(ROOT)!r})
METHOD_ROOT = REPO_ROOT / {root_relative(method_root)!r}
BODY_MESH_ROOT = REPO_ROOT / {root_relative(HSR_MESH_ROOT)!r}
RECORDS = {records_literal}
BODY_PART = {body_part!r}
METHOD = {method!r}
VOLUMES_PER_SCAN = {sample_count}

def read_ply_mesh(path, max_faces=None):
    ply = PlyData.read(str(path))
    vertex = ply['vertex']
    xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(float)
    props = {{prop.name for prop in vertex.properties}}
    if {{'red', 'green', 'blue'}}.issubset(props):
        rgb = np.column_stack([vertex['red'], vertex['green'], vertex['blue']]).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 190, dtype=np.uint8)
    triangles = []
    for face in ply['face'].data['vertex_indices']:
        face = list(face)
        for offset in range(1, len(face) - 1):
            triangles.append((face[0], face[offset], face[offset + 1]))
    faces = np.asarray(triangles, dtype=np.int32)
    if max_faces is not None and max_faces > 0 and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
        used = np.unique(faces.reshape(-1))
        remap = np.full(len(xyz), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        xyz = xyz[used]
        rgb = rgb[used]
        faces = remap[faces]
    return xyz, faces, rgb

def rgb_strings(rgb):
    return [f"rgb({{int(r)}},{{int(g)}},{{int(b)}})" for r, g, b in rgb]

def mesh_trace(xyz, faces, rgb, name, visible, opacity=1.0, lighting=None):
    return go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=rgb_strings(rgb),
        flatshading=False,
        name=name,
        visible=visible,
        opacity=opacity,
        lighting=lighting or dict(ambient=0.92, diffuse=0.6, roughness=0.9, specular=0.04),
        lightposition=dict(x=0, y=-2, z=2),
        hoverinfo='skip',
        showlegend=False,
    )

def normalize_to_body(body_xyz, lesion_xyz):
    center = body_xyz.mean(axis=0)
    scale = float(np.max(np.ptp(body_xyz, axis=0))) or 1.0
    return (body_xyz - center) / scale, (lesion_xyz - center) / scale

def combine_lesions(rows):
    xyzs, faces_list, rgbs = [], [], []
    offset = 0
    for row in rows:
        xyz, faces, rgb = read_ply_mesh(METHOD_ROOT / 'data' / row['volume_mesh_path'])
        xyzs.append(xyz)
        faces_list.append(faces + offset)
        rgbs.append(rgb)
        offset += len(xyz)
    return np.vstack(xyzs), np.vstack(faces_list), np.vstack(rgbs)

def camera_for_body_part(body_part):
    if body_part == 'back':
        return dict(eye=dict(x=0.0, y=-2.15, z=0.45), center=dict(x=0, y=0, z=0.04))
    if body_part == 'face':
        return dict(eye=dict(x=0.0, y=2.05, z=0.70), center=dict(x=0, y=0, z=0.15))
    if body_part == 'hands':
        return dict(eye=dict(x=0.55, y=2.05, z=0.35), center=dict(x=0, y=0, z=-0.02))
    if body_part == 'feet':
        return dict(eye=dict(x=0.25, y=2.10, z=0.18), center=dict(x=0, y=0, z=-0.24))
    return dict(eye=dict(x=0.0, y=2.15, z=0.45), center=dict(x=0, y=0, z=0.04))

def make_combined_dropdown_figure(max_body_faces={max_body_faces}):
    rows_by_scan = {{}}
    for record in RECORDS:
        rows_by_scan.setdefault(record['scan_id'], []).append(record)
    traces = []
    trace_ranges = []
    for scan_id in sorted(rows_by_scan):
        rows = rows_by_scan[scan_id][:VOLUMES_PER_SCAN]
        visible = not trace_ranges
        start = len(traces)
        body_xyz, body_faces, body_rgb = read_ply_mesh(BODY_MESH_ROOT / f"{{scan_id}}_closed_textured_mesh.ply", max_faces=max_body_faces)
        lesion_xyz, lesion_faces, lesion_rgb = combine_lesions(rows)
        body_xyz, lesion_xyz = normalize_to_body(body_xyz, lesion_xyz)
        traces.extend([
            mesh_trace(
                body_xyz, body_faces, body_rgb,
                f"{{scan_id}} colored mesh",
                visible=visible,
                opacity=1.0,
                lighting=dict(ambient=0.92, diffuse=0.6, roughness=0.9, specular=0.04),
            ),
            mesh_trace(
                lesion_xyz, lesion_faces, lesion_rgb,
                f"{{len(rows)}} {{BODY_PART}} lesion volumes",
                visible=visible,
                opacity=1.0,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            ),
        ])
        trace_ranges.append((scan_id, start, 2, len(rows)))
        print(
            f"{{scan_id}}: {{len(body_xyz):,}} closed baked-color vertices, "
            f"{{len(body_faces):,}} faces, mean RGB={{np.round(body_rgb.mean(axis=0), 1).tolist()}}, "
            f"{{len(rows)}} {{BODY_PART}} {{METHOD}} lesion volumes"
        )
    buttons = []
    for scan_id, start, count, volume_count in trace_ranges:
        visible = [False] * len(traces)
        for trace_idx in range(start, start + count):
            visible[trace_idx] = True
        buttons.append(dict(
            label=scan_id,
            method='update',
            args=[
                {{'visible': visible}},
                {{'title': f"{{scan_id}} baked textured closed mesh with {{volume_count}} {{BODY_PART}} {{METHOD}} lesion volumes"}},
            ],
        ))
    initial_scan, _, _, initial_count = trace_ranges[0]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{{initial_scan}} baked textured closed mesh with {{initial_count}} {{BODY_PART}} {{METHOD}} lesion volumes",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera_for_body_part(BODY_PART),
            bgcolor='rgb(242, 244, 247)',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=44, b=0),
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            x=0.02,
            xanchor='left',
            y=0.98,
            yanchor='top',
        )
        ],
        width=980,
        height=780,
        showlegend=False,
        paper_bgcolor='white',
    )
    return fig

combined_fig = make_combined_dropdown_figure()
combined_fig
""".strip()

    nb = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_markdown_cell("## Combined Dropdown Viewer"),
            nbf.v4.new_code_cell(source),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    notebook_path = out_dir / viewer_name
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(out_dir)}},
    ).execute()
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = ""
    nbf.write(nb, notebook_path)
    return notebook_path, manifest_path


def write_rgb_depth_gif(method_root: Path, method: str, pair_rows: list[dict[str, Any]], frame_count: int = 12) -> Path:
    out_path = method_root / "visualization" / "gifs" / f"{method}_rgb_depth_preview.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    tile_size = 180
    label_height = 26
    font_color = (22, 27, 34)
    for row in pair_rows[:frame_count]:
        rgb_path = method_root / "data" / str(row["image_path"])
        depth_path = method_root / "data" / str(row["depth_vis_path"])
        rgb = Image.open(rgb_path).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        depth = Image.open(depth_path).convert("L").resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert("RGB")
        frame = Image.new("RGB", (tile_size * 2, tile_size + label_height), "white")
        frame.paste(rgb, (0, label_height))
        frame.paste(depth, (tile_size, label_height))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 7), "RGB", fill=font_color)
        draw.text((tile_size + 8, 7), "Depth", fill=font_color)
        frames.append(np.asarray(frame))
    imageio.mimsave(out_path, frames, duration=0.55, loop=0)
    return out_path


def clear_old_visualizations(method_root: Path, method: str) -> None:
    expected = {
        method_root / "visualization" / "plotly" / f"{method}_closed_body_lesion_viewer.ipynb",
        method_root / "visualization" / "plotly" / f"{method}_closed_body_lesion_manifest.json",
        method_root / "visualization" / "gifs" / f"{method}_rgb_depth_preview.gif",
    }
    for subdir in [method_root / "visualization" / "plotly", method_root / "visualization" / "gifs"]:
        if not subdir.exists():
            continue
        for path in subdir.iterdir():
            if path.is_file() and path not in expected:
                path.unlink()


def repair_method_folder(
    split: str,
    body_part: str,
    method: str,
    source_rows: list[dict[str, str]],
    source_data_root: Path,
    sample_count: int,
    max_body_faces: int,
) -> dict[str, Any]:
    method_root = SYNTHETIC_ROOT / split / "body_parts" / body_part / method
    data_root = method_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    clear_old_visualizations(method_root, method)
    plotly_sample_count = plotly_sample_count_for_split(split, sample_count)

    pair_rows = build_target_pair_rows(
        source_rows=source_rows,
        source_data_root=source_data_root,
        target_data_root=data_root,
        split=split,
        body_part=body_part,
        method=method,
    )
    write_csv(data_root / "camera_depth_manifest.csv", pair_rows)
    settings_rows = update_settings(method_root, pair_rows, split, body_part, method)
    notebook_path, plotly_manifest_path = write_reconstruction_notebook(
        method_root=method_root,
        split=split,
        body_part=body_part,
        method=method,
        pair_rows=pair_rows,
        sample_count=plotly_sample_count,
        max_body_faces=max_body_faces,
    )
    gif_path = write_rgb_depth_gif(method_root, method, pair_rows)
    summary = {
        "split": split,
        "body_part": body_part,
        "method": method,
        "setting_count": len(settings_rows),
        "rgb_depth_pair_count": len(pair_rows),
        "image_count": len(list((data_root / "images").glob("*.png"))),
        "depth_npy_count": len(list((data_root / "depth").glob("*_depth.npy"))),
        "depth_png_count": len(list((data_root / "depth").glob("*_depth_mm.png"))),
        "depth_vis_count": len(list((data_root / "depth_vis").glob("*.png"))),
        "volume_mesh_count": len(list((data_root / "volumes").glob("*.ply"))),
        "settings": root_relative(data_root / "settings.csv"),
        "camera_depth_manifest": root_relative(data_root / "camera_depth_manifest.csv"),
        "source_pair_manifest": root_relative(source_data_root / "manifest.csv"),
        "visualization_plotly": root_relative(notebook_path),
        "visualization_plotly_manifest": root_relative(plotly_manifest_path),
        "visualization_gif": root_relative(gif_path),
        "visualization_type": "closed_body_mesh_with_body_part_lesion_volumes",
        "plotly_lesion_volumes_per_scan": plotly_sample_count,
        "pair_storage": "hardlink_or_copy_from_same_split_body_part_physics_aug_growth",
    }
    write_json(method_root / "summary.json", summary)
    return summary


def repair_existing_method_visualization(
    split: str,
    body_part: str,
    method: str,
    sample_count: int,
    max_body_faces: int,
) -> dict[str, Any] | None:
    method_root = SYNTHETIC_ROOT / split / "body_parts" / body_part / method
    data_root = method_root / "data"
    pair_manifest_path = data_root / "camera_depth_manifest.csv"
    if not pair_manifest_path.exists():
        return None

    pair_rows = read_csv(pair_manifest_path)
    settings_path = data_root / "settings.csv"
    settings_rows = read_csv(settings_path) if settings_path.exists() else []
    plotly_sample_count = plotly_sample_count_for_split(split, sample_count)
    notebook_path, plotly_manifest_path = write_reconstruction_notebook(
        method_root=method_root,
        split=split,
        body_part=body_part,
        method=method,
        pair_rows=pair_rows,
        sample_count=plotly_sample_count,
        max_body_faces=max_body_faces,
    )
    gif_path = write_rgb_depth_gif(method_root, method, pair_rows)
    summary = {
        "split": split,
        "body_part": body_part,
        "method": method,
        "setting_count": len(settings_rows),
        "rgb_depth_pair_count": len(pair_rows),
        "image_count": len(list((data_root / "images").glob("*.png"))),
        "depth_npy_count": len(list((data_root / "depth").glob("*_depth.npy"))),
        "depth_png_count": len(list((data_root / "depth").glob("*_depth_mm.png"))),
        "depth_vis_count": len(list((data_root / "depth_vis").glob("*.png"))),
        "volume_mesh_count": len(list((data_root / "volumes").glob("*.ply"))),
        "settings": root_relative(settings_path),
        "camera_depth_manifest": root_relative(pair_manifest_path),
        "source_pair_manifest": pair_rows[0].get("source_pair_manifest", "") if pair_rows else "",
        "visualization_plotly": root_relative(notebook_path),
        "visualization_plotly_manifest": root_relative(plotly_manifest_path),
        "visualization_gif": root_relative(gif_path),
        "visualization_type": "closed_body_mesh_with_body_part_lesion_volumes",
        "plotly_lesion_volumes_per_scan": plotly_sample_count,
        "pair_storage": "existing_camera_depth_manifest",
    }
    write_json(method_root / "summary.json", summary)
    return summary


def repair_all(sample_count: int, max_body_faces: int) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in SPLITS:
        for body_part in BODY_PARTS:
            source_data_root = SYNTHETIC_ROOT / split / "body_parts" / body_part / "physics_aug_growth" / "data"
            source_manifest = source_data_root / "manifest.csv"
            if not source_manifest.exists():
                missing.append(root_relative(source_manifest))
                for method in METHODS:
                    existing_summary = repair_existing_method_visualization(
                        split=split,
                        body_part=body_part,
                        method=method,
                        sample_count=sample_count,
                        max_body_faces=max_body_faces,
                    )
                    if existing_summary is not None:
                        summaries.append(existing_summary)
                continue
            source_rows = normalize_source_manifest(source_data_root)
            for method in METHODS:
                summaries.append(
                    repair_method_folder(
                        split=split,
                        body_part=body_part,
                        method=method,
                        source_rows=source_rows,
                        source_data_root=source_data_root,
                        sample_count=sample_count,
                        max_body_faces=max_body_faces,
                    )
                )
    output = {
        "schema": "body_part_first_synthetic_assets_v2",
        "splits": SPLITS,
        "body_parts": BODY_PARTS,
        "methods": list(METHODS),
        "method_folder_count": len(summaries),
        "expected_rgb_depth_pairs_per_method_folder": 1000,
        "expected_settings_per_method_folder": 1000,
        "plotly_viewer": "hsr_style_combined_dropdown_closed_body_mesh_with_lesion_volumes",
        "requested_notebook_lesion_volumes_per_scan": sample_count,
        "notebook_lesion_volumes_per_scan_by_split": {
            split: plotly_sample_count_for_split(split, sample_count) for split in SPLITS
        },
        "max_body_faces_per_plotly_body_trace": max_body_faces,
        "missing_sources": missing,
        "summaries": summaries,
    }
    write_json(SYNTHETIC_SUMMARY_DATA_ROOT / "body_part_first_assets_summary.json", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=12, help="Number of lesion volumes per scan to include in each Plotly notebook.")
    parser.add_argument("--max-body-faces", type=int, default=0, help="Maximum body mesh faces embedded per scan trace. Use 0 for the full HSR mesh.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = repair_all(sample_count=args.sample_count, max_body_faces=args.max_body_faces)
    print(
        json.dumps(
            {
                "method_folder_count": summary["method_folder_count"],
                "missing_source_count": len(summary["missing_sources"]),
                "missing_sources": summary["missing_sources"][:20],
                "requested_notebook_lesion_volumes_per_scan": summary[
                    "requested_notebook_lesion_volumes_per_scan"
                ],
                "notebook_lesion_volumes_per_scan_by_split": summary[
                    "notebook_lesion_volumes_per_scan_by_split"
                ],
                "max_body_faces_per_plotly_body_trace": summary["max_body_faces_per_plotly_body_trace"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

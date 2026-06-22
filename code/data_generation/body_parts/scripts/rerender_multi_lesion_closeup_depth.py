#!/usr/bin/env python3
"""Rerender existing multi-lesion synthetic depth pairs with lesion-centered cameras."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh

from build_body_part_multi_lesion_depth_dataset import (
    BODY_PARTS,
    ROOT,
    LesionRecord,
    ScanSurface,
    camera_for_lesion_closeup,
    depth_visual,
    render_pair,
    root_relative,
    save_depth_png,
)


DEFAULT_BODY_PART_ROOT = ROOT / "data" / "synthetic" / "multiple_lesion" / "body_parts"
DEFAULT_METHODS = [
    "gaussian",
    "gaussian_diffusion",
    "gaussian_interpolation",
    "spheres",
    "spheres_diffusion",
    "spheres_interpolation",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def resolve_method_data_path(method_root: Path, relative_or_root_path: str) -> Path:
    path = Path(relative_or_root_path)
    if path.is_absolute():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return method_root / "data" / path


def load_lesion_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values() if geom is not None])
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_colors = np.asarray(mesh.visual.vertex_colors)
    if vertex_colors.size == 0:
        rgb = np.full((len(vertices), 3), 190, dtype=np.uint8)
    else:
        rgb = np.clip(vertex_colors[:, :3], 0, 255).astype(np.uint8)
    return vertices, faces, rgb


def lesion_records_from_metadata(metadata: dict[str, Any]) -> list[LesionRecord]:
    return [LesionRecord(**lesion) for lesion in metadata["lesions"]]


def row_camera_updates(
    row: dict[str, str],
    camera: dict[str, Any],
    settings: dict[str, float],
    target_lesion: LesionRecord,
    valid_depth_pixels: int,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "camera_mode": "lesion_closeup_random",
        "valid_depth_pixels": valid_depth_pixels,
        "fov_deg": settings["fov_deg"],
        "roll_deg": settings["roll_deg"],
        "off_axis_deg": settings["off_axis_deg"],
        "frame_margin": settings["frame_margin"],
        "frame_half_width_m": settings["frame_half_width_m"],
        "frame_half_height_m": settings["frame_half_height_m"],
        "camera_distance_m": settings["camera_distance_m"],
        "target_lesion_index": int(target_lesion.lesion_index),
        "target_lesion_face_index": int(target_lesion.face_index),
        "target_lesion_radius_m": float(target_lesion.radius_m),
        "target_lesion_height_m": float(target_lesion.height_m),
        "target_u_offset_m": settings["target_u_offset_m"],
        "target_v_offset_m": settings["target_v_offset_m"],
        "target_normal_offset_m": settings["target_normal_offset_m"],
        "ambient": settings["ambient"],
        "directional_intensity": settings["directional_intensity"],
        "light_yaw_offset": settings["light_yaw_offset"],
        "light_pitch_offset": settings["light_pitch_offset"],
        "eye_xyz": json.dumps(camera["eye_xyz"]),
        "target_xyz": json.dumps(camera["target_xyz"]),
        "view_direction_xyz": json.dumps(camera["view_direction_xyz"]),
        "camera_to_world": json.dumps(camera["camera_to_world"]),
    }
    updated = dict(row)
    updated.update(updates)
    return updated


def settings_row_updates(row: dict[str, str], manifest_row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    for key in [
        "target_xyz",
        "eye_xyz",
        "camera_to_world",
        "fov_deg",
        "roll_deg",
        "off_axis_deg",
        "camera_distance_m",
        "frame_half_width_m",
        "frame_half_height_m",
        "target_lesion_index",
        "target_lesion_radius_m",
        "target_lesion_height_m",
    ]:
        if key in manifest_row:
            updated[key] = manifest_row[key]
    updated["camera_mode"] = "lesion_closeup_random"
    return updated


def update_method_tables(method_root: Path, updates_by_sample: dict[str, dict[str, Any]]) -> None:
    manifest_path = method_root / "data" / "camera_depth_manifest.csv"
    settings_path = method_root / "data" / "settings.csv"
    if manifest_path.exists():
        rows = read_rows(manifest_path)
        fieldnames = list(rows[0].keys()) if rows else []
        rows = [
            dict(row, **updates_by_sample[row["sample_id"]])
            if row["sample_id"] in updates_by_sample
            else row
            for row in rows
        ]
        write_rows(manifest_path, rows, fieldnames)
    if settings_path.exists():
        rows = read_rows(settings_path)
        fieldnames = list(rows[0].keys()) if rows else []
        rows = [
            settings_row_updates(row, updates_by_sample[row["sample_id"]])
            if row["sample_id"] in updates_by_sample
            else row
            for row in rows
        ]
        write_rows(settings_path, rows, fieldnames)


def update_summary(method_root: Path) -> None:
    summary_path = method_root / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["camera_mode"] = "lesion_closeup_random"
    summary["framing"] = "random close-up camera centered near a sampled visible lesion"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def rerender_body_part(
    body_part_root: Path,
    body_part: str,
    methods: list[str],
    renderer: pyrender.OffscreenRenderer,
    scans: dict[str, ScanSurface],
    seed: int,
    limit: int | None,
) -> int:
    representative_root = body_part_root / "gaussian"
    if not representative_root.exists():
        representative_root = body_part_root / methods[0]
    rows = read_rows(representative_root / "data" / "camera_depth_manifest.csv")
    if limit is not None:
        rows = rows[:limit]

    updates_by_sample: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        metadata_path = resolve_method_data_path(representative_root, row["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scan = scans.setdefault(metadata["scan_id"], ScanSurface(metadata["scan_id"]))
        lesions = lesion_records_from_metadata(metadata)
        rng = np.random.default_rng(seed + int(metadata["seed"]) + index * 7919)
        camera, settings, target_lesion = camera_for_lesion_closeup(lesions, body_part, rng)

        volume_path = resolve_method_data_path(representative_root, row["volume_mesh_path"])
        lesion_vertices, lesion_faces, lesion_rgb = load_lesion_mesh(volume_path)
        rgb, depth = render_pair(renderer, scan, lesion_vertices, lesion_faces, lesion_rgb, camera, settings)

        image_path = resolve_method_data_path(representative_root, row["image_path"])
        depth_npy_path = resolve_method_data_path(representative_root, row["depth_npy_path"])
        depth_png_path = resolve_method_data_path(representative_root, row["depth_png_path"])
        depth_vis_path = resolve_method_data_path(representative_root, row["depth_vis_path"])
        imageio.imwrite(image_path, rgb)
        np.save(depth_npy_path, depth)
        save_depth_png(depth, depth_png_path)
        imageio.imwrite(depth_vis_path, depth_visual(depth))

        metadata["camera_mode"] = "lesion_closeup_random"
        metadata["camera"] = camera
        metadata["target_lesion_index"] = int(target_lesion.lesion_index)
        metadata["camera_policy"] = "random close-up camera centered near a sampled visible lesion"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        valid_depth_pixels = int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))
        updates_by_sample[row["sample_id"]] = row_camera_updates(row, camera, settings, target_lesion, valid_depth_pixels)
        if (index + 1) % 50 == 0:
            print(f"[{body_part}] rerendered {index + 1}/{len(rows)}", flush=True)

    for method in methods:
        method_root = body_part_root / method
        if method_root.exists():
            update_method_tables(method_root, updates_by_sample)
            update_summary(method_root)
    return len(rows)


def update_split_summary(body_part_root: Path, total_rows: int) -> None:
    summary = {
        "dataset": "multiple_lesion_body_parts",
        "body_part_root": root_relative(body_part_root),
        "camera_mode": "lesion_closeup_random",
        "framing": "random close-up camera centered near a sampled visible lesion",
        "rerendered_unique_hardlinked_samples": total_rows,
    }
    (body_part_root.parent / "closeup_rerender_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-part-root", default=root_relative(DEFAULT_BODY_PART_ROOT))
    parser.add_argument("--body-part", action="append", choices=BODY_PARTS, default=None)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    body_part_root = Path(args.body_part_root)
    if not body_part_root.is_absolute():
        body_part_root = ROOT / body_part_root
    body_parts = args.body_part or BODY_PARTS
    methods = args.method or DEFAULT_METHODS
    scans: dict[str, ScanSurface] = {}
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    total = 0
    try:
        for body_part in body_parts:
            total += rerender_body_part(
                body_part_root / body_part,
                body_part,
                methods,
                renderer,
                scans,
                args.seed,
                args.limit,
            )
    finally:
        renderer.delete()
    update_split_summary(body_part_root, total)
    print(json.dumps({"rerendered_unique_hardlinked_samples": total}, indent=2), flush=True)


if __name__ == "__main__":
    main()

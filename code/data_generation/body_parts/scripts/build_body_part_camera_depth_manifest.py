#!/usr/bin/env python3
"""Build a consolidated camera/depth manifest for body-part synthetic samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = ROOT / "data" / "synthetic" / "multiple_lesion" / "body_parts" / "physics_aug_growth" / "body_parts_dataset"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_rows(dataset_root: Path, body_part: str) -> list[dict[str, str]]:
    manifest_path = dataset_root / body_part / "data" / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolved_part_data_path(dataset_root: Path, body_part: str, relative_path: str) -> Path:
    return dataset_root / body_part / "data" / relative_path


def build_camera_row(dataset_root: Path, body_part: str, row: dict[str, str]) -> dict[str, Any]:
    metadata_path = resolved_part_data_path(dataset_root, body_part, row["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    camera = metadata["camera"]
    image_path = resolved_part_data_path(dataset_root, body_part, row["image_path"])
    depth_npy_path = resolved_part_data_path(dataset_root, body_part, row["depth_npy_path"])
    depth_png_path = resolved_part_data_path(dataset_root, body_part, row["depth_png_path"])
    depth_vis_path = resolved_part_data_path(dataset_root, body_part, row["depth_vis_path"])
    volume_mesh_path = resolved_part_data_path(dataset_root, body_part, row["mesh_path"])

    return {
        "sample_id": row["sample_id"],
        "body_part": body_part,
        "scan_id": row["scan_id"],
        "patient_volume_index": int(row["patient_volume_index"]),
        "seed": int(row["seed"]),
        "image_path": root_relative(image_path),
        "depth_npy_path": root_relative(depth_npy_path),
        "depth_png_path": root_relative(depth_png_path),
        "depth_vis_path": root_relative(depth_vis_path),
        "volume_mesh_path": root_relative(volume_mesh_path),
        "metadata_path": root_relative(metadata_path),
        "camera_mode": row["camera_mode"],
        "depth_type": row["depth_type"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "valid_depth_pixels": int(row["valid_depth_pixels"]),
        "radius_m": float(row["radius_m"]),
        "lesion_height_m": float(row["lesion_height_m"]),
        "support_radius_m": float(row["support_radius_m"]),
        "spherical_cap_volume_ml": float(row["spherical_cap_volume_ml"]),
        "fov_deg": float(camera["fov_deg"]),
        "angle_rad": float(camera["angle_rad"]),
        "off_axis_deg": float(camera["off_axis_deg"]),
        "roll_deg": float(camera["roll_deg"]),
        "frame_scale": float(camera["frame_scale"]),
        "frame_half_height_m": float(camera["frame_half_height_m"]),
        "camera_distance_m": float(camera["camera_distance_m"]),
        "ambient": float(camera["ambient"]),
        "directional_intensity": float(camera["directional_intensity"]),
        "light_yaw_offset": float(camera["light_yaw_offset"]),
        "light_pitch_offset": float(camera["light_pitch_offset"]),
        "eye_xyz": json.dumps(camera["eye_xyz"]),
        "target_xyz": json.dumps(camera["target_xyz"]),
        "camera_to_world": json.dumps(camera["camera_to_world"]),
    }


def write_camera_manifests(dataset_root: Path, body_parts: list[str]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for body_part in body_parts:
        for row in load_rows(dataset_root, body_part):
            all_rows.append(build_camera_row(dataset_root, body_part, row))

    output_dir = dataset_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "camera_depth_manifest.csv"
    jsonl_path = output_dir / "camera_depth_manifest.jsonl"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    unique_camera_keys = {
        (
            row["body_part"],
            row["scan_id"],
            row["patient_volume_index"],
            row["fov_deg"],
            row["angle_rad"],
            row["off_axis_deg"],
            row["roll_deg"],
            row["eye_xyz"],
            row["target_xyz"],
        )
        for row in all_rows
    }
    by_part = {body_part: sum(row["body_part"] == body_part for row in all_rows) for body_part in body_parts}
    by_scan = {}
    for row in all_rows:
        by_scan[row["scan_id"]] = by_scan.get(row["scan_id"], 0) + 1
    summary = {
        "dataset": "body_parts",
        "output_root": root_relative(dataset_root),
        "body_parts": body_parts,
        "camera_depth_row_count": len(all_rows),
        "unique_camera_setting_count": len(unique_camera_keys),
        "depth_map_count": sum(1 for row in all_rows if Path(ROOT / row["depth_npy_path"]).exists()),
        "depth_png_count": sum(1 for row in all_rows if Path(ROOT / row["depth_png_path"]).exists()),
        "image_count": sum(1 for row in all_rows if Path(ROOT / row["image_path"]).exists()),
        "by_part": by_part,
        "by_scan": by_scan,
        "camera_depth_manifest_csv": root_relative(csv_path),
        "camera_depth_manifest_jsonl": root_relative(jsonl_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    root_summary = {
        "dataset": "body_parts",
        "output_root": root_relative(dataset_root),
        "body_parts": body_parts,
        "total_rgb_depth_pairs": len(all_rows),
        "unique_camera_setting_count": len(unique_camera_keys),
        "camera_depth_manifest_csv": root_relative(csv_path),
        "camera_depth_manifest_jsonl": root_relative(jsonl_path),
        "parts": {
            body_part: {
                "folder": root_relative(dataset_root / body_part),
                "sample_count": by_part[body_part],
                "manifest": root_relative(dataset_root / body_part / "data" / "manifest.csv"),
                "summary": root_relative(dataset_root / body_part / "summary.json"),
            }
            for body_part in body_parts
        },
    }
    (dataset_root / "summary.json").write_text(json.dumps(root_summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=root_relative(DEFAULT_DATASET_ROOT))
    parser.add_argument("--body-part", action="append", choices=BODY_PARTS, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    summary = write_camera_manifests(dataset_root, args.body_part or BODY_PARTS)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

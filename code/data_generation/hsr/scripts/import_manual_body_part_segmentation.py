#!/usr/bin/env python3
"""Import labels exported by the offline HSR manual segmentation app."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_DATA_DIR = ROOT / "data" / "hsr" / "body_part_segmentation" / "data"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "hsr" / "body_part_segmentation" / "manual"

LABEL_NAMES = ["front", "back", "face", "arms", "hands", "legs", "feet", "clothes"]
LABEL_ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
LABEL_COLORS = {
    "front": "#00A6A6",
    "back": "#7B61FF",
    "face": "#FF5A36",
    "arms": "#2CA02C",
    "hands": "#F2C94C",
    "legs": "#1F77B4",
    "feet": "#D946EF",
    "clothes": "#8A8A8A",
}
TIE_BREAK_PRIORITY = [
    LABEL_ID["clothes"],
    LABEL_ID["hands"],
    LABEL_ID["feet"],
    LABEL_ID["face"],
    LABEL_ID["arms"],
    LABEL_ID["legs"],
    LABEL_ID["front"],
    LABEL_ID["back"],
]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    color = hex_color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


def label_rgb01() -> np.ndarray:
    return np.asarray([hex_to_rgb01(LABEL_COLORS[name]) for name in LABEL_NAMES], dtype=np.float64)


def majority_face_labels(triangles: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    face_labels = np.empty(len(triangles), dtype=np.uint8)
    triangle_labels = vertex_labels[triangles]
    for idx, labels in enumerate(triangle_labels):
        counts = np.bincount(labels, minlength=len(LABEL_NAMES))
        max_count = counts.max()
        for label_id in TIE_BREAK_PRIORITY:
            if counts[label_id] == max_count:
                face_labels[idx] = label_id
                break
    return face_labels


def write_label_mesh(
    out_path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_labels: np.ndarray,
) -> None:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_rgb01()[vertex_labels])
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False, compressed=False, write_vertex_colors=True)


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "hsr_manual_body_part_segmentation_v1":
        raise ValueError("Annotation JSON does not use schema hsr_manual_body_part_segmentation_v1")
    if payload.get("labels") != LABEL_NAMES:
        raise ValueError(f"Annotation labels do not match expected labels: {LABEL_NAMES}")
    if not isinstance(payload.get("scans"), list) or not payload["scans"]:
        raise ValueError("Annotation JSON must contain a non-empty scans array.")


def label_counts(vertex_labels: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(vertex_labels == label_id)) for label_id, name in enumerate(LABEL_NAMES)}


def save_scan(
    saved_scan: dict[str, Any],
    base_data_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    scan_id = str(saved_scan.get("scan_id", ""))
    if not scan_id:
        raise ValueError("Saved scan entry is missing scan_id.")

    base_npz_path = base_data_dir / f"{scan_id}_body_part_segmentation.npz"
    if not base_npz_path.exists():
        raise FileNotFoundError(f"Missing base segmentation for {scan_id}: {base_npz_path}")

    base = np.load(base_npz_path, allow_pickle=False)
    vertices = base["vertices"].astype(np.float32)
    triangles = base["triangles"].astype(np.int32)
    expected_count = len(vertices)
    vertex_labels = np.asarray(saved_scan.get("vertex_labels"), dtype=np.uint8)

    if vertex_labels.shape != (expected_count,):
        raise ValueError(
            f"{scan_id} has {vertex_labels.size} labels in the annotation JSON, "
            f"but the base mesh has {expected_count} vertices."
        )
    if vertex_labels.size and int(vertex_labels.max()) >= len(LABEL_NAMES):
        raise ValueError(f"{scan_id} contains label ids outside 0..{len(LABEL_NAMES) - 1}.")

    face_labels = majority_face_labels(triangles, vertex_labels)
    npz_path = data_dir / f"{scan_id}_body_part_segmentation.npz"
    colored_mesh_path = data_dir / f"{scan_id}_body_part_colored_mesh.ply"

    np.savez_compressed(
        npz_path,
        scan_id=base["scan_id"],
        vertices=vertices,
        triangles=triangles,
        vertex_colors=base["vertex_colors"].astype(np.uint8),
        vertex_normals=base["vertex_normals"].astype(np.float32),
        vertex_labels=vertex_labels.astype(np.uint8),
        face_labels=face_labels.astype(np.uint8),
        label_names=base["label_names"],
        label_colors=base["label_colors"],
        front_coord=base["front_coord"].astype(np.float32),
        front_sign=base["front_sign"],
        height=base["height"],
        params_json=base["params_json"],
    )
    write_label_mesh(colored_mesh_path, vertices, triangles, vertex_labels)

    counts = label_counts(vertex_labels)
    return {
        "scan_id": scan_id,
        "base_npz": root_relative(base_npz_path),
        "npz": root_relative(npz_path),
        "colored_mesh": root_relative(colored_mesh_path),
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "label_counts": counts,
    }


def import_annotations(
    annotations_path: Path,
    base_data_dir: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    validate_payload(payload)

    scans = [save_scan(scan, base_data_dir, data_dir) for scan in payload["scans"]]
    copied_annotations_path = data_dir / "manual_body_part_annotations.json"
    if annotations_path.resolve() != copied_annotations_path.resolve():
        shutil.copy2(annotations_path, copied_annotations_path)

    summary = {
        "dataset": output_root.name,
        "method": "manual labels exported by offline HSR body-part segmentation app",
        "annotations": root_relative(copied_annotations_path),
        "source_annotations": root_relative(annotations_path),
        "copied_annotations": root_relative(copied_annotations_path),
        "package_id": payload.get("package_id"),
        "exported_at": payload.get("exported_at"),
        "labels": LABEL_NAMES,
        "label_colors": LABEL_COLORS,
        "scans": scans,
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="manual_body_part_annotations.json exported by the app")
    parser.add_argument("--base-data-dir", type=Path, default=DEFAULT_BASE_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = import_annotations(
        annotations_path=args.annotations,
        base_data_dir=args.base_data_dir,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

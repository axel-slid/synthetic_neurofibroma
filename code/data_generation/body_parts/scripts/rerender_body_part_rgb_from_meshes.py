#!/usr/bin/env python3
"""Refresh body-part synthetic RGB renders from corrected lesion volume meshes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import pyrender
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_body_part_multi_lesion_depth_dataset import ScanSurface, render_pair  # noqa: E402


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    faces = np.vstack(ply["face"].data["vertex_indices"]).astype(np.int32)
    return xyz, faces, rgb


def iter_manifest_paths(roots: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for root in roots:
        if root.is_file() and root.name in {"manifest.csv", "camera_depth_manifest.csv"}:
            manifests.append(root)
        elif root.is_dir():
            manifests.extend(root.rglob("camera_depth_manifest.csv"))
            manifests.extend(path for path in root.rglob("manifest.csv") if path.parent.name == "data")
    return sorted(set(manifests))


def atomic_copy_to_path(source: Path, destination: Path) -> None:
    # Copy over the existing inode so hardlinked duplicate paths update together.
    with destination.open("wb") as handle:
        with source.open("rb") as src:
            shutil.copyfileobj(src, handle)


def refresh_rgb(
    renderer: pyrender.OffscreenRenderer,
    scan_cache: dict[str, ScanSurface],
    data_root: Path,
    row: dict[str, str],
    seen_image_inodes: set[tuple[int, int]],
    dry_run: bool,
) -> bool:
    image_value = row.get("image_path", "")
    mesh_value = row.get("volume_mesh_path") or row.get("mesh_path", "")
    metadata_value = row.get("metadata_path", "")
    if not image_value or not mesh_value or not metadata_value:
        return False

    image_path = resolve_path(data_root, image_value)
    mesh_path = resolve_path(data_root, mesh_value)
    metadata_path = resolve_path(data_root, metadata_value)
    if not image_path.exists() or not mesh_path.exists() or not metadata_path.exists():
        return False

    stat = image_path.stat()
    image_inode = (stat.st_dev, stat.st_ino)
    if image_inode in seen_image_inodes:
        return False
    seen_image_inodes.add(image_inode)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scan_id = str(metadata["scan_id"])
    if scan_id not in scan_cache:
        scan_cache[scan_id] = ScanSurface(scan_id)

    if dry_run:
        return True

    lesion_vertices, lesion_faces, lesion_rgb = read_ply_mesh(mesh_path)
    camera = metadata["camera"]
    rgb, _depth = render_pair(
        renderer,
        scan_cache[scan_id],
        lesion_vertices,
        lesion_faces,
        lesion_rgb,
        camera,
        camera,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        imageio.imwrite(tmp_path, rgb)
        atomic_copy_to_path(tmp_path, image_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="*", default=[root_relative(ROOT / "data" / "synthetic")])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(value) if Path(value).is_absolute() else ROOT / value for value in args.root]
    manifests = iter_manifest_paths(roots)
    renderer = None if args.dry_run else pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    scan_cache: dict[str, ScanSurface] = {}
    seen_image_inodes: set[tuple[int, int]] = set()
    refreshed = 0
    visited_rows = 0
    try:
        for manifest_path in manifests:
            data_root = manifest_path.parent
            for row in read_rows(manifest_path):
                visited_rows += 1
                did_refresh = refresh_rgb(
                    renderer,
                    scan_cache,
                    data_root,
                    row,
                    seen_image_inodes,
                    args.dry_run,
                )
                if did_refresh:
                    refreshed += 1
                    if refreshed % 250 == 0:
                        print(f"refreshed={refreshed} visited_rows={visited_rows}", flush=True)
                    if args.limit is not None and refreshed >= args.limit:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        if renderer is not None:
            renderer.delete()

    action = "would_refresh" if args.dry_run else "refreshed"
    print(f"{action}={refreshed} visited_rows={visited_rows} manifests={len(manifests)}", flush=True)


if __name__ == "__main__":
    main()

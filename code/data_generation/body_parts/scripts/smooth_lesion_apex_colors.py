#!/usr/bin/env python3
"""Smooth one-vertex apex color artifacts in generated lesion PLY files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

# Cap topologies used by the synthetic lesion generators.
# Values are (radial_segments, angular_segments).
CAP_TOPOLOGIES = (
    (12, 40),
    (7, 24),
    (28, 96),
    (30, 112),
)
COLOR_FIELDS = ("red", "green", "blue")
VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("alpha", "u1"),
    ]
)


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def infer_cap_topology(vertex_count: int, face_count: int) -> tuple[int, int] | None:
    for radial_segments, angular_segments in CAP_TOPOLOGIES:
        vertices_per_cap = 1 + radial_segments * angular_segments
        faces_per_cap = angular_segments + (radial_segments - 1) * 2 * angular_segments
        if (
            vertex_count % vertices_per_cap == 0
            and face_count % faces_per_cap == 0
            and vertex_count // vertices_per_cap == face_count // faces_per_cap
        ):
            return vertices_per_cap, angular_segments
    return None


def read_ply_header(path: Path) -> tuple[int, int, int, list[str], str]:
    vertex_count: int | None = None
    face_count: int | None = None
    vertex_properties: list[str] = []
    active_element: str | None = None
    format_name: str | None = None
    header_size = 0

    with path.open("rb") as handle:
        first = handle.readline()
        header_size += len(first)
        if first != b"ply\n":
            raise ValueError("missing ply header")
        for raw_line in handle:
            header_size += len(raw_line)
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                format_name = parts[1]
            elif parts[:2] == ["element", "vertex"] and len(parts) == 3:
                active_element = "vertex"
                vertex_count = int(parts[2])
            elif parts[:2] == ["element", "face"] and len(parts) == 3:
                active_element = "face"
                face_count = int(parts[2])
            elif parts[0] == "property" and active_element == "vertex" and len(parts) >= 3:
                vertex_properties.append(parts[-1])
        else:
            raise ValueError("missing end_header")

    if vertex_count is None or face_count is None or format_name is None:
        raise ValueError("incomplete ply header")
    return header_size, vertex_count, face_count, vertex_properties, format_name


def smooth_ply(path: Path, dry_run: bool) -> tuple[bool, str]:
    try:
        header_size, vertex_count, face_count, vertex_properties, format_name = read_ply_header(path)
    except ValueError as exc:
        return False, str(exc)

    expected_properties = ["x", "y", "z", "red", "green", "blue", "alpha"]
    if format_name != "binary_little_endian":
        return False, f"unsupported format {format_name}"
    if vertex_properties != expected_properties:
        return False, f"unsupported vertex properties {vertex_properties}"

    topology = infer_cap_topology(vertex_count, face_count)
    if topology is None:
        return False, f"unsupported topology vertices={vertex_count} faces={face_count}"

    vertices_per_cap, angular_segments = topology
    mode = "r" if dry_run else "r+"
    vertex = np.memmap(path, dtype=VERTEX_DTYPE, mode=mode, offset=header_size, shape=(vertex_count,))
    changed = False
    for offset in range(0, vertex_count, vertices_per_cap):
        ring = slice(offset + 1, offset + 1 + angular_segments)
        for field in COLOR_FIELDS:
            replacement = int(np.rint(vertex[field][ring].astype(np.float32).mean()))
            if int(vertex[field][offset]) != replacement:
                if not dry_run:
                    vertex[field][offset] = replacement
                changed = True

    if changed and not dry_run:
        vertex.flush()
    return changed, f"caps={vertex_count // vertices_per_cap}"


def iter_ply_files(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".ply":
            paths.append(root)
        elif root.is_dir():
            paths.extend(root.rglob("*.ply"))
    return sorted(set(paths))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="*",
        default=[root_relative(ROOT / "data" / "synthetic")],
        help="PLY file or directory to repair. Defaults to data/synthetic.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [resolve_path(path_value) for path_value in args.root]
    paths = iter_ply_files(roots)
    if args.limit is not None:
        paths = paths[: args.limit]

    changed = 0
    skipped = 0
    for index, path in enumerate(paths, start=1):
        did_change, detail = smooth_ply(path, args.dry_run)
        if did_change:
            changed += 1
            if args.verbose:
                print(f"smoothed {root_relative(path)} {detail}", flush=True)
        else:
            skipped += 1
            if args.verbose:
                print(f"skipped {root_relative(path)} {detail}", flush=True)
        if index % 1000 == 0:
            print(f"processed={index} changed={changed} skipped={skipped}", flush=True)

    action = "would_smooth" if args.dry_run else "smoothed"
    print(f"{action}={changed} skipped={skipped} total={len(paths)}", flush=True)


if __name__ == "__main__":
    main()

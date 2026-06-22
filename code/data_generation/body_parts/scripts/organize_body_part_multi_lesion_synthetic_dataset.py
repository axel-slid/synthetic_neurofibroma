#!/usr/bin/env python3
"""Create a clean top-level synthetic multi-lesion body-part dataset tree."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_ROOT = (
    ROOT
    / "data"
    / "synthetic"
    / "multiple_lesion"
    / "body_parts"
    / "physics_aug_growth"
    / "body_parts_multi_lesion"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "synthetic" / "body_parts_multi_lesion"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]
PATH_COLUMNS = [
    "image_path",
    "depth_npy_path",
    "depth_png_path",
    "depth_vis_path",
    "volume_mesh_path",
    "metadata_path",
]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_relative(dataset_root: Path, path: Path) -> str:
    return str(path.relative_to(dataset_root))


def resolve_root(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def destination_for_source(source_root: Path, output_root: Path, source_file: Path) -> Path:
    return output_root / source_file.relative_to(source_root)


def rewrite_row_paths(
    row: dict[str, str],
    source_root: Path,
    output_root: Path,
    root_relative_paths: bool,
) -> dict[str, Any]:
    rewritten: dict[str, Any] = dict(row)
    for column in PATH_COLUMNS:
        source_file = resolve_root(row[column])
        destination = destination_for_source(source_root, output_root, source_file)
        rewritten[column] = root_relative(destination) if root_relative_paths else dataset_relative(output_root, destination)
    return rewritten


def build_summary(output_root: Path, root_relative_manifest: Path, dataset_manifest: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_part = {body_part: sum(row["body_part"] == body_part for row in rows) for body_part in BODY_PARTS}
    by_scan: dict[str, int] = {}
    for row in rows:
        by_scan[row["scan_id"]] = by_scan.get(row["scan_id"], 0) + 1
    lesion_counts = [int(row["lesion_count"]) for row in rows]
    return {
        "dataset": "body_parts_multi_lesion",
        "output_root": root_relative(output_root),
        "body_parts": BODY_PARTS,
        "camera_depth_row_count": len(rows),
        "sample_count": len(rows),
        "settings_per_body_part": by_part,
        "unique_settings_per_body_part": {
            body_part: len({row["seed"] for row in rows if row["body_part"] == body_part})
            for body_part in BODY_PARTS
        },
        "by_part": by_part,
        "by_scan": by_scan,
        "lesion_count_min": min(lesion_counts),
        "lesion_count_max": max(lesion_counts),
        "lesion_count_mean": float(np.mean(lesion_counts)),
        "unique_seed_count": len({row["seed"] for row in rows}),
        "camera_mode": "lesion_closeup_random",
        "framing": "random close-up camera centered near a sampled visible lesion",
        "lesion_pattern_source": "10-100 random spherical-cap NF-like lesions with interpolated skin-color texture per image",
        "camera_depth_manifest_csv": root_relative(root_relative_manifest),
        "manifest_csv": root_relative(dataset_manifest),
    }


def write_manifests(source_root: Path, output_root: Path, source_rows: list[dict[str, str]]) -> dict[str, Any]:
    root_rows = [rewrite_row_paths(row, source_root, output_root, root_relative_paths=True) for row in source_rows]
    dataset_rows = [rewrite_row_paths(row, source_root, output_root, root_relative_paths=False) for row in source_rows]

    root_manifest = output_root / "data" / "camera_depth_manifest.csv"
    root_jsonl = output_root / "data" / "camera_depth_manifest.jsonl"
    dataset_manifest = output_root / "data" / "manifest.csv"
    dataset_jsonl = output_root / "data" / "manifest.jsonl"
    write_rows(root_manifest, root_rows)
    write_jsonl(root_jsonl, root_rows)
    write_rows(dataset_manifest, dataset_rows)
    write_jsonl(dataset_jsonl, dataset_rows)

    for body_part in BODY_PARTS:
        part_dataset_rows = [row for row in dataset_rows if row["body_part"] == body_part]
        part_root_rows = [row for row in root_rows if row["body_part"] == body_part]
        part_dir = output_root / "data" / body_part
        write_rows(part_dir / "manifest.csv", part_dataset_rows)
        write_jsonl(part_dir / "manifest.jsonl", part_dataset_rows)
        write_rows(part_dir / "camera_depth_manifest.csv", part_root_rows)
        write_jsonl(part_dir / "camera_depth_manifest.jsonl", part_root_rows)
        lesion_counts = [int(row["lesion_count"]) for row in part_dataset_rows]
        part_summary = {
            "body_part": body_part,
            "sample_count": len(part_dataset_rows),
            "setting_count": len(part_dataset_rows),
            "unique_seed_count": len({row["seed"] for row in part_dataset_rows}),
            "lesion_count_min": min(lesion_counts),
            "lesion_count_max": max(lesion_counts),
            "camera_mode": "lesion_closeup_random",
            "manifest_csv": root_relative(part_dir / "manifest.csv"),
            "camera_depth_manifest_csv": root_relative(part_dir / "camera_depth_manifest.csv"),
            "folders": [
                root_relative(part_dir / "images"),
                root_relative(part_dir / "depth"),
                root_relative(part_dir / "depth_vis"),
                root_relative(part_dir / "metadata"),
                root_relative(part_dir / "volumes"),
            ],
        }
        (part_dir / "summary.json").write_text(json.dumps(part_summary, indent=2) + "\n", encoding="utf-8")

    summary = build_summary(output_root, root_manifest, dataset_manifest, root_rows)
    (output_root / "data" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def make_montage(output_root: Path, rows: list[dict[str, Any]], count: int) -> Path:
    output_path = output_root / "visualizations" / f"montage_{count}_rgb_depth.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    per_part = max(1, count // len(BODY_PARTS))
    for body_part in BODY_PARTS:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        indices = np.linspace(0, len(part_rows) - 1, min(per_part, len(part_rows)), dtype=int)
        selected.extend(part_rows[int(index)] for index in indices)
    if len(selected) < count:
        chosen_ids = {row["sample_id"] for row in selected}
        selected.extend(row for row in rows if row["sample_id"] not in chosen_ids)
    selected = selected[:count]

    tile_h = 96
    columns = 10
    tiles = []
    for row in selected:
        rgb = Image.open(output_root / row["image_path"]).convert("RGB")
        depth = Image.open(output_root / row["depth_vis_path"]).convert("L").convert("RGB")
        if depth.size != rgb.size:
            depth = depth.resize(rgb.size, Image.Resampling.LANCZOS)
        figure = Image.new("RGB", (rgb.width * 2, rgb.height), "white")
        figure.paste(rgb, (0, 0))
        figure.paste(depth, (rgb.width, 0))
        tiles.append(figure.resize((tile_h * 2, tile_h), Image.Resampling.LANCZOS))

    rows_count = int(np.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * tile_h * 2, rows_count * tile_h), "white")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    montage.save(output_path)
    return output_path


def organize(source_root: Path, output_root: Path, overwrite: bool, montage_count: int) -> dict[str, Any]:
    source_manifest = source_root / "data" / "camera_depth_manifest.csv"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest}")
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "visualizations").mkdir(parents=True, exist_ok=True)

    source_rows = read_rows(source_manifest)
    for row in source_rows:
        for column in PATH_COLUMNS:
            source_file = resolve_root(row[column])
            link_or_copy(source_file, destination_for_source(source_root, output_root, source_file))

    summary = write_manifests(source_root, output_root, source_rows)
    dataset_rows = read_rows(output_root / "data" / "manifest.csv")
    montage_path = make_montage(output_root, dataset_rows, montage_count)
    summary["visualizations"] = {
        "montage": root_relative(montage_path),
    }
    (output_root / "data" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=root_relative(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=root_relative(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--montage-count", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    summary = organize(source_root, output_root, args.overwrite, args.montage_count)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

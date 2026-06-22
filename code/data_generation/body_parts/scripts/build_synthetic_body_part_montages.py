#!/usr/bin/env python3
"""Build 100-sample RGB/GT-depth montages for synthetic body-part method folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SYNTHETIC_ROOT = ROOT / "data" / "synthetic"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evenly_spaced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    if count <= 1:
        return rows[:1]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    selected: list[dict[str, str]] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            selected.append(rows[index])
            seen.add(index)
    cursor = 0
    while len(selected) < count and cursor < len(rows):
        if cursor not in seen:
            selected.append(rows[cursor])
            seen.add(cursor)
        cursor += 1
    return selected[:count]


def resolve_data_path(method_root: Path, relative_path: str) -> Path:
    data_path = method_root / "data" / relative_path
    if data_path.exists():
        return data_path
    method_path = method_root / relative_path
    if method_path.exists():
        return method_path
    raise FileNotFoundError(f"Could not resolve {relative_path!r} under {method_root}")


def square_thumb(path: Path, tile_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    tile.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
    return tile


def make_pair_tile(method_root: Path, row: dict[str, str], tile_size: int) -> Image.Image:
    rgb_path = resolve_data_path(method_root, row["image_path"])
    depth_key = "depth_vis_path" if row.get("depth_vis_path") else "depth_png_path"
    depth_path = resolve_data_path(method_root, row[depth_key])

    rgb = square_thumb(rgb_path, tile_size)
    depth = square_thumb(depth_path, tile_size)
    tile = Image.new("RGB", (tile_size * 2, tile_size), "white")
    tile.paste(rgb, (0, 0))
    tile.paste(depth, (tile_size, 0))
    return tile


def build_montage(
    method_root: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    count: int,
    tile_size: int,
    columns: int,
) -> list[dict[str, str]]:
    selected = evenly_spaced_rows(rows, count)
    tiles = [make_pair_tile(method_root, row, tile_size) for row in selected]
    row_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * tile_size * 2, row_count * tile_size), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile.width
        y = (index // columns) * tile.height
        montage.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, optimize=True)
    return selected


def write_montage_manifest(
    method_root: Path,
    manifest_path: Path,
    rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    montage_path: Path,
    count: int,
    tile_size: int,
    columns: int,
) -> Path:
    split = method_root.parents[2].name
    body_part = method_root.parent.name
    method = method_root.name
    payload: dict[str, Any] = {
        "split": split,
        "body_part": body_part,
        "method": method,
        "source_manifest": repo_relative(manifest_path),
        "source_row_count": len(rows),
        "requested_count": count,
        "selected_count": len(selected_rows),
        "tile_size_px": tile_size,
        "columns": columns,
        "montage": repo_relative(montage_path),
        "sample_ids": [row.get("sample_id", "") for row in selected_rows],
    }
    out_path = montage_path.parent / "montage_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def discover_manifests(synthetic_root: Path, splits: list[str]) -> list[Path]:
    manifests: list[Path] = []
    for split in splits:
        split_root = synthetic_root / split / "body_parts"
        manifests.extend(sorted(split_root.glob("*/*/data/camera_depth_manifest.csv")))
    return sorted(manifests)


def build_all(
    synthetic_root: Path,
    splits: list[str],
    count: int,
    tile_size: int,
    columns: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    manifests = discover_manifests(synthetic_root, splits)
    if not manifests:
        raise FileNotFoundError(f"No camera_depth_manifest.csv files found under {synthetic_root}")

    for manifest_path in manifests:
        method_root = manifest_path.parents[1]
        rows = read_manifest(manifest_path)
        if not rows:
            print(f"SKIP empty manifest: {repo_relative(manifest_path)}")
            continue
        method = method_root.name
        montage_dir = method_root / "visualization" / "montage"
        montage_path = montage_dir / f"{method}_{count}_rgb_gt_depth_montage.png"
        selected_rows = build_montage(method_root, rows, montage_path, count, tile_size, columns)
        manifest_out = write_montage_manifest(
            method_root=method_root,
            manifest_path=manifest_path,
            rows=rows,
            selected_rows=selected_rows,
            montage_path=montage_path,
            count=count,
            tile_size=tile_size,
            columns=columns,
        )
        summary = {
            "method_root": repo_relative(method_root),
            "montage": repo_relative(montage_path),
            "manifest": repo_relative(manifest_out),
            "selected_count": len(selected_rows),
        }
        summaries.append(summary)
        print(f"WROTE {summary['montage']}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["single_lesion", "multiple_lesion"],
        choices=["single_lesion", "multiple_lesion"],
        help="Synthetic split folders to process.",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=96)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SYNTHETIC_ROOT / "data" / "body_part_montage_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = build_all(
        synthetic_root=args.synthetic_root,
        splits=args.splits,
        count=args.count,
        tile_size=args.tile_size,
        columns=args.columns,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"montages": summaries}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {repo_relative(args.summary)}")
    print(f"Total montages: {len(summaries)}")


if __name__ == "__main__":
    main()

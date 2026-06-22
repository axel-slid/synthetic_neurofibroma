#!/usr/bin/env python3
"""Refresh saved Plotly lesion vertex colors from corrected body-part PLY files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = ROOT / "data" / "synthetic"
NEWPLOT = "Plotly.newPlot("

JSON_DECODER = json.JSONDecoder()
PLY_COLOR_CACHE: dict[tuple[int, int], list[str]] = {}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def json_value_end(text: str, start: int) -> int:
    """Return the exclusive end offset for a JSON value starting at start."""
    start = skip_ws(text, start)
    if start >= len(text):
        raise ValueError("Expected JSON value, found end of text")

    opener = text[start]
    pairs = {"[": "]", "{": "}"}
    if opener not in pairs:
        _, end = JSON_DECODER.raw_decode(text, start)
        return end

    stack = [pairs[opener]]
    in_string = False
    escaped = False
    index = start + 1
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index + 1
        index += 1

    raise ValueError("Unterminated JSON value")


def locate_newplot_data(text: str, start: int = 0) -> tuple[int, int, int] | None:
    """Find a real Plotly.newPlot call and return call offset plus data bounds."""
    search_from = start
    while True:
        call_start = text.find(NEWPLOT, search_from)
        if call_start < 0:
            return None
        index = skip_ws(text, call_start + len(NEWPLOT))
        try:
            _, index = JSON_DECODER.raw_decode(text, index)
            index = skip_ws(text, index)
            if index >= len(text) or text[index] != ",":
                raise ValueError("Expected comma after Plotly div id")
            data_start = skip_ws(text, index + 1)
            if data_start >= len(text) or text[data_start] != "[":
                raise ValueError("Expected Plotly data array")
            data_end = json_value_end(text, data_start)
            return call_start, data_start, data_end
        except (json.JSONDecodeError, ValueError):
            search_from = call_start + len(NEWPLOT)


def replace_newplot_data(text: str, data_json: str, parse_first: bool) -> tuple[str, list[dict[str, Any]] | None, int]:
    """Replace Plotly data arrays in text. Optionally parse the first one."""
    parts: list[str] = []
    parsed_data: list[dict[str, Any]] | None = None
    replacements = 0
    cursor = 0
    while True:
        located = locate_newplot_data(text, cursor)
        if located is None:
            parts.append(text[cursor:])
            break
        _, data_start, data_end = located
        if parse_first and parsed_data is None:
            parsed_data = json.loads(text[data_start:data_end])
            data_json = json.dumps(parsed_data, separators=(",", ":"))
        parts.append(text[cursor:data_start])
        parts.append(data_json)
        cursor = data_end
        replacements += 1
    return "".join(parts), parsed_data, replacements


def notebook_and_manifest_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for manifest_path in sorted(root.glob("*/body_parts/*/*/visualization/plotly/*_closed_body_lesion_manifest.json")):
        method = manifest_path.name.removesuffix("_closed_body_lesion_manifest.json")
        notebook_path = manifest_path.with_name(f"{method}_closed_body_lesion_viewer.ipynb")
        if notebook_path.exists():
            pairs.append((notebook_path, manifest_path))
    return pairs


def volume_path_for_record(notebook_path: Path, record: dict[str, Any]) -> Path:
    method_root = notebook_path.parents[2]
    data_root = method_root / "data"
    raw_value = str(record.get("volume_mesh_path") or record.get("mesh_path") or "")
    if not raw_value:
        raise ValueError(f"Missing volume mesh path in {notebook_path}")
    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        return raw_path
    return data_root / raw_path


def ply_vertexcolors(path: Path) -> list[str]:
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino)
    cached = PLY_COLOR_CACHE.get(key)
    if cached is not None:
        return cached

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    props = {prop.name for prop in vertex.properties}
    if {"red", "green", "blue"}.issubset(props):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    else:
        rgb = np.full((len(vertex), 3), 190, dtype=np.uint8)
    colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]
    PLY_COLOR_CACHE[key] = colors
    return colors


def expected_lesion_colors(notebook_path: Path, manifest_path: Path) -> list[list[str]]:
    manifest = load_json(manifest_path)
    records_by_scan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest.get("records", []):
        records_by_scan[str(record.get("scan_id", ""))].append(record)

    colors_by_scan: list[list[str]] = []
    for scan_id in sorted(records_by_scan):
        colors: list[str] = []
        for record in records_by_scan[scan_id]:
            colors.extend(ply_vertexcolors(volume_path_for_record(notebook_path, record)))
        colors_by_scan.append(colors)
    return colors_by_scan


def refresh_plotly_data(
    data: list[dict[str, Any]],
    lesion_colors_by_scan: list[list[str]],
    notebook_path: Path,
) -> tuple[int, int]:
    changed_traces = 0
    checked_traces = 0

    trace_indices = lesion_trace_indices(data, lesion_colors_by_scan, notebook_path)
    for trace_index, colors in zip(trace_indices, lesion_colors_by_scan, strict=True):
        trace = data[trace_index]
        checked_traces += 1
        current = trace.get("vertexcolor")
        if current != colors:
            trace["vertexcolor"] = colors
            changed_traces += 1
    return changed_traces, checked_traces


def valid_lesion_trace(trace: dict[str, Any], expected_color_count: int) -> bool:
    return (
        trace.get("type") == "mesh3d"
        and isinstance(trace.get("vertexcolor"), list)
        and len(trace["vertexcolor"]) == expected_color_count
    )


def lesion_trace_indices(
    data: list[dict[str, Any]],
    lesion_colors_by_scan: list[list[str]],
    notebook_path: Path,
) -> list[int]:
    scan_count = len(lesion_colors_by_scan)
    expected_lengths = [len(colors) for colors in lesion_colors_by_scan]

    # Current notebooks have body + lesion traces per scan. Older repair runs
    # included an extra body-part overlay before the lesion trace.
    for traces_per_scan, lesion_offset in ((2, 1), (3, 2)):
        if len(data) != scan_count * traces_per_scan:
            continue
        indices = [scan_index * traces_per_scan + lesion_offset for scan_index in range(scan_count)]
        if all(valid_lesion_trace(data[index], expected_lengths[scan_index]) for scan_index, index in enumerate(indices)):
            return indices

    named_candidates = [
        index
        for index, trace in enumerate(data)
        if trace.get("type") == "mesh3d" and "lesion" in str(trace.get("name", "")).lower()
    ]
    if len(named_candidates) == scan_count and all(
        valid_lesion_trace(data[index], expected_lengths[scan_index])
        for scan_index, index in enumerate(named_candidates)
    ):
        return named_candidates

    expected_trace_counts = " or ".join(str(scan_count * count) for count in (2, 3))
    raise ValueError(
        f"{notebook_path}: expected {expected_trace_counts} Plotly traces with recognizable lesion traces, "
        f"found {len(data)}"
    )


def html_value_as_text(value: Any) -> tuple[str, bool]:
    if isinstance(value, list):
        return "".join(value), True
    if isinstance(value, str):
        return value, False
    raise TypeError(f"Unsupported text/html payload type: {type(value).__name__}")


def restore_html_value(text: str, was_list: bool) -> str | list[str]:
    if was_list:
        return [text]
    return text


def refresh_notebook(notebook_path: Path, manifest_path: Path, dry_run: bool) -> dict[str, int]:
    notebook = load_json(notebook_path)
    lesion_colors_by_scan = expected_lesion_colors(notebook_path, manifest_path)

    updated_data_json: str | None = None
    changed_traces = 0
    checked_traces = 0
    html_payloads = 0
    html_replacements = 0
    notebook_changed = False

    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            data_payload = output.get("data")
            if not isinstance(data_payload, dict) or "text/html" not in data_payload:
                continue
            html_text, was_list = html_value_as_text(data_payload["text/html"])
            if NEWPLOT not in html_text:
                continue

            if updated_data_json is None:
                replaced_html, parsed_data, replacements = replace_newplot_data(
                    html_text,
                    data_json="[]",
                    parse_first=True,
                )
                if parsed_data is None:
                    continue
                trace_changes, trace_checks = refresh_plotly_data(parsed_data, lesion_colors_by_scan, notebook_path)
                changed_traces += trace_changes
                checked_traces += trace_checks
                updated_data_json = json.dumps(parsed_data, separators=(",", ":"))
                if trace_changes:
                    replaced_html, _, replacements = replace_newplot_data(
                        html_text,
                        data_json=updated_data_json,
                        parse_first=False,
                    )
                else:
                    replaced_html = html_text
            else:
                replaced_html, _, replacements = replace_newplot_data(
                    html_text,
                    data_json=updated_data_json,
                    parse_first=False,
                )

            html_payloads += 1
            html_replacements += replacements
            if replaced_html != html_text:
                data_payload["text/html"] = restore_html_value(replaced_html, was_list)
                notebook_changed = True

    if updated_data_json is None:
        raise ValueError(f"{notebook_path}: no Plotly.newPlot data found")
    if checked_traces != len(lesion_colors_by_scan):
        raise ValueError(
            f"{notebook_path}: checked {checked_traces} lesion traces, "
            f"expected {len(lesion_colors_by_scan)}"
        )
    if notebook_changed and not dry_run:
        write_json(notebook_path, notebook)

    return {
        "changed_traces": changed_traces,
        "checked_traces": checked_traces,
        "html_payloads": html_payloads,
        "html_replacements": html_replacements,
        "notebook_changed": int(notebook_changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing notebooks.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N notebooks.")
    parser.add_argument("--notebook", type=Path, default=None, help="Refresh a single notebook path.")
    args = parser.parse_args()

    if args.notebook is not None:
        notebook_path = args.notebook
        method = notebook_path.name.removesuffix("_closed_body_lesion_viewer.ipynb")
        manifest_path = notebook_path.with_name(f"{method}_closed_body_lesion_manifest.json")
        pairs = [(notebook_path, manifest_path)]
    else:
        pairs = notebook_and_manifest_pairs(args.root)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    totals = {
        "notebooks": 0,
        "notebooks_changed": 0,
        "changed_traces": 0,
        "checked_traces": 0,
        "html_payloads": 0,
        "html_replacements": 0,
    }
    for index, (notebook_path, manifest_path) in enumerate(pairs, start=1):
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        result = refresh_notebook(notebook_path, manifest_path, dry_run=args.dry_run)
        totals["notebooks"] += 1
        totals["notebooks_changed"] += result["notebook_changed"]
        totals["changed_traces"] += result["changed_traces"]
        totals["checked_traces"] += result["checked_traces"]
        totals["html_payloads"] += result["html_payloads"]
        totals["html_replacements"] += result["html_replacements"]
        rel_path = notebook_path.relative_to(ROOT) if notebook_path.is_relative_to(ROOT) else notebook_path
        print(
            f"[{index}/{len(pairs)}] {rel_path} "
            f"changed_traces={result['changed_traces']} html_replacements={result['html_replacements']}"
        )

    mode = "dry_run" if args.dry_run else "written"
    print(
        f"{mode}: notebooks={totals['notebooks']} notebooks_changed={totals['notebooks_changed']} "
        f"changed_traces={totals['changed_traces']} checked_traces={totals['checked_traces']} "
        f"html_payloads={totals['html_payloads']} html_replacements={totals['html_replacements']}"
    )


if __name__ == "__main__":
    main()

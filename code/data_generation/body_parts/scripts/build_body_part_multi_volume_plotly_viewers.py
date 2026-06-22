#!/usr/bin/env python3
"""Build full-body Plotly viewers with many synthetic volumes per body part."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = ROOT / "data" / "synthetic" / "multiple_lesion" / "body_parts" / "physics_aug_growth" / "body_parts_dataset"
DEFAULT_VISUALIZATION_ROOT = (
    ROOT / "data" / "synthetic" / "multiple_lesion" / "visualization" / "physics_aug_growth" / "body_parts_dataset"
)
HSR_MESH_ROOT = ROOT / "data" / "hsr" / "visualizations" / "meshes"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]
SCAN_IDS = ["HSR0018-Body-070", "HSR0152-Body-090"]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rgb_strings(rgb: np.ndarray) -> list[str]:
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]


def read_colored_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if mesh.has_vertex_colors():
        rgb = np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255.0), 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 185, dtype=np.uint8)
    return xyz, faces, rgb


def combine_lesion_meshes(part_root: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_xyz: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    offset = 0
    data_root = part_root / "data"
    for row in rows:
        xyz, faces, rgb = read_colored_mesh(data_root / row["mesh_path"])
        all_xyz.append(xyz)
        all_faces.append(faces + offset)
        all_rgb.append(rgb)
        offset += len(xyz)
    return np.vstack(all_xyz), np.vstack(all_faces), np.vstack(all_rgb)


def load_part_rows(part_root: Path, scan_id: str, max_volumes_per_scan: int) -> list[dict[str, str]]:
    with (part_root / "data" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scan_id"] == scan_id]
    rows.sort(key=lambda row: int(row["patient_volume_index"]))
    return rows[:max_volumes_per_scan]


def normalize_to_body(
    body_xyz: np.ndarray,
    lesion_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center = body_xyz.mean(axis=0)
    scale = float(np.max(np.ptp(body_xyz, axis=0)))
    if scale <= 0:
        scale = 1.0
    return (body_xyz - center) / scale, (lesion_xyz - center) / scale, center, scale


def build_scan_traces(part_root: Path, body_part: str, scan_id: str, max_volumes_per_scan: int) -> tuple[list[Any], dict[str, Any]]:
    rows = load_part_rows(part_root, scan_id, max_volumes_per_scan)
    if not rows:
        raise FileNotFoundError(f"No manifest rows found for {body_part} {scan_id}")

    body_xyz, body_faces, body_rgb = read_colored_mesh(HSR_MESH_ROOT / f"{scan_id}_closed_textured_mesh.ply")
    lesion_xyz, lesion_faces, lesion_rgb = combine_lesion_meshes(part_root, rows)
    body_plot_xyz, lesion_plot_xyz, center, scale = normalize_to_body(body_xyz, lesion_xyz)

    anchors = []
    hover_text = []
    for row in rows:
        metadata_path = part_root / "data" / row["metadata_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        anchor = (np.asarray(metadata["anchor_xyz"], dtype=np.float32) - center) / scale
        anchors.append(anchor)
        hover_text.append(
            f"{metadata['sample_id']}<br>"
            f"radius={metadata['radius_m'] * 1000:.1f} mm<br>"
            f"height={metadata['height_m'] * 1000:.1f} mm<br>"
            f"volume={metadata['spherical_cap_volume_ml']:.2f} mL"
        )
    anchors_arr = np.asarray(anchors, dtype=np.float32)

    traces = [
        go.Mesh3d(
            x=body_plot_xyz[:, 0],
            y=body_plot_xyz[:, 1],
            z=body_plot_xyz[:, 2],
            i=body_faces[:, 0],
            j=body_faces[:, 1],
            k=body_faces[:, 2],
            vertexcolor=rgb_strings(body_rgb),
            flatshading=False,
            lighting=dict(ambient=0.95, diffuse=0.55, specular=0.04, roughness=0.9),
            name=f"{scan_id} textured HSR body",
            hoverinfo="skip",
            visible=False,
            showlegend=True,
        ),
        go.Mesh3d(
            x=lesion_plot_xyz[:, 0],
            y=lesion_plot_xyz[:, 1],
            z=lesion_plot_xyz[:, 2],
            i=lesion_faces[:, 0],
            j=lesion_faces[:, 1],
            k=lesion_faces[:, 2],
            vertexcolor=rgb_strings(lesion_rgb),
            flatshading=False,
            lighting=dict(ambient=0.72, diffuse=0.76, specular=0.20, roughness=0.58),
            name=f"{len(rows)} synthetic {body_part} volumes",
            hoverinfo="skip",
            visible=False,
            showlegend=True,
        ),
        go.Scatter3d(
            x=anchors_arr[:, 0],
            y=anchors_arr[:, 1],
            z=anchors_arr[:, 2],
            mode="markers",
            marker=dict(size=3.5, color="rgba(180, 35, 35, 0.62)"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="volume centers",
            visible=False,
            showlegend=True,
        ),
    ]
    record = {
        "scan_id": scan_id,
        "body_part": body_part,
        "volume_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
    }
    return traces, record


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


def make_figure(part_root: Path, body_part: str, max_volumes_per_scan: int) -> tuple[go.Figure, list[dict[str, Any]]]:
    traces: list[Any] = []
    records: list[dict[str, Any]] = []
    trace_ranges = []
    for scan_id in SCAN_IDS:
        start = len(traces)
        scan_traces, record = build_scan_traces(part_root, body_part, scan_id, max_volumes_per_scan)
        traces.extend(scan_traces)
        records.append(record)
        trace_ranges.append((scan_id, start, len(scan_traces), record["volume_count"]))

    for trace_idx in range(trace_ranges[0][1], trace_ranges[0][1] + trace_ranges[0][2]):
        traces[trace_idx].visible = True

    buttons = []
    for scan_id, start, count, volume_count in trace_ranges:
        visible = [False] * len(traces)
        for trace_idx in range(start, start + count):
            visible[trace_idx] = True
        buttons.append(
            {
                "label": f"{scan_id} ({volume_count} volumes)",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{body_part}: {scan_id} with {volume_count} synthetic volumes"},
                ],
            }
        )

    initial_scan = trace_ranges[0][0]
    initial_count = trace_ranges[0][3]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{body_part}: {initial_scan} with {initial_count} synthetic volumes",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(242,244,247)",
            aspectmode="data",
            camera=camera_for_body_part(body_part),
        ),
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        width=1040,
        height=820,
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor="white",
        showlegend=True,
    )
    return fig, records


def notebook_source(body_part: str, max_volumes_per_scan: int) -> str:
    return f"""
from pathlib import Path
import csv, json
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

BODY_PART = {body_part!r}
MAX_VOLUMES_PER_SCAN = {max_volumes_per_scan}
SCAN_IDS = {SCAN_IDS!r}

REPO_ROOT = next(parent for parent in (Path.cwd(), *Path.cwd().parents) if (parent / 'data' / 'hsr').exists())
DATASET_ROOT = REPO_ROOT / 'data' / 'synthetic' / 'multiple_lesion' / 'body_parts' / 'physics_aug_growth' / 'body_parts_dataset' / BODY_PART
HSR_MESH_ROOT = REPO_ROOT / 'data' / 'hsr' / 'visualizations' / 'meshes'

def rgb_strings(rgb):
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return [f"rgb({{int(r)}},{{int(g)}},{{int(b)}})" for r, g, b in rgb]

def read_colored_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if mesh.has_vertex_colors():
        rgb = np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255.0), 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 185, dtype=np.uint8)
    return xyz, faces, rgb

def load_rows(scan_id):
    with (DATASET_ROOT / 'data' / 'manifest.csv').open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['scan_id'] == scan_id]
    rows.sort(key=lambda row: int(row['patient_volume_index']))
    return rows[:MAX_VOLUMES_PER_SCAN]

def combine_lesions(rows):
    xyzs, faces_list, rgbs = [], [], []
    offset = 0
    for row in rows:
        xyz, faces, rgb = read_colored_mesh(DATASET_ROOT / 'data' / row['mesh_path'])
        xyzs.append(xyz)
        faces_list.append(faces + offset)
        rgbs.append(rgb)
        offset += len(xyz)
    return np.vstack(xyzs), np.vstack(faces_list), np.vstack(rgbs)

def scan_traces(scan_id):
    rows = load_rows(scan_id)
    body_xyz, body_faces, body_rgb = read_colored_mesh(HSR_MESH_ROOT / f'{{scan_id}}_closed_textured_mesh.ply')
    lesion_xyz, lesion_faces, lesion_rgb = combine_lesions(rows)
    center = body_xyz.mean(axis=0)
    scale = float(np.max(np.ptp(body_xyz, axis=0))) or 1.0
    body_xyz = (body_xyz - center) / scale
    lesion_xyz = (lesion_xyz - center) / scale
    anchors, hover_text = [], []
    for row in rows:
        meta = json.loads((DATASET_ROOT / 'data' / row['metadata_path']).read_text(encoding='utf-8'))
        anchors.append((np.asarray(meta['anchor_xyz'], dtype=np.float32) - center) / scale)
        hover_text.append(
            f"{{meta['sample_id']}}<br>radius={{meta['radius_m'] * 1000:.1f}} mm<br>"
            f"height={{meta['height_m'] * 1000:.1f}} mm<br>volume={{meta['spherical_cap_volume_ml']:.2f}} mL"
        )
    anchors = np.asarray(anchors, dtype=np.float32)
    return [
        go.Mesh3d(
            x=body_xyz[:, 0], y=body_xyz[:, 1], z=body_xyz[:, 2],
            i=body_faces[:, 0], j=body_faces[:, 1], k=body_faces[:, 2],
            vertexcolor=rgb_strings(body_rgb),
            flatshading=False,
            lighting=dict(ambient=0.95, diffuse=0.55, specular=0.04, roughness=0.9),
            name=f"{{scan_id}} textured HSR body",
            hoverinfo='skip',
            visible=False,
        ),
        go.Mesh3d(
            x=lesion_xyz[:, 0], y=lesion_xyz[:, 1], z=lesion_xyz[:, 2],
            i=lesion_faces[:, 0], j=lesion_faces[:, 1], k=lesion_faces[:, 2],
            vertexcolor=rgb_strings(lesion_rgb),
            flatshading=False,
            lighting=dict(ambient=0.72, diffuse=0.76, specular=0.20, roughness=0.58),
            name=f"{{len(rows)}} synthetic {{BODY_PART}} volumes",
            hoverinfo='skip',
            visible=False,
        ),
        go.Scatter3d(
            x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2],
            mode='markers',
            marker=dict(size=3.5, color='rgba(180, 35, 35, 0.62)'),
            text=hover_text,
            hovertemplate='%{{text}}<extra></extra>',
            name='volume centers',
            visible=False,
        ),
    ], len(rows)

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

def make_body_part_volume_figure():
    all_traces = []
    trace_ranges = []
    for scan_id in SCAN_IDS:
        start = len(all_traces)
        traces, count = scan_traces(scan_id)
        all_traces.extend(traces)
        trace_ranges.append((scan_id, start, len(traces), count))
    for trace_idx in range(trace_ranges[0][1], trace_ranges[0][1] + trace_ranges[0][2]):
        all_traces[trace_idx].visible = True
    buttons = []
    for scan_id, start, count, volume_count in trace_ranges:
        visible = [False] * len(all_traces)
        for trace_idx in range(start, start + count):
            visible[trace_idx] = True
        buttons.append(dict(
            label=f"{{scan_id}} ({{volume_count}} volumes)",
            method='update',
            args=[{{'visible': visible}}, {{'title': f"{{BODY_PART}}: {{scan_id}} with {{volume_count}} synthetic volumes"}}],
        ))
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"{{BODY_PART}}: {{trace_ranges[0][0]}} with {{trace_ranges[0][3]}} synthetic volumes",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgb(242,244,247)', aspectmode='data',
            camera=camera_for_body_part(BODY_PART),
        ),
        updatemenus=[dict(buttons=buttons, direction='down', x=0.02, y=0.98, xanchor='left', yanchor='top')],
        width=1040, height=820,
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor='white',
        showlegend=True,
    )
    return fig
""".strip()


def write_notebook(
    part_root: Path,
    visualization_part_root: Path,
    body_part: str,
    max_volumes_per_scan: int,
) -> dict[str, Any]:
    fig, records = make_figure(part_root, body_part, max_volumes_per_scan)
    payload = json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))
    cells = [
        nbf.v4.new_markdown_cell(f"# {body_part} multiple-volume closed textured Plotly viewer"),
        nbf.v4.new_markdown_cell(
            f"This notebook shows the textured HSR body with up to {max_volumes_per_scan} "
            f"synthetic {body_part} volumes per patient scan. Use the dropdown to switch patients; "
            "drag/zoom the Plotly scene to inspect the volumes."
        ),
        nbf.v4.new_code_cell(notebook_source(body_part, max_volumes_per_scan)),
    ]
    output_cell = nbf.v4.new_code_cell("make_body_part_volume_figure()")
    output_cell["execution_count"] = 1
    output_cell["outputs"] = [
        nbf.v4.new_output(
            output_type="display_data",
            data={
                "application/vnd.plotly.v1+json": payload,
                "text/plain": f"<Plotly Figure: {body_part} multi-volume viewer>",
            },
            metadata={},
        )
    ]
    cells.append(output_cell)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    output_dir = visualization_part_root / "plotly"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / f"{body_part}_multiple_volume_closed_plotly_viewer.ipynb"
    nbf.write(nb, notebook_path)

    manifest_path = output_dir / f"{body_part}_multiple_volume_closed_plotly_manifest.json"
    manifest = {
        "body_part": body_part,
        "notebook": root_relative(notebook_path),
        "max_volumes_per_scan": max_volumes_per_scan,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "body_part": body_part,
        "notebook": root_relative(notebook_path),
        "manifest": root_relative(manifest_path),
        "volume_count_total": sum(record["volume_count"] for record in records),
        "records": records,
    }


def build_viewers(
    dataset_root: Path,
    visualization_root: Path,
    body_parts: list[str],
    max_volumes_per_scan: int,
) -> list[dict[str, Any]]:
    outputs = []
    for body_part in body_parts:
        part_root = dataset_root / body_part
        visualization_part_root = visualization_root / body_part
        if not part_root.exists():
            raise FileNotFoundError(f"Missing body-part folder: {part_root}")
        outputs.append(write_notebook(part_root, visualization_part_root, body_part, max_volumes_per_scan))
        print(outputs[-1]["notebook"], flush=True)
    summary_path = visualization_root / "visualizations_plotly_multi_volume_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=root_relative(DEFAULT_DATASET_ROOT))
    parser.add_argument("--visualization-root", default=root_relative(DEFAULT_VISUALIZATION_ROOT))
    parser.add_argument("--body-part", action="append", choices=BODY_PARTS, default=None)
    parser.add_argument("--max-volumes-per-scan", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    visualization_root = Path(args.visualization_root)
    if not visualization_root.is_absolute():
        visualization_root = ROOT / visualization_root
    if args.max_volumes_per_scan < 10 or args.max_volumes_per_scan > 100:
        raise ValueError("--max-volumes-per-scan must be between 10 and 100")
    outputs = build_viewers(dataset_root, visualization_root, args.body_part or BODY_PARTS, args.max_volumes_per_scan)
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()

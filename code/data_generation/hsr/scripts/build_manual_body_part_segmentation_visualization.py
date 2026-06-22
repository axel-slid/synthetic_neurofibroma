#!/usr/bin/env python3
"""Build rotating visualizations for manual HSR body-part segmentations."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

ROOT = Path(__file__).resolve().parents[4]
SEG_ROOT = ROOT / "data" / "hsr" / "body_part_segmentation"
MANUAL_DATA_DIR = SEG_ROOT / "manual" / "data"
VIZ_DIR = SEG_ROOT / "manual" / "visualizations"


def hex_to_rgb255(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def rgb_string(hex_color: str) -> str:
    red, green, blue = hex_to_rgb255(hex_color)
    return f"rgb({red},{green},{blue})"


def load_manual_manifest() -> dict[str, object]:
    manifest_path = MANUAL_DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manual segmentation manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def parse_params(npz: np.lib.npyio.NpzFile) -> dict[str, float]:
    raw = npz["params_json"]
    return json.loads(str(raw.item() if raw.shape == () else raw))


def transform_vertices(npz: np.lib.npyio.NpzFile, x_shift: float) -> np.ndarray:
    vertices = npz["vertices"].astype(np.float32)
    params = parse_params(npz)
    x_center = float(params["x_center"])
    z_min = float(params["z_min"])
    front_coord = npz["front_coord"].astype(np.float32)
    transformed = np.column_stack(
        [
            vertices[:, 0] - x_center + x_shift,
            front_coord,
            vertices[:, 2] - z_min,
        ]
    )
    return transformed.astype(np.float32)


def remap_faces(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.ravel())
    remap = np.full(int(faces.max()) + 1, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return used, remap[faces].astype(np.int32)


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= target_faces:
        return vertices.astype(np.float32), faces.astype(np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    return (
        np.asarray(simplified.vertices, dtype=np.float32),
        np.asarray(simplified.triangles, dtype=np.int32),
    )


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, np.maximum(lengths, 1e-8), out=np.zeros_like(normals), where=lengths > 0)


def split_scan_traces(
    npz_path: Path,
    scan_id: str,
    x_shift: float,
    label_names: list[str],
    label_colors: dict[str, str],
    target_faces_per_scan: int,
    base_target_faces_per_scan: int,
    overlay_opacity: float,
    base_opacity: float,
    overlay_offset: float,
    show_legend: bool,
) -> tuple[list[go.BaseTraceType], dict[str, object]]:
    npz = np.load(npz_path, allow_pickle=False)
    vertices = transform_vertices(npz, x_shift=x_shift)
    triangles = npz["triangles"].astype(np.int32)
    face_labels = npz["face_labels"].astype(np.uint8)
    vertex_normals = compute_vertex_normals(vertices, triangles)
    total_labeled_faces = max(int(np.sum(np.isin(face_labels, np.arange(len(label_names))))), 1)

    traces: list[go.BaseTraceType] = []
    label_face_counts: dict[str, int] = {}
    simplified_face_counts: dict[str, int] = {}

    base_vertices, base_faces = simplify_mesh(vertices, triangles, base_target_faces_per_scan)
    traces.append(
        go.Mesh3d(
            x=base_vertices[:, 0],
            y=base_vertices[:, 1],
            z=base_vertices[:, 2],
            i=base_faces[:, 0],
            j=base_faces[:, 1],
            k=base_faces[:, 2],
            color="rgb(214,196,184)",
            opacity=base_opacity,
            flatshading=False,
            lighting=dict(ambient=0.82, diffuse=0.72, specular=0.03, roughness=0.92),
            hovertemplate=f"{scan_id}<br>body surface<extra></extra>",
            name="body surface",
            legendgroup="body surface",
            showlegend=show_legend,
        )
    )

    for label_id, label_name in enumerate(label_names):
        face_mask = face_labels == label_id
        if not np.any(face_mask):
            continue
        label_faces = triangles[face_mask]
        label_face_counts[label_name] = int(len(label_faces))
        used, remapped = remap_faces(label_faces)
        part_vertices = vertices[used] + vertex_normals[used] * overlay_offset
        proportional_target = max(80, int(target_faces_per_scan * len(label_faces) / total_labeled_faces))
        part_vertices, remapped = simplify_mesh(part_vertices, remapped, proportional_target)
        simplified_face_counts[label_name] = int(len(remapped))
        traces.append(
            go.Mesh3d(
                x=part_vertices[:, 0],
                y=part_vertices[:, 1],
                z=part_vertices[:, 2],
                i=remapped[:, 0],
                j=remapped[:, 1],
                k=remapped[:, 2],
                color=rgb_string(label_colors[label_name]),
                opacity=overlay_opacity,
                flatshading=False,
                lighting=dict(ambient=0.88, diffuse=0.62, specular=0.025, roughness=0.9),
                hovertemplate=f"{scan_id}<br>{label_name}<extra></extra>",
                name=label_name,
                legendgroup=label_name,
                showlegend=show_legend,
            )
        )

    z_max = float(np.max(vertices[:, 2]))
    traces.append(
        go.Scatter3d(
            x=[x_shift],
            y=[0.0],
            z=[z_max + 0.09],
            mode="text",
            text=[scan_id],
            textfont=dict(size=14, color="rgb(32,34,38)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    metadata = {
        "scan_id": scan_id,
        "npz": str(npz_path.relative_to(ROOT)),
        "raw_label_faces": label_face_counts,
        "plotly_base_faces": int(len(base_faces)),
        "plotly_label_faces": simplified_face_counts,
        "height_m": float(np.asarray(npz["height"]).item()),
    }
    return traces, metadata


def camera_for_angle(angle: float) -> dict[str, dict[str, float]]:
    radius = 2.45
    return {
        "eye": {
            "x": radius * math.sin(angle),
            "y": radius * math.cos(angle),
            "z": 0.42,
        },
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }


def make_figure(
    traces: list[go.BaseTraceType],
    bounds_xyz: np.ndarray,
    frames: int,
) -> go.Figure:
    animation_frames = []
    for frame_index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, frames, endpoint=False)):
        animation_frames.append(
            go.Frame(
                name=f"{frame_index + 1:03d}",
                layout=go.Layout(scene_camera=camera_for_angle(float(angle))),
            )
        )

    steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in animation_frames
    ]

    xyz_min = bounds_xyz.min(axis=0)
    xyz_max = bounds_xyz.max(axis=0)
    pad = np.array([0.10, 0.10, 0.05], dtype=np.float32)
    fig = go.Figure(data=traces, frames=animation_frames)
    fig.update_layout(
        title=dict(text="Manual HSR body-part segmentation overlay - two scans", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[float(xyz_min[0] - pad[0]), float(xyz_max[0] + pad[0])]),
            yaxis=dict(visible=False, range=[float(xyz_min[1] - pad[1]), float(xyz_max[1] + pad[1])]),
            zaxis=dict(visible=False, range=[float(xyz_min[2] - pad[2]), float(xyz_max[2] + pad[2])]),
            bgcolor="rgb(248,249,251)",
            aspectmode="data",
            camera=camera_for_angle(0.0),
        ),
        width=1250,
        height=880,
        margin=dict(l=0, r=0, t=58, b=0),
        paper_bgcolor="white",
        legend=dict(
            x=0.015,
            y=0.985,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(210,210,210,0.9)",
            borderwidth=1,
            itemsizing="constant",
        ),
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "len": 0.86,
                "steps": steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 90, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def compact_payload(value: object) -> object:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [compact_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_payload(item) for key, item in value.items()}
    return value


def write_code_free_notebook(notebook_path: Path, figure: go.Figure) -> None:
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    payload = compact_payload(payload)
    cells = [
        nbf.v4.new_code_cell(
            source="",
            execution_count=None,
            metadata={"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
            outputs=[
                nbf.v4.new_output(
                    output_type="display_data",
                    data={
                        "application/vnd.plotly.v1+json": payload,
                        "text/plain": "<Plotly Figure: manual HSR body-part segmentation>",
                    },
                    metadata={},
                )
            ],
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", pygments_lexer="ipython3"),
        ),
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)


def render_rotation_gif(figure: go.Figure, gif_path: Path, fps: int) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    working = go.Figure(data=figure.data, layout=figure.layout)
    images = []
    with tempfile.TemporaryDirectory(prefix=f"{gif_path.stem}_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for frame_index, frame in enumerate(figure.frames):
            if frame.layout and frame.layout.scene and frame.layout.scene.camera:
                working.update_layout(scene_camera=frame.layout.scene.camera)
            png_path = tmp_dir / f"frame_{frame_index:03d}.png"
            working.write_image(png_path, scale=1)
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=1 / fps, loop=0)


def build_visualization(args: argparse.Namespace) -> None:
    manifest = load_manual_manifest()
    label_names = [str(name) for name in manifest["labels"]]
    label_colors = {str(key): str(value) for key, value in manifest["label_colors"].items()}
    scans = manifest["scans"]
    if len(scans) != 2:
        raise ValueError(f"Expected exactly two scans in manual segmentation manifest; got {len(scans)}")

    shifts = [-args.spacing / 2.0, args.spacing / 2.0]
    traces: list[go.BaseTraceType] = []
    bounds_parts = []
    scan_metadata = []
    for idx, (scan, shift) in enumerate(zip(scans, shifts)):
        scan_id = str(scan["scan_id"])
        npz_path = ROOT / str(scan["npz"])
        scan_traces, metadata = split_scan_traces(
            npz_path=npz_path,
            scan_id=scan_id,
            x_shift=shift,
            label_names=label_names,
            label_colors=label_colors,
            target_faces_per_scan=args.target_faces_per_scan,
            base_target_faces_per_scan=args.base_target_faces_per_scan,
            overlay_opacity=args.overlay_opacity,
            base_opacity=args.base_opacity,
            overlay_offset=args.overlay_offset,
            show_legend=(idx == 0),
        )
        traces.extend(scan_traces)
        scan_npz = np.load(npz_path, allow_pickle=False)
        bounds_parts.append(transform_vertices(scan_npz, x_shift=shift))
        scan_metadata.append(metadata)

    figure = make_figure(traces, np.vstack(bounds_parts), frames=args.frames)
    gif_path = VIZ_DIR / "manual_body_part_segmentation_two_people_rotating.gif"
    notebook_path = VIZ_DIR / "manual_body_part_segmentation_two_people_rotating.ipynb"
    render_rotation_gif(figure, gif_path, fps=args.fps)
    write_code_free_notebook(notebook_path, figure)

    viz_manifest = {
        "dataset": "manual_hsr_body_part_segmentation_visualization",
        "source_manifest": str((MANUAL_DATA_DIR / "manifest.json").relative_to(ROOT)),
        "gif": str(gif_path.relative_to(ROOT)),
        "notebook": str(notebook_path.relative_to(ROOT)),
        "frames": args.frames,
        "fps": args.fps,
        "target_faces_per_scan": args.target_faces_per_scan,
        "base_target_faces_per_scan": args.base_target_faces_per_scan,
        "overlay_opacity": args.overlay_opacity,
        "base_opacity": args.base_opacity,
        "overlay_offset_m": args.overlay_offset,
        "scan_ids": [str(scan["scan_id"]) for scan in scans],
        "labels": label_names,
        "label_colors": label_colors,
        "scans": scan_metadata,
    }
    manifest_path = VIZ_DIR / "manual_body_part_segmentation_two_people_manifest.json"
    manifest_path.write_text(json.dumps(viz_manifest, indent=2), encoding="utf-8")
    print(gif_path)
    print(notebook_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--target-faces-per-scan", type=int, default=24000)
    parser.add_argument("--base-target-faces-per-scan", type=int, default=18000)
    parser.add_argument("--overlay-opacity", type=float, default=0.48)
    parser.add_argument("--base-opacity", type=float, default=0.72)
    parser.add_argument("--overlay-offset", type=float, default=0.003)
    parser.add_argument("--spacing", type=float, default=1.35)
    return parser.parse_args()


def main() -> None:
    build_visualization(parse_args())


if __name__ == "__main__":
    main()

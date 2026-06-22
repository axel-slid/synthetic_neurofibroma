#!/usr/bin/env python3
"""Build heuristic body-part segmentations for decimated HSR body meshes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import nbformat
import numpy as np
import open3d as o3d
from matplotlib.colors import rgb_to_hsv
from nbclient import NotebookClient

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]

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


@dataclass
class SegmentationParams:
    x_center: float
    y_center: float
    z_min: float
    z_max: float
    height: float
    x_span: float
    y_span: float
    front_sign: int
    front_low_head_median_z: float
    front_high_head_median_z: float
    torso_half_width: float
    arm_x_threshold: float
    hand_x_threshold: float
    hand_z_norm_cut: float
    foot_z_norm_cut: float
    leg_z_norm_cut: float
    face_z_norm_min: float
    face_z_norm_max: float


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    color = hex_color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


def hex_to_rgb255(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


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


def segment_vertices(
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    leg_z_norm_cut: float,
    face_z_norm_min: float,
    face_z_norm_max: float,
) -> tuple[np.ndarray, SegmentationParams, np.ndarray]:
    x, y, z = vertices.T
    z_min = float(z.min())
    z_max = float(z.max())
    height = max(z_max - z_min, np.finfo(float).eps)
    z_norm = (z - z_min) / height
    x_span = max(float(x.max() - x.min()), np.finfo(float).eps)
    y_span = max(float(y.max() - y.min()), np.finfo(float).eps)

    body_center_mask = (z_norm > 0.35) & (z_norm < 0.75)
    x_center = float(np.median(x[body_center_mask])) if body_center_mask.any() else float(np.median(x))
    abs_x = np.abs(x - x_center)

    head_mask = (z_norm > 0.78) & (z_norm < 0.94) & (abs_x < 0.18 * x_span)
    if head_mask.sum() < 100:
        head_mask = (z_norm > 0.75) & (z_norm < 0.96)
    head_y = y[head_mask]
    y_low_cut, y_high_cut = np.quantile(head_y, [0.18, 0.82])
    low_head_side = head_mask & (y <= y_low_cut)
    high_head_side = head_mask & (y >= y_high_cut)
    low_head_median_z = float(np.median(z[low_head_side]))
    high_head_median_z = float(np.median(z[high_head_side]))

    # In these A-pose HSR scans, the anterior lower-face side sits lower than
    # the posterior cap/back-of-head side. This also handles scans whose y
    # direction is flipped.
    front_sign = 1 if high_head_median_z < low_head_median_z else -1

    torso_ref = (z_norm > 0.45) & (z_norm < 0.75) & (abs_x < 0.26 * x_span)
    y_center = float(np.median(y[torso_ref])) if torso_ref.any() else float(np.median(y))
    front_coord = front_sign * (y - y_center)

    torso_band = (z_norm > 0.47) & (z_norm < 0.76)
    if torso_band.any():
        torso_half_width = float(max(np.quantile(abs_x[torso_band], 0.58), 0.16 * x_span))
    else:
        torso_half_width = float(0.18 * x_span)
    shoulder_band = (z_norm > 0.62) & (z_norm < 0.82)
    shoulder_half_width = float(np.quantile(abs_x[shoulder_band], 0.64)) if shoulder_band.any() else torso_half_width
    arm_x_threshold = float(max(0.22 * x_span, min(shoulder_half_width * 0.95, 0.33 * x_span)))

    raw_arms = ((z_norm > 0.33) & (z_norm < 0.79) & (abs_x > arm_x_threshold)) | (
        (z_norm > 0.58)
        & (z_norm < 0.86)
        & (abs_x > max(0.18 * x_span, torso_half_width * 0.86))
    )
    raw_arms &= z_norm > 0.37
    raw_legs = (z_norm < leg_z_norm_cut) & ~raw_arms

    hand_z_norm_cut = 0.50
    hand_extra_z_norm_cut = 0.54
    hand_x_threshold = float(max(arm_x_threshold * 0.96, 0.30 * x_span))
    distal_hand_x_threshold = float(max(hand_x_threshold, 0.32 * x_span))
    hands = raw_arms & (abs_x > hand_x_threshold) & (
        (z_norm < hand_z_norm_cut)
        | ((z_norm < hand_extra_z_norm_cut) & (abs_x > distal_hand_x_threshold))
    )
    foot_z_norm_cut = 0.14
    feet = raw_legs & (z_norm < foot_z_norm_cut)
    arms = raw_arms & ~hands
    legs = raw_legs & ~feet

    face_front_cut = float(np.quantile(front_coord[head_mask], 0.42))
    face = (
        (z_norm > face_z_norm_min)
        & (z_norm < face_z_norm_max)
        & (abs_x < 0.19 * x_span)
        & (front_coord > face_front_cut)
    )

    colors = np.clip(vertex_colors, 0.0, 1.0)
    hsv = rgb_to_hsv(colors.reshape(-1, 1, 3)).reshape(-1, 3)
    hue, saturation, value = hsv.T
    red, green, blue = colors.T

    cap = (
        (z_norm > 0.88)
        & (z_norm < 0.985)
        & (abs_x < 0.24 * x_span)
        & (saturation > 0.25)
        & (value > 0.25)
        & (hue > 0.43)
        & (hue < 0.62)
    )
    lower_clothing_region = (z_norm > 0.35) & (z_norm < 0.59) & (abs_x < 0.34 * x_span) & ~raw_arms
    gray_or_white_fabric = (saturation < 0.18) & (value > 0.18)
    dark_gray_fabric = (saturation < 0.30) & (value < 0.32)
    red_fabric = (
        (red > 0.35)
        & (red > green * 1.38)
        & (red > blue * 1.30)
        & ((green / np.maximum(red, 1e-6)) < 0.69)
        & (saturation > 0.25)
    )
    blue_fabric = (hue > 0.43) & (hue < 0.66) & (saturation > 0.25) & (value > 0.20)
    lower_clothes = lower_clothing_region & (
        gray_or_white_fabric | dark_gray_fabric | red_fabric | blue_fabric
    )
    clothes = cap | lower_clothes

    labels = np.full(len(vertices), LABEL_ID["back"], dtype=np.uint8)
    labels[front_coord >= 0] = LABEL_ID["front"]
    labels[legs] = LABEL_ID["legs"]
    labels[arms] = LABEL_ID["arms"]
    labels[feet] = LABEL_ID["feet"]
    labels[hands] = LABEL_ID["hands"]
    labels[face & ~raw_arms] = LABEL_ID["face"]
    labels[clothes] = LABEL_ID["clothes"]

    params = SegmentationParams(
        x_center=x_center,
        y_center=y_center,
        z_min=z_min,
        z_max=z_max,
        height=float(height),
        x_span=x_span,
        y_span=y_span,
        front_sign=int(front_sign),
        front_low_head_median_z=low_head_median_z,
        front_high_head_median_z=high_head_median_z,
        torso_half_width=torso_half_width,
        arm_x_threshold=arm_x_threshold,
        hand_x_threshold=hand_x_threshold,
        hand_z_norm_cut=float(hand_z_norm_cut),
        foot_z_norm_cut=float(foot_z_norm_cut),
        leg_z_norm_cut=float(leg_z_norm_cut),
        face_z_norm_min=float(face_z_norm_min),
        face_z_norm_max=float(face_z_norm_max),
    )
    return labels, params, front_coord


def read_mesh(mesh_path: Path) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    else:
        vertex_colors = np.full_like(vertices, 0.72, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    return mesh, vertices, triangles, vertex_colors, normals


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


def make_rotation_gif(
    out_path: Path,
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    vertex_labels: np.ndarray,
    params: SegmentationParams,
    scan_id: str,
    max_points: int,
    frames: int,
    fps: int,
) -> None:
    rng = np.random.default_rng(7)
    if len(vertices) > max_points:
        sample_idx = rng.choice(len(vertices), size=max_points, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(len(vertices))

    v = vertices[sample_idx]
    texture_rgb = np.clip(np.rint(vertex_colors[sample_idx] * 255.0), 0, 255).astype(np.uint8)
    labels = vertex_labels[sample_idx]
    lateral = v[:, 0] - params.x_center
    anterior = params.front_sign * (v[:, 1] - params.y_center)
    z = v[:, 2]
    radius = float(np.quantile(np.sqrt(lateral**2 + anterior**2), 0.995)) * 1.08
    z_pad = 0.04 * params.height
    label_rgb = np.asarray([hex_to_rgb255(LABEL_COLORS[name]) for name in LABEL_NAMES], dtype=np.uint8)
    colors = label_rgb[labels].copy()
    clothes_mask = labels == LABEL_ID["clothes"]
    colors[clothes_mask] = texture_rgb[clothes_mask]
    frame_images = []
    width, height_px = 640, 819
    plot_left, plot_top, plot_right, plot_bottom = 20, 42, width - 20, height_px - 18
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    point_radius = 2
    legend_rows = [(name, label_rgb[LABEL_ID[name]]) for name in LABEL_NAMES]

    for angle in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        horizontal = lateral * np.cos(angle) + anterior * np.sin(angle)
        depth = -lateral * np.sin(angle) + anterior * np.cos(angle)
        draw_order = np.argsort(depth)

        image = Image.new("RGB", (width, height_px), "white")
        draw = ImageDraw.Draw(image)
        title = f"{scan_id} body part segmentation"
        title_box = draw.textbbox((0, 0), title)
        draw.text(((width - (title_box[2] - title_box[0])) // 2, 8), title, fill=(0, 0, 0))

        x_px = plot_left + (horizontal + radius) / (2 * radius) * plot_width
        y_px = plot_top + (params.z_max + z_pad - z) / (params.height + 2 * z_pad) * plot_height
        for idx in draw_order:
            x_i = int(round(x_px[idx]))
            y_i = int(round(y_px[idx]))
            color = tuple(int(channel) for channel in colors[idx])
            draw.ellipse(
                (
                    x_i - point_radius,
                    y_i - point_radius,
                    x_i + point_radius,
                    y_i + point_radius,
                ),
                fill=color,
            )

        legend_x, legend_y = 56, height_px - 164
        legend_w, legend_h = 112, 136
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
            radius=4,
            fill=(255, 255, 255),
            outline=(210, 210, 210),
        )
        for row_idx, (label_name, color) in enumerate(legend_rows):
            y_i = legend_y + 12 + row_idx * 15
            swatch = tuple(int(channel) for channel in color)
            draw.ellipse((legend_x + 13, y_i + 2, legend_x + 22, y_i + 11), fill=swatch)
            text = "clothes texture" if label_name == "clothes" else label_name
            draw.text((legend_x + 32, y_i), text, fill=(0, 0, 0))

        frame_images.append(np.asarray(image))

    imageio.mimsave(out_path, frame_images, duration=1 / fps, loop=0)


def make_combined_gif(viz_dir: Path, scan_ids: list[str], fps: int) -> Path:
    gif_paths = [viz_dir / f"{scan_id}_body_part_segmentation_overlay.gif" for scan_id in scan_ids]
    for gif_path in gif_paths:
        if not gif_path.exists():
            raise FileNotFoundError(f"Missing GIF for combined visualization: {gif_path}")

    readers = [imageio.mimread(path) for path in gif_paths]
    frame_count = min(len(frames) for frames in readers)
    combined_frames = []
    gap = 18
    for frame_idx in range(frame_count):
        frames_rgb = []
        for frames in readers:
            frame = np.asarray(frames[frame_idx])
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            frames_rgb.append(frame)
        max_h = max(frame.shape[0] for frame in frames_rgb)
        total_w = sum(frame.shape[1] for frame in frames_rgb) + gap * (len(frames_rgb) - 1)
        canvas = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
        x = 0
        for frame in frames_rgb:
            y = (max_h - frame.shape[0]) // 2
            canvas[y : y + frame.shape[0], x : x + frame.shape[1]] = frame
            x += frame.shape[1] + gap
        combined_frames.append(canvas)

    out_path = viz_dir / "HSR_body_part_segmentation_overlay_combined.gif"
    imageio.mimsave(out_path, combined_frames, duration=1 / fps, loop=0)
    return out_path


def build_plotly_notebook(notebook_path: Path) -> None:
    source = r'''
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook_connected"

DATA_DIR = Path("../data").resolve()
SCAN_FILES = sorted(DATA_DIR.glob("*_body_part_segmentation.npz"))
LABEL_NAMES = ["front", "back", "face", "arms", "hands", "legs", "feet", "clothes"]
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


def remap_faces(triangles, face_mask):
    faces = triangles[face_mask]
    used = np.unique(faces)
    inverse = np.full(triangles.max() + 1, -1, dtype=np.int32)
    inverse[used] = np.arange(len(used), dtype=np.int32)
    return used, inverse[faces]


def scan_id_from_path(path):
    return path.name.replace("_body_part_segmentation.npz", "")


def make_traces(npz_path, visible=False):
    data = np.load(npz_path)
    vertices = data["vertices"]
    triangles = data["triangles"]
    vertex_colors = data["vertex_colors"]
    normals = data["vertex_normals"]
    face_labels = data["face_labels"]
    height = float(data["height"])
    scan_id = scan_id_from_path(npz_path)
    overlay_vertices = vertices + normals * (0.004 * height)

    traces = [
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            vertexcolor=vertex_colors,
            opacity=1.0,
            name=f"{scan_id} texture",
            legendgroup=scan_id,
            hoverinfo="skip",
            visible=visible,
            showlegend=True,
        )
    ]

    for label_id, label_name in enumerate(LABEL_NAMES):
        face_mask = face_labels == label_id
        if not np.any(face_mask):
            continue
        used, remapped = remap_faces(triangles, face_mask)
        sub_vertices = overlay_vertices[used]
        traces.append(
            go.Mesh3d(
                x=sub_vertices[:, 0],
                y=sub_vertices[:, 1],
                z=sub_vertices[:, 2],
                i=remapped[:, 0],
                j=remapped[:, 1],
                k=remapped[:, 2],
                vertexcolor=vertex_colors[used] if label_name == "clothes" else None,
                color=None if label_name == "clothes" else LABEL_COLORS[label_name],
                opacity=1.0,
                name="clothes texture" if label_name == "clothes" else label_name,
                legendgroup=scan_id,
                hovertemplate=f"{scan_id}<br>{label_name}<extra></extra>",
                visible=visible,
                showlegend=True,
            )
        )
    return traces, int(data["front_sign"])


all_traces = []
scan_ranges = []
for idx, npz_path in enumerate(SCAN_FILES):
    scan_id = scan_id_from_path(npz_path)
    traces, front_sign = make_traces(npz_path, visible=(idx == 0))
    start = len(all_traces)
    all_traces.extend(traces)
    scan_ranges.append((scan_id, start, len(traces), front_sign))

buttons = []
for scan_id, start, count, front_sign in scan_ranges:
    visible = [False] * len(all_traces)
    for trace_idx in range(start, start + count):
        visible[trace_idx] = True
    buttons.append(
        {
            "label": scan_id,
            "method": "update",
            "args": [
                {"visible": visible},
                {
                    "title": f"{scan_id} body-part segmentation overlay",
                    "scene": {"camera": {"eye": {"x": 0.0, "y": 2.25 * front_sign, "z": 0.45}}},
                },
            ],
        }
    )

initial_title = f"{scan_ranges[0][0]} body-part segmentation overlay" if scan_ranges else "HSR body-part segmentation overlay"
initial_front_sign = scan_ranges[0][3] if scan_ranges else 1
fig = go.Figure(data=all_traces)
fig.update_layout(
    title=initial_title,
    width=1000,
    height=850,
    margin={"l": 0, "r": 0, "t": 52, "b": 0},
    scene={
        "aspectmode": "data",
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "zaxis": {"visible": False},
        "camera": {"eye": {"x": 0.0, "y": 2.25 * initial_front_sign, "z": 0.45}},
    },
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
    legend={"itemsizing": "constant"},
)
fig
'''.strip()

    markdown = (
        "## HSR Body-Part Segmentation Overlay\n\n"
        "The colored overlay is a deterministic geometry and texture-color heuristic over the decimated HSR meshes. "
        "Use the dropdown to switch scans; drag the Plotly scene to inspect front, back, face, arms, hands, legs, feet, and clothes."
    )
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(markdown),
            nbformat.v4.new_code_cell(source),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(notebook_path.parent)}})
    client.execute()
    nbformat.write(nb, notebook_path)


def process_scan(
    mesh_path: Path,
    scan_id: str,
    data_dir: Path,
    viz_dir: Path,
    leg_z_norm_cut: float,
    face_z_norm_min: float,
    face_z_norm_max: float,
    gif_frames: int,
    gif_fps: int,
    gif_points: int,
) -> dict[str, object]:
    mesh, vertices, triangles, vertex_colors, normals = read_mesh(mesh_path)
    vertex_labels, params, front_coord = segment_vertices(
        vertices,
        vertex_colors,
        leg_z_norm_cut=leg_z_norm_cut,
        face_z_norm_min=face_z_norm_min,
        face_z_norm_max=face_z_norm_max,
    )
    face_labels = majority_face_labels(triangles, vertex_labels)
    label_counts = {name: int(np.sum(vertex_labels == label_id)) for label_id, name in enumerate(LABEL_NAMES)}

    npz_path = data_dir / f"{scan_id}_body_part_segmentation.npz"
    label_mesh_path = data_dir / f"{scan_id}_body_part_colored_mesh.ply"
    gif_path = viz_dir / f"{scan_id}_body_part_segmentation_overlay.gif"

    np.savez_compressed(
        npz_path,
        scan_id=np.asarray(scan_id),
        vertices=vertices.astype(np.float32),
        triangles=triangles.astype(np.int32),
        vertex_colors=np.clip(np.rint(vertex_colors * 255.0), 0, 255).astype(np.uint8),
        vertex_normals=normals.astype(np.float32),
        vertex_labels=vertex_labels.astype(np.uint8),
        face_labels=face_labels.astype(np.uint8),
        label_names=np.asarray(LABEL_NAMES),
        label_colors=np.asarray([LABEL_COLORS[name] for name in LABEL_NAMES]),
        front_coord=front_coord.astype(np.float32),
        front_sign=np.asarray(params.front_sign, dtype=np.int8),
        height=np.asarray(params.height, dtype=np.float32),
        params_json=np.asarray(json.dumps(asdict(params), indent=2, sort_keys=True)),
    )
    write_label_mesh(label_mesh_path, vertices, triangles, vertex_labels)
    make_rotation_gif(
        gif_path,
        vertices,
        vertex_colors,
        vertex_labels,
        params,
        scan_id,
        max_points=gif_points,
        frames=gif_frames,
        fps=gif_fps,
    )

    return {
        "scan_id": scan_id,
        "source_mesh": str(mesh_path.relative_to(ROOT)),
        "npz": str(npz_path.relative_to(ROOT)),
        "colored_mesh": str(label_mesh_path.relative_to(ROOT)),
        "gif": str(gif_path.relative_to(ROOT)),
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "front_sign": int(params.front_sign),
        "label_counts": label_counts,
        "params": asdict(params),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsr-root", type=Path, default=ROOT / "data" / "hsr")
    parser.add_argument("--dataset-name", default="body_part_segmentation")
    parser.add_argument("--scan-id", action="append", default=None)
    parser.add_argument("--leg-z-norm-cut", type=float, default=0.51)
    parser.add_argument("--face-z-norm-min", type=float, default=0.825)
    parser.add_argument("--face-z-norm-max", type=float, default=0.965)
    parser.add_argument("--gif-frames", type=int, default=36)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-points", type=int, default=80_000)
    args = parser.parse_args()

    dataset_dir = args.hsr_root / args.dataset_name
    data_dir = dataset_dir / "data"
    viz_dir = dataset_dir / "visualizations"
    data_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    scan_ids = args.scan_id or ["HSR0018-Body-070", "HSR0152-Body-090"]
    mesh_dir = args.hsr_root / "visualizations" / "meshes"
    summary = {
        "dataset": args.dataset_name,
        "method": "deterministic A-pose geometry and texture-color heuristic on decimated HSR meshes",
        "labels": LABEL_NAMES,
        "label_colors": LABEL_COLORS,
        "scans": [],
    }

    for scan_id in scan_ids:
        mesh_path = mesh_dir / f"{scan_id}_closed_textured_mesh.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing decimated HSR mesh: {mesh_path}")
        scan_summary = process_scan(
            mesh_path,
            scan_id,
            data_dir,
            viz_dir,
            leg_z_norm_cut=args.leg_z_norm_cut,
            face_z_norm_min=args.face_z_norm_min,
            face_z_norm_max=args.face_z_norm_max,
            gif_frames=args.gif_frames,
            gif_fps=args.gif_fps,
            gif_points=args.gif_points,
        )
        summary["scans"].append(scan_summary)
        counts = ", ".join(f"{name}={count:,}" for name, count in scan_summary["label_counts"].items())
        print(f"{scan_id}: front_sign={scan_summary['front_sign']}; {counts}")

    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    notebook_path = viz_dir / "hsr_body_part_segmentation_overlay.ipynb"
    build_plotly_notebook(notebook_path)
    combined_gif_path = make_combined_gif(viz_dir, scan_ids, args.gif_fps)
    print(f"Wrote manifest: {manifest_path.relative_to(ROOT)}")
    print(f"Wrote Plotly notebook: {notebook_path.relative_to(ROOT)}")
    print(f"Wrote combined GIF: {combined_gif_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

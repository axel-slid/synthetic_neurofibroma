#!/usr/bin/env python3
"""Build the body-part physics-growth GIF used by the README."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
from matplotlib import colormaps
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_multiple_lesion_physics_dataset import (  # noqa: E402
    HSR_MESH_ROOT,
    LesionSpec,
    build_local_shape,
    read_colored_ply,
)

ROOT = Path(__file__).resolve().parents[4]
BODY_PART_ROOT_CANDIDATES = (
    ROOT / "data" / "synthetic" / "multiple_lesion_physics" / "body_parts",
    ROOT / "data" / "synthetic" / "multiple_lesion_physics_simple" / "body_parts",
)
DEFAULT_BODY_PART_ROOT = next((path for path in BODY_PART_ROOT_CANDIDATES if path.exists()), BODY_PART_ROOT_CANDIDATES[0])
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "multiple_lesion_physics_growth_progression.gif"
DEFAULT_BODY_PARTS = ("back", "face", "front")
PANEL_WIDTH = 430
PANEL_HEIGHT = 250
PANEL_GAP = 8
ROW_GAP = 6
HEADER_HEIGHT = 0
BACKGROUND = np.array([244, 246, 249], dtype=np.uint8)


@dataclass
class BodyPartSimulation:
    body_part: str
    sample_id: str
    scan_id: str
    frame_count: int
    body_vertices: np.ndarray
    body_faces: np.ndarray
    body_colors: np.ndarray
    lesions: list[dict[str, Any]]
    specs: list[LesionSpec]


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return (vector / length).astype(np.float32)


def look_at_camera_to_world(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def rgba(rgb: np.ndarray) -> np.ndarray:
    alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=1)


def depth_to_rainbow(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    valid = depth > 0.0
    normalized_depth = np.zeros_like(depth, dtype=np.float32)
    normalized_depth[valid] = np.clip((depth[valid] - depth_min) / max(depth_max - depth_min, 1e-6), 0.0, 1.0)
    # Invert so nearer anatomy is warm/red and farther anatomy is cool/violet.
    mapped = colormaps["rainbow"](1.0 - normalized_depth)[:, :, :3]
    depth_rgb = np.full((*depth.shape, 3), BACKGROUND, dtype=np.uint8)
    depth_rgb[valid] = np.clip(np.rint(mapped[valid] * 255.0), 0, 255).astype(np.uint8)
    return depth_rgb


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def add_view_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), label, fill=(20, 20, 20), font=font(13))
    return np.asarray(image)


def method_root(body_part_root: Path, body_part: str) -> Path:
    return body_part_root / body_part / "physics"


def select_metadata(body_part_root: Path, body_part: str) -> dict[str, Any]:
    root = method_root(body_part_root, body_part)
    manifest_path = root / "data" / "camera_depth_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {manifest_path}")

    preferred_scan_rows = [row for row in rows if row.get("scan_id") == "HSR0018-Body-070"]
    candidate_rows = preferred_scan_rows or rows
    row = max(candidate_rows, key=lambda item: int(item.get("lesion_count", 0)))
    metadata_path = root / "data" / row["metadata_path"]
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def spec_from_metadata(lesion: dict[str, Any]) -> LesionSpec:
    final_height = float(lesion["final_height_m"])
    lesion_index = int(lesion["lesion_index"])
    # Match the deterministic reconstruction used by the body-part Plotly viewer.
    lateral = 0.42 * math.sin(1.61803398875 * (lesion_index + 1))
    twist = 0.58 * math.cos(0.754877666 * (lesion_index + 3))
    lobe_amp = 0.018 + 0.012 * (0.5 + 0.5 * math.sin(0.37 * (lesion_index + 5)))
    pear_bias = 0.28 + 0.14 * (0.5 + 0.5 * math.cos(0.29 * (lesion_index + 7)))
    anchor = [float(value) for value in lesion["anchor_xyz"]]
    return LesionSpec(
        lesion_id=f"lesion_{lesion_index:03d}",
        target_x=anchor[0],
        target_y=anchor[1],
        target_z=anchor[2],
        target_vertex_index=int(lesion["face_index"]),
        final_height=final_height,
        support_radius=float(lesion["support_radius_m"]),
        neck_radius=float(lesion["neck_radius_m"]),
        bulb_radius=float(lesion["bulb_radius_m"]),
        stalk_fraction=float(lesion["stalk_fraction"]),
        gravity_scale=float(lesion["gravity_scale"]),
        flop_distance=float(lesion["flop_distance_m"]),
        arch_height=max(0.004, 0.24 * final_height),
        distal_center_height=max(0.002, 0.12 * final_height),
        sag=max(0.002, 0.08 * final_height),
        lateral=float(lateral),
        twist=float(twist),
        lobe_amp=float(lobe_amp),
        pear_bias=float(pear_bias),
        growth_delay=float(lesion["growth_delay"]),
        growth_duration=float(lesion["growth_duration"]),
        growth_power=float(lesion["growth_power"]),
        growth_rate=float(lesion["growth_rate"]),
        color_rgb=tuple(int(value) for value in lesion["lesion_rgb"]),
    )


def lesion_vertices_for_frame(
    lesion: dict[str, Any],
    spec: LesionSpec,
    frame_index: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = np.asarray(lesion["anchor_xyz"], dtype=np.float32)
    normal = normalized(np.asarray(lesion["normal_xyz"], dtype=np.float32))
    tangent_u = normalized(np.asarray(lesion["tangent_u_xyz"], dtype=np.float32))
    tangent_v = normalized(np.asarray(lesion["tangent_v_xyz"], dtype=np.float32))
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_direction_2d = np.array([float(gravity_world @ tangent_u), float(gravity_world @ tangent_v)], dtype=np.float32)
    local_xyz, faces, _radial_weight, _state = build_local_shape(
        spec,
        frame_index,
        frame_count,
        gravity_direction_2d,
        radial_segments=7,
        angular_segments=24,
    )
    local_points = local_xyz[:, [0, 2]].astype(np.float32)
    heights = np.maximum(local_xyz[:, 1], 0.0).astype(np.float32)
    vertices = anchor + local_points[:, 0, None] * tangent_u + local_points[:, 1, None] * tangent_v
    vertices = vertices + heights[:, None] * normal
    colors = np.tile(np.asarray(spec.color_rgb, dtype=np.uint8), (len(vertices), 1))
    return vertices.astype(np.float32), faces.astype(np.int32), colors


def load_simulation(body_part_root: Path, body_part: str) -> BodyPartSimulation:
    metadata = select_metadata(body_part_root, body_part)
    body_vertices, body_faces, body_colors = read_colored_ply(HSR_MESH_ROOT / f"{metadata['scan_id']}_closed_textured_mesh.ply")
    lesions = list(metadata["lesions"])
    return BodyPartSimulation(
        body_part=body_part,
        sample_id=str(metadata["sample_id"]),
        scan_id=str(metadata["scan_id"]),
        frame_count=int(metadata.get("simulation_frame_count", 100)),
        body_vertices=body_vertices.astype(np.float32),
        body_faces=body_faces.astype(np.int32),
        body_colors=body_colors.astype(np.uint8),
        lesions=lesions,
        specs=[spec_from_metadata(lesion) for lesion in lesions],
    )


def combine_lesion_frame(simulation: BodyPartSimulation, frame_index: int) -> tuple[trimesh.Trimesh, np.ndarray]:
    vertex_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    vertex_offset = 0
    for lesion, spec in zip(simulation.lesions, simulation.specs):
        vertices, faces, colors = lesion_vertices_for_frame(lesion, spec, frame_index, simulation.frame_count)
        vertex_parts.append(vertices)
        face_parts.append(faces + vertex_offset)
        color_parts.append(colors)
        vertex_offset += len(vertices)

    vertices = np.vstack(vertex_parts).astype(np.float32)
    faces = np.vstack(face_parts).astype(np.int32)
    colors = np.vstack(color_parts).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba(colors), process=False)
    return mesh, vertices


def body_mesh(simulation: BodyPartSimulation) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=simulation.body_vertices,
        faces=simulation.body_faces,
        vertex_colors=rgba(simulation.body_colors),
        process=False,
    )


def camera_for_body_part(body_part: str, body_vertices: np.ndarray, lesion_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    body_min = body_vertices.min(axis=0)
    body_max = body_vertices.max(axis=0)
    lesion_min = lesion_vertices.min(axis=0)
    lesion_max = lesion_vertices.max(axis=0)
    target = (lesion_min + lesion_max) / 2.0
    aspect = PANEL_WIDTH / PANEL_HEIGHT
    lesion_width = float(max(lesion_max[0] - lesion_min[0], 0.05))
    lesion_height = float(max(lesion_max[2] - lesion_min[2], 0.05))

    if body_part == "face":
        yfov_deg = 32.0
        frame_height = max(0.34, lesion_height * 1.65, lesion_width / aspect * 1.65)
        target[2] = max(target[2], body_min[2] + 0.78 * (body_max[2] - body_min[2]))
        y_direction = 1.0
    elif body_part == "back":
        yfov_deg = 38.0
        frame_height = max(0.82, lesion_height * 1.35, lesion_width / aspect * 1.35)
        y_direction = -1.0
    else:
        yfov_deg = 40.0
        frame_height = max(0.78, lesion_height * 1.32, lesion_width / aspect * 1.32)
        y_direction = 1.0

    distance = 0.5 * frame_height / math.tan(math.radians(yfov_deg) / 2.0)
    eye = np.array([target[0], target[1] + y_direction * distance, target[2] + 0.02], dtype=np.float64)
    return eye, target.astype(np.float64), yfov_deg


def render_body_part_stack_gif(
    body_part_root: Path,
    output_path: Path,
    body_parts: tuple[str, ...],
    gif_frames: int,
    fps: int,
) -> None:
    simulations = [load_simulation(body_part_root, body_part) for body_part in body_parts]
    sample_indices = {
        simulation.body_part: np.unique(np.linspace(0, simulation.frame_count - 1, gif_frames, dtype=np.int32))
        for simulation in simulations
    }
    rendered_rows: dict[str, list[np.ndarray]] = {simulation.body_part: [] for simulation in simulations}
    rendered_depths: dict[str, list[np.ndarray]] = {simulation.body_part: [] for simulation in simulations}

    renderer = pyrender.OffscreenRenderer(viewport_width=PANEL_WIDTH, viewport_height=PANEL_HEIGHT)
    try:
        for simulation in simulations:
            render_body = pyrender.Mesh.from_trimesh(body_mesh(simulation), smooth=True)
            _final_mesh, final_vertices = combine_lesion_frame(simulation, simulation.frame_count - 1)
            eye, target, yfov_deg = camera_for_body_part(simulation.body_part, simulation.body_vertices, final_vertices)
            camera_pose = look_at_camera_to_world(eye, target, np.array([0.0, 0.0, 1.0], dtype=np.float64))

            for frame_index in sample_indices[simulation.body_part]:
                lesion_mesh, _vertices = combine_lesion_frame(simulation, int(frame_index))
                scene = pyrender.Scene(bg_color=[244, 246, 249, 255], ambient_light=[0.78, 0.78, 0.78])
                scene.add(render_body)
                scene.add(pyrender.Mesh.from_trimesh(lesion_mesh, smooth=True))
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
                scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(yfov_deg), znear=0.01, zfar=8.0), pose=camera_pose)
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                rendered_rows[simulation.body_part].append(color[:, :, :3])
                rendered_depths[simulation.body_part].append(depth)
    finally:
        renderer.delete()

    depth_ranges: dict[str, tuple[float, float]] = {}
    for body_part in body_parts:
        valid_depths = np.concatenate([depth[depth > 0.0] for depth in rendered_depths[body_part]])
        depth_ranges[body_part] = (
            float(np.percentile(valid_depths, 1.0)),
            float(np.percentile(valid_depths, 99.0)),
        )

    row_width = PANEL_WIDTH * 2 + PANEL_GAP
    row_gap = np.full((ROW_GAP, row_width, 3), BACKGROUND, dtype=np.uint8)
    col_gap = np.full((PANEL_HEIGHT, PANEL_GAP, 3), BACKGROUND, dtype=np.uint8)
    header = np.full((HEADER_HEIGHT, row_width, 3), BACKGROUND, dtype=np.uint8)
    labels = {"back": "Back physics", "face": "Face physics", "front": "Front physics"}
    frames = []
    for sample_offset in range(len(next(iter(sample_indices.values())))):
        rows = []
        for body_part in body_parts:
            depth_min, depth_max = depth_ranges[body_part]
            color = add_view_label(rendered_rows[body_part][sample_offset], labels.get(body_part, body_part.title()))
            depth = depth_to_rainbow(rendered_depths[body_part][sample_offset], depth_min, depth_max)
            rows.append(np.concatenate([color, col_gap, depth], axis=1))
        stacked = np.concatenate([header, rows[0], row_gap, rows[1], row_gap, rows[2]], axis=0)
        frames.append(stacked)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1 / fps, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-part-root", type=Path, default=DEFAULT_BODY_PART_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--body-parts", nargs="+", default=list(DEFAULT_BODY_PARTS), choices=["back", "face", "front"])
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    body_parts = tuple(args.body_parts)
    if len(body_parts) != 3:
        raise ValueError("README physics preview expects exactly three body parts")
    render_body_part_stack_gif(args.body_part_root, args.output, body_parts, args.gif_frames, args.fps)
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

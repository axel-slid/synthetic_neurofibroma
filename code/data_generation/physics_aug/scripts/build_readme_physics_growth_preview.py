#!/usr/bin/env python3
"""Build the fixed-camera opaque physics-growth GIF used by the README."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
from matplotlib import colormaps
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NPZ = ROOT / "data" / "synthetic" / "multiple_lesion_physics" / "data" / "lesion_frame_vertices.npz"
DEFAULT_METRICS = ROOT / "data" / "synthetic" / "multiple_lesion_physics" / "data" / "frame_metrics.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "multiple_lesion_physics_growth_progression.gif"


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


def combine_lesion_frame(
    lesion_vertices: np.ndarray,
    lesion_faces: np.ndarray,
    lesion_colors: np.ndarray,
    frame_index: int,
) -> trimesh.Trimesh:
    lesion_count, _frame_count, vertex_count, _xyz = lesion_vertices.shape
    vertices = lesion_vertices[:, frame_index].reshape(lesion_count * vertex_count, 3)
    colors = lesion_colors.reshape(lesion_count * vertex_count, 3)
    faces = np.concatenate(
        [lesion_faces + lesion_index * vertex_count for lesion_index in range(lesion_count)],
        axis=0,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba(colors), process=False)


def depth_to_rainbow(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    valid = depth > 0.0
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = np.clip((depth[valid] - depth_min) / max(depth_max - depth_min, 1e-6), 0.0, 1.0)
    # Invert so nearer anatomy is warm/red and farther anatomy is cool/violet.
    mapped = colormaps["rainbow"](1.0 - normalized)[:, :, :3]
    depth_rgb = np.full((*depth.shape, 3), [244, 246, 249], dtype=np.uint8)
    depth_rgb[valid] = np.clip(np.rint(mapped[valid] * 255.0), 0, 255).astype(np.uint8)
    return depth_rgb


def total_final_volume_ml(lesion_vertices: np.ndarray, lesion_faces: np.ndarray) -> float:
    total_volume_m3 = 0.0
    for vertices in lesion_vertices[:, -1]:
        mesh = trimesh.Trimesh(vertices=vertices, faces=lesion_faces, process=False)
        total_volume_m3 += abs(float(mesh.volume))
    return total_volume_m3 * 1_000_000.0


def frame_growth_fractions(metrics_path: Path) -> dict[int, float]:
    growth_by_frame: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row["frame_index"])
            growth_by_frame[frame_index] = growth_by_frame.get(frame_index, 0.0) + float(row["growth"])
    max_growth = max(growth_by_frame.values())
    return {frame_index: growth / max_growth for frame_index, growth in growth_by_frame.items()}


def add_volume_label(frame: np.ndarray, volume_ml: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text = f"Total lesion volume: {volume_ml:,.0f} mL"
    padding_x = 14
    padding_y = 9
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x1 = image.width - text_w - padding_x * 2 - 16
    y1 = 14
    x2 = image.width - 16
    y2 = y1 + text_h + padding_y * 2
    draw.rounded_rectangle((x1, y1, x2, y2), radius=5, fill=(255, 255, 255), outline=(35, 35, 35))
    draw.text((x1 + padding_x, y1 + padding_y), text, fill=(20, 20, 20), font=font)
    return np.asarray(image)


def render_fixed_camera_growth_gif(
    npz_path: Path,
    metrics_path: Path,
    output_path: Path,
    gif_frames: int,
    fps: int,
) -> None:
    payload = np.load(npz_path)
    body_vertices = payload["body_plot_vertices"].astype(np.float32)
    body_faces = payload["body_plot_faces"].astype(np.int32)
    body_colors = payload["body_plot_colors"].astype(np.uint8)
    lesion_vertices = payload["lesion_vertices"].astype(np.float32)
    lesion_faces = payload["lesion_faces"].astype(np.int32)
    lesion_colors = payload["lesion_colors"].astype(np.uint8)
    final_volume_ml = total_final_volume_ml(lesion_vertices, lesion_faces)
    growth_fractions = frame_growth_fractions(metrics_path)

    body_mesh = trimesh.Trimesh(
        vertices=body_vertices,
        faces=body_faces,
        vertex_colors=rgba(body_colors),
        process=False,
    )
    render_body = pyrender.Mesh.from_trimesh(body_mesh, smooth=True)

    final_lesion_vertices = lesion_vertices[:, -1].reshape(-1, 3)
    lesion_center = np.median(final_lesion_vertices, axis=0)
    target = np.array([lesion_center[0], lesion_center[1], lesion_center[2] + 0.02], dtype=np.float64)
    eye = np.array([target[0], target[1] - 1.25, target[2] + 0.02], dtype=np.float64)
    camera_pose = look_at_camera_to_world(eye, target, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    frame_count = lesion_vertices.shape[1]
    sample_indices = np.unique(np.linspace(0, frame_count - 1, gif_frames, dtype=np.int32))
    renderer = pyrender.OffscreenRenderer(viewport_width=720, viewport_height=520)
    color_frames = []
    depth_frames = []
    try:
        for frame_index in sample_indices:
            scene = pyrender.Scene(bg_color=[244, 246, 249, 255], ambient_light=[0.78, 0.78, 0.78])
            scene.add(render_body)
            lesion_mesh = combine_lesion_frame(lesion_vertices, lesion_faces, lesion_colors, int(frame_index))
            scene.add(pyrender.Mesh.from_trimesh(lesion_mesh, smooth=True))
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
            scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(36.0), znear=0.01, zfar=8.0), pose=camera_pose)
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color_frames.append(color[:, :, :3])
            depth_frames.append(depth)
    finally:
        renderer.delete()

    valid_depths = np.concatenate([depth[depth > 0.0] for depth in depth_frames])
    depth_min = float(np.percentile(valid_depths, 1.0))
    depth_max = float(np.percentile(valid_depths, 99.0))
    gap = np.full((color_frames[0].shape[0], 8, 3), 244, dtype=np.uint8)
    frames = [
        add_volume_label(
            np.concatenate([color, gap, depth_to_rainbow(depth, depth_min, depth_max)], axis=1),
            final_volume_ml * growth_fractions[int(frame_index)],
        )
        for frame_index, color, depth in zip(sample_indices, color_frames, depth_frames)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1 / fps, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    render_fixed_camera_growth_gif(args.npz, args.metrics, args.output, args.gif_frames, args.fps)
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

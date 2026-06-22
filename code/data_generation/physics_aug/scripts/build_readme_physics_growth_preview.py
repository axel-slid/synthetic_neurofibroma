#!/usr/bin/env python3
"""Build the fixed-camera opaque physics-growth GIF used by the README."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NPZ = ROOT / "data" / "synthetic" / "multiple_lesion_physics" / "data" / "lesion_frame_vertices.npz"
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


def render_fixed_camera_growth_gif(npz_path: Path, output_path: Path, gif_frames: int, fps: int) -> None:
    payload = np.load(npz_path)
    body_vertices = payload["body_plot_vertices"].astype(np.float32)
    body_faces = payload["body_plot_faces"].astype(np.int32)
    body_colors = payload["body_plot_colors"].astype(np.uint8)
    lesion_vertices = payload["lesion_vertices"].astype(np.float32)
    lesion_faces = payload["lesion_faces"].astype(np.int32)
    lesion_colors = payload["lesion_colors"].astype(np.uint8)

    body_mesh = trimesh.Trimesh(
        vertices=body_vertices,
        faces=body_faces,
        vertex_colors=rgba(body_colors),
        process=False,
    )
    render_body = pyrender.Mesh.from_trimesh(body_mesh, smooth=True)

    xyz_min = body_vertices.min(axis=0)
    xyz_max = body_vertices.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    target = np.array([center[0], center[1], center[2]], dtype=np.float64)
    eye = np.array([center[0], xyz_min[1] - 2.75, center[2] + 0.04], dtype=np.float64)
    camera_pose = look_at_camera_to_world(eye, target, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    frame_count = lesion_vertices.shape[1]
    sample_indices = np.unique(np.linspace(0, frame_count - 1, gif_frames, dtype=np.int32))
    renderer = pyrender.OffscreenRenderer(viewport_width=900, viewport_height=650)
    frames = []
    try:
        for frame_index in sample_indices:
            scene = pyrender.Scene(bg_color=[244, 246, 249, 255], ambient_light=[0.78, 0.78, 0.78])
            scene.add(render_body)
            lesion_mesh = combine_lesion_frame(lesion_vertices, lesion_faces, lesion_colors, int(frame_index))
            scene.add(pyrender.Mesh.from_trimesh(lesion_mesh, smooth=True))
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
            scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(34.0), znear=0.01, zfar=8.0), pose=camera_pose)
            color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            frames.append(color[:, :, :3])
    finally:
        renderer.delete()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1 / fps, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    render_fixed_camera_growth_gif(args.npz, args.output, args.gif_frames, args.fps)
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

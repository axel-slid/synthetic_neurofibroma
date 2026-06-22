#!/usr/bin/env python3
"""Build the Fitzpatrick original-image plus rotating 3D surface GIF for README."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[4]
FITZPATRICK_PLOTLY_ROOT = (
    ROOT
    / "data"
    / "skin"
    / "fitzpatrick"
    / "visualizations"
    / "depth_pro"
    / "plotly"
    / "fitzpatrick_neurofibromatosis"
)
DEFAULT_SAMPLE_ID = "fitz_nf_0011"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "fitzpatrick_depthpro_surface_rotation.gif"
PANEL_WIDTH = 430
PANEL_HEIGHT = 360
PANEL_GAP = 8
BACKGROUND = (244, 246, 249)


def rgba(rgb: np.ndarray) -> np.ndarray:
    alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=1)


def centered_surface_mesh(surface_path: Path, angle_rad: float, depth_scale: float) -> trimesh.Trimesh:
    payload = np.load(surface_path)
    vertices = payload["vertices"].astype(np.float32)
    faces = payload["triangles"].astype(np.int32)
    colors = payload["colors"].astype(np.uint8)

    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices[:, 1] *= -1.0
    vertices[:, 2] *= depth_scale
    faces = np.vstack([faces, faces[:, ::-1]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba(colors), process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_rad, [0.0, 1.0, 0.0]))
    return mesh


def render_surface(surface_path: Path, angle_rad: float, depth_scale: float) -> np.ndarray:
    mesh = centered_surface_mesh(surface_path, angle_rad, depth_scale)
    scene = pyrender.Scene(bg_color=[*BACKGROUND, 255], ambient_light=[0.86, 0.86, 0.86])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[:3, 3] = [0.0, 0.0, 3.2]
    scene.add(pyrender.PerspectiveCamera(yfov=math.radians(36.0), znear=0.01, zfar=10.0), pose=camera_pose)

    light_pose = np.eye(4, dtype=np.float64)
    light_pose[:3, 3] = [0.0, -1.0, 3.0]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.4), pose=light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=PANEL_WIDTH, viewport_height=PANEL_HEIGHT)
    try:
        color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return color[:, :, :3]


def original_image_panel(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.contain(image, (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    panel.paste(image, ((PANEL_WIDTH - image.width) // 2, (PANEL_HEIGHT - image.height) // 2))
    return panel


def build_frame(original_panel: Image.Image, surface_rgb: np.ndarray) -> np.ndarray:
    frame = Image.new("RGB", (PANEL_WIDTH * 2 + PANEL_GAP, PANEL_HEIGHT), BACKGROUND)
    frame.paste(original_panel, (0, 0))
    frame.paste(Image.fromarray(surface_rgb).convert("RGB"), (PANEL_WIDTH + PANEL_GAP, 0))
    return np.asarray(frame)


def build_gif(sample_id: str, output_path: Path, frames: int, fps: int, depth_scale: float) -> None:
    image_path = FITZPATRICK_PLOTLY_ROOT / "images" / f"{sample_id}.jpg"
    surface_path = FITZPATRICK_PLOTLY_ROOT / "surfaces" / f"{sample_id}_depthpro_surface_64.npz"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing Fitzpatrick image: {image_path}")
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing Fitzpatrick surface: {surface_path}")

    original_panel = original_image_panel(image_path)
    images = []
    for frame_index in range(frames):
        angle = 2.0 * math.pi * frame_index / frames
        surface_rgb = render_surface(surface_path, angle, depth_scale)
        images.append(build_frame(original_panel, surface_rgb))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, duration=1 / fps, loop=0)


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--depth-scale", type=float, default=0.85)
    args = parser.parse_args()

    build_gif(args.sample_id, args.output, args.frames, args.fps, args.depth_scale)
    print(f"Wrote {root_relative(args.output)}")


if __name__ == "__main__":
    main()

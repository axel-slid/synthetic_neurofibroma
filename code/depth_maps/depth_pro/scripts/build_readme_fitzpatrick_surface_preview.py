#!/usr/bin/env python3
"""Build the stacked Fitzpatrick image, depth map, and 3D surface GIF for README."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
from matplotlib import colormaps
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
DEFAULT_SAMPLE_IDS = ("fitz_nf_0011", "fitz_nf_0037", "fitz_nf_0064")
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "fitzpatrick_depthpro_surface_rotation.gif"
OUTPUT_WIDTH = 868
PANEL_WIDTH = (OUTPUT_WIDTH - 2 * 8) // 3
PANEL_HEIGHT = 250
PANEL_GAP = 8
ROW_GAP = 6
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


def depth_panel(surface_path: Path) -> Image.Image:
    payload = np.load(surface_path)
    vertices = payload["vertices"].astype(np.float32)
    side = int(round(math.sqrt(len(vertices))))
    if side * side != len(vertices):
        raise ValueError(f"Expected square surface vertex count for {surface_path}, got {len(vertices)}")

    z_grid = vertices[:, 2].reshape(side, side)
    depth_min = float(np.nanmin(z_grid))
    depth_max = float(np.nanmax(z_grid))
    normalized_depth = np.clip((z_grid - depth_min) / max(depth_max - depth_min, 1e-8), 0.0, 1.0)
    depth_rgb = np.clip(colormaps["magma"](normalized_depth)[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(depth_rgb).resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.BICUBIC).convert("RGB")
    return image


def build_row(original_panel: Image.Image, depth_map_panel: Image.Image, surface_rgb: np.ndarray) -> Image.Image:
    row = Image.new("RGB", (OUTPUT_WIDTH, PANEL_HEIGHT), BACKGROUND)
    row.paste(original_panel, (0, 0))
    row.paste(depth_map_panel, (PANEL_WIDTH + PANEL_GAP, 0))
    row.paste(Image.fromarray(surface_rgb).convert("RGB"), ((PANEL_WIDTH + PANEL_GAP) * 2, 0))
    return row


def build_frame(rows: list[Image.Image]) -> np.ndarray:
    height = len(rows) * PANEL_HEIGHT + max(0, len(rows) - 1) * ROW_GAP
    frame = Image.new("RGB", (OUTPUT_WIDTH, height), BACKGROUND)
    for row_index, row in enumerate(rows):
        frame.paste(row, (0, row_index * (PANEL_HEIGHT + ROW_GAP)))
    return np.asarray(frame)


def build_gif(
    sample_ids: tuple[str, ...],
    output_path: Path,
    frames: int,
    fps: int,
    depth_scale: float,
    front_yaw_degrees: float,
) -> None:
    if not sample_ids:
        raise ValueError("At least one Fitzpatrick sample ID is required")

    samples: list[tuple[Image.Image, Image.Image, Path]] = []
    for sample_id in sample_ids:
        image_path = FITZPATRICK_PLOTLY_ROOT / "images" / f"{sample_id}.jpg"
        surface_path = FITZPATRICK_PLOTLY_ROOT / "surfaces" / f"{sample_id}_depthpro_surface_64.npz"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing Fitzpatrick image: {image_path}")
        if not surface_path.exists():
            raise FileNotFoundError(f"Missing Fitzpatrick surface: {surface_path}")
        samples.append((original_image_panel(image_path), depth_panel(surface_path), surface_path))

    images = []
    yaw_rad = math.radians(front_yaw_degrees)
    for frame_index in range(frames):
        angle = yaw_rad * math.sin(2.0 * math.pi * frame_index / frames)
        rows = []
        for original_panel, depth_map_panel, surface_path in samples:
            surface_rgb = render_surface(surface_path, angle, depth_scale)
            rows.append(build_row(original_panel, depth_map_panel, surface_rgb))
        images.append(build_frame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, duration=1 / fps, loop=0)


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default=None, help="Render one Fitzpatrick sample ID.")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Render stacked Fitzpatrick sample IDs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--depth-scale", type=float, default=0.85)
    parser.add_argument("--front-yaw-degrees", type=float, default=14.0)
    args = parser.parse_args()

    if args.sample_ids:
        sample_ids = tuple(args.sample_ids)
    elif args.sample_id:
        sample_ids = (args.sample_id,)
    else:
        sample_ids = DEFAULT_SAMPLE_IDS

    build_gif(sample_ids, args.output, args.frames, args.fps, args.depth_scale, args.front_yaw_degrees)
    print(f"Wrote {root_relative(args.output)}")


if __name__ == "__main__":
    main()

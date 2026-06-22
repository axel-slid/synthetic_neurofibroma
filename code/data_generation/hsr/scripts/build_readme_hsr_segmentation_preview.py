#!/usr/bin/env python3
"""Build the README HSR body-part segmentation GIF with a fixed color key."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEGMENTATION_ROOT = ROOT / "data" / "hsr" / "body_part_segmentation" / "manual" / "data"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "hsr_body_part_segmentation_overlay_combined.gif"
SCAN_IDS = ("HSR0018-Body-070", "HSR0152-Body-090")
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
PANEL_WIDTH = 430
PANEL_HEIGHT = 560
PANEL_GAP = 8
LEGEND_HEIGHT = 68
BACKGROUND = (244, 246, 249)
DEFAULT_MASK_ALPHA = 0.48


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def rgba(rgb: np.ndarray) -> np.ndarray:
    alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=1)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


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


def rotation_about_z(angle_rad: float, center: np.ndarray) -> np.ndarray:
    transform = trimesh.transformations.translation_matrix(center)
    transform = transform @ trimesh.transformations.rotation_matrix(angle_rad, [0.0, 0.0, 1.0])
    transform = transform @ trimesh.transformations.translation_matrix(-center)
    return transform


def load_label_mesh(npz_path: Path, mask_alpha: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
    payload = np.load(npz_path)
    label_names = [str(value) for value in payload["label_names"].tolist()]
    label_rgb = np.asarray([hex_to_rgb(LABEL_COLORS[name]) for name in label_names], dtype=np.uint8)
    vertices = payload["vertices"].astype(np.float32)
    faces = payload["triangles"].astype(np.int32)
    base_colors = payload["vertex_colors"].astype(np.float32)
    labels = payload["vertex_labels"].astype(np.int32)
    overlay_colors = label_rgb[labels].astype(np.float32)
    colors = np.clip((1.0 - mask_alpha) * base_colors + mask_alpha * overlay_colors, 0, 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba(colors), process=False)
    return mesh, vertices


def render_scan_panel(mesh: trimesh.Trimesh, vertices: np.ndarray, angle_rad: float) -> np.ndarray:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    rotated = mesh.copy()
    rotated.apply_transform(rotation_about_z(angle_rad, center.astype(np.float64)))

    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    xy_radius = float(np.quantile(np.linalg.norm(vertices[:, :2] - center[:2], axis=1), 0.995))
    aspect = PANEL_WIDTH / PANEL_HEIGHT
    frame_height = max(height * 1.10, (2.0 * xy_radius) / aspect * 1.10)
    yfov = math.radians(34.0)
    distance = 0.5 * frame_height / math.tan(yfov / 2.0)
    target = np.array([center[0], center[1], center[2] + 0.01 * height], dtype=np.float64)
    eye = np.array([center[0], center[1] - distance, target[2]], dtype=np.float64)
    camera_pose = look_at_camera_to_world(eye, target, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    scene = pyrender.Scene(bg_color=[*BACKGROUND, 255], ambient_light=[0.92, 0.92, 0.92])
    scene.add(pyrender.Mesh.from_trimesh(rotated, smooth=True))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.8), pose=camera_pose)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, znear=0.01, zfar=8.0), pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=PANEL_WIDTH, viewport_height=PANEL_HEIGHT)
    try:
        color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return color[:, :, :3]


def legend_image(width: int) -> Image.Image:
    image = Image.new("RGB", (width, LEGEND_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    label_font = font(12)
    names = list(LABEL_COLORS)
    start_x = 18
    row_y = (12, 38)
    col_w = 104
    for idx, name in enumerate(names):
        row = idx // 4
        col = idx % 4
        x = start_x + col * col_w
        y = row_y[row]
        draw.rounded_rectangle((x, y + 2, x + 16, y + 18), radius=2, fill=hex_to_rgb(LABEL_COLORS[name]))
        draw.text((x + 24, y + 1), name, fill=(22, 28, 36), font=label_font)
    return image


def fixed_palette_image(frames: list[Image.Image]) -> Image.Image:
    palette_colors = [hex_to_rgb(color) for color in LABEL_COLORS.values()]
    palette_colors.extend([BACKGROUND, (22, 28, 36), (255, 255, 255), (0, 0, 0)])

    if frames:
        source = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
        for frame_index, frame in enumerate(frames):
            source.paste(frame.convert("RGB"), (0, frame_index * frames[0].height))
        adaptive_count = 256 - len(palette_colors)
        adaptive = source.quantize(colors=adaptive_count, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        adaptive_palette = adaptive.getpalette()[: adaptive_count * 3]
        adaptive_colors = [
            tuple(adaptive_palette[index : index + 3])
            for index in range(0, len(adaptive_palette), 3)
        ]
        seen = set(palette_colors)
        for color in adaptive_colors:
            if color not in seen:
                palette_colors.append(color)
                seen.add(color)

    palette_colors = palette_colors[:256]
    palette_colors.extend([BACKGROUND] * (256 - len(palette_colors)))

    palette = Image.new("P", (16, 16))
    palette.putpalette([channel for color in palette_colors for channel in color])
    return palette


def stamp_fixed_legend_swatches(frame: Image.Image) -> None:
    draw = ImageDraw.Draw(frame)
    start_x = 18
    row_y = (12, 38)
    col_w = 104
    for idx, _name in enumerate(LABEL_COLORS):
        row = idx // 4
        col = idx % 4
        x = start_x + col * col_w
        y = row_y[row]
        draw.rounded_rectangle((x, y + 2, x + 16, y + 18), radius=2, fill=idx)


def save_fixed_palette_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("At least one frame is required")
    palette = fixed_palette_image(frames)
    quantized_frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    for frame in quantized_frames:
        stamp_fixed_legend_swatches(frame)
    duration_ms = max(1, int(round(1000 / fps)))
    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def build_gif(segmentation_root: Path, output_path: Path, frames: int, fps: int, mask_alpha: float) -> None:
    meshes = []
    for scan_id in SCAN_IDS:
        mesh, vertices = load_label_mesh(segmentation_root / f"{scan_id}_body_part_segmentation.npz", mask_alpha)
        meshes.append((mesh, vertices))

    width = PANEL_WIDTH * len(meshes) + PANEL_GAP * (len(meshes) - 1)
    legend = legend_image(width)
    images = []
    for frame_index in range(frames):
        angle = 2.0 * math.pi * frame_index / frames
        canvas = Image.new("RGB", (width, LEGEND_HEIGHT + PANEL_HEIGHT), BACKGROUND)
        canvas.paste(legend, (0, 0))
        x = 0
        for mesh, vertices in meshes:
            panel = Image.fromarray(render_scan_panel(mesh, vertices, angle)).convert("RGB")
            canvas.paste(panel, (x, LEGEND_HEIGHT))
            x += PANEL_WIDTH + PANEL_GAP
        images.append(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fixed_palette_gif(images, output_path, fps)


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--mask-alpha", type=float, default=DEFAULT_MASK_ALPHA)
    args = parser.parse_args()

    build_gif(args.segmentation_root, args.output, args.frames, args.fps, args.mask_alpha)
    print(f"Wrote {root_relative(args.output)}")


if __name__ == "__main__":
    main()

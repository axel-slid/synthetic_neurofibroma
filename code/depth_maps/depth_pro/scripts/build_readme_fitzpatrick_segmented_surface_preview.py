#!/usr/bin/env python3
"""Build a segmented Fitzpatrick image/depth/surface blinking GIF for README."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pyrender
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import measure, morphology
import trimesh

import build_readme_fitzpatrick_surface_preview as base

DEFAULT_OUTPUT = base.ROOT / "docs" / "assets" / "fitzpatrick_depthpro_segmented_surface_blink.gif"
DEFAULT_NOTEBOOK = base.ROOT / "data" / "skin" / "fitzpatrick" / "visualizations" / "depth_pro" / "fitzpatrick_depth_pro_segmented_surface_blink.ipynb"
DEFAULT_MASK_ROOT = (
    base.ROOT
    / "data"
    / "skin"
    / "fitzpatrick"
    / "visualizations"
    / "depth_pro"
    / "segmentation_masks"
    / "fitzpatrick_neurofibromatosis"
)
DEFAULT_MANIFEST = DEFAULT_MASK_ROOT / "lesion_mask_manifest.csv"
DEFAULT_MANUAL_OUTPUT = base.ROOT / "docs" / "assets" / "fitzpatrick_depthpro_manual_segmented_surface_blink.gif"
DEFAULT_MANUAL_NOTEBOOK = (
    base.ROOT
    / "data"
    / "skin"
    / "fitzpatrick"
    / "visualizations"
    / "depth_pro"
    / "fitzpatrick_depth_pro_manual_segmented_surface_blink.ipynb"
)
DEFAULT_MANUAL_ROOT = (
    base.ROOT
    / "data"
    / "skin"
    / "fitzpatrick"
    / "visualizations"
    / "depth_pro"
    / "manual_polygon_masks"
    / "fitzpatrick_3x3_samples_from_upload"
)
DEFAULT_MANUAL_POLYGON_MASK_ROOT = DEFAULT_MANUAL_ROOT / "polygon_masks"
DEFAULT_MANUAL_RESAMPLED_MASK_ROOT = DEFAULT_MANUAL_ROOT / "resampled_masks_reference_blink"
DEFAULT_MANUAL_MANIFEST = DEFAULT_MANUAL_ROOT / "manual_segmented_surface_blink_manifest.csv"
SEGMENT_COLOR = np.asarray([0, 255, 210], dtype=np.float32)


def lesion_mask_from_depth_and_texture(surface_path: Path, image_path: Path, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces, colors = base.resampled_surface(surface_path, image_path, side)
    z_grid = vertices[:, 2].reshape(side, side).astype(np.float32)
    rgb_grid = colors.reshape(side, side, 3).astype(np.uint8)

    z_smooth = ndi.gaussian_filter(z_grid, 1.0)
    baseline = ndi.gaussian_filter(z_smooth, side * 0.075)
    relief = ndi.gaussian_filter(z_smooth - baseline, 1.0)

    lab = cv2.cvtColor(rgb_grid, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness, green_red, blue_yellow = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    local_lightness = ndi.gaussian_filter(lightness, side * 0.04)
    darker_than_neighborhood = local_lightness - lightness
    color_anomaly = (
        (darker_than_neighborhood > np.percentile(darker_than_neighborhood, 72))
        | (green_red > np.percentile(green_red, 72))
        | (blue_yellow > np.percentile(blue_yellow, 70))
    )

    high_relief = relief > np.percentile(relief, 88)
    very_high_relief = relief > np.percentile(relief, 94)
    foreground_height = z_smooth > np.percentile(z_smooth, 62)
    raw_mask = (very_high_relief & foreground_height) | (high_relief & color_anomaly)

    raw_mask = morphology.binary_closing(raw_mask, morphology.disk(3))
    raw_mask = ndi.binary_fill_holes(raw_mask)
    raw_mask = morphology.binary_opening(raw_mask, morphology.disk(1))

    labels = measure.label(raw_mask)
    mask = np.zeros_like(raw_mask, dtype=bool)
    total_pixels = side * side
    border_margin = max(3, side // 50)
    for region in measure.regionprops(labels, intensity_image=relief):
        min_row, min_col, max_row, max_col = region.bbox
        touches_edge = (
            min_row <= border_margin
            or min_col <= border_margin
            or max_row >= side - border_margin
            or max_col >= side - border_margin
        )
        if touches_edge:
            continue
        if region.area < max(10, int(total_pixels * 0.00045)):
            continue
        if region.area > total_pixels * 0.30:
            continue
        if (max_row - min_row) > side * 0.75 or (max_col - min_col) > side * 0.75:
            continue
        if region.mean_intensity < np.percentile(relief, 64):
            continue
        mask[labels == region.label] = True

    mask = morphology.binary_dilation(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool), vertices, faces, colors


def save_mask(mask: np.ndarray, mask_path: Path) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)


def load_manual_polygon_mask(mask_root: Path, sample_id: str, image_path: Path, side: int) -> tuple[np.ndarray, np.ndarray, Path]:
    mask_path = mask_root / f"{sample_id}_polygon_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing manual polygon mask: {mask_path}")

    image_size = Image.open(image_path).size
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != image_size:
        mask_image = mask_image.resize(image_size, Image.Resampling.NEAREST)

    image_mask = np.asarray(mask_image, dtype=np.uint8) > 0
    surface_mask = np.asarray(mask_image.resize((side, side), Image.Resampling.NEAREST), dtype=np.uint8) > 0
    return image_mask, surface_mask, mask_path


def mask_for_original_panel(mask: np.ndarray, image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    contained = ImageOps.contain(image, (base.PANEL_WIDTH, base.PANEL_HEIGHT), Image.Resampling.LANCZOS)
    resized_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        contained.size,
        Image.Resampling.NEAREST,
    )
    panel_mask = Image.new("L", (base.PANEL_WIDTH, base.PANEL_HEIGHT), 0)
    panel_mask.paste(
        resized_mask,
        ((base.PANEL_WIDTH - contained.width) // 2, (base.PANEL_HEIGHT - contained.height) // 2),
    )
    return panel_mask


def mask_for_full_panel(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        (base.PANEL_WIDTH, base.PANEL_HEIGHT),
        Image.Resampling.NEAREST,
    )


def overlay_mask(panel: Image.Image, mask_panel: Image.Image, overlay_strength: float) -> Image.Image:
    overlay_strength = float(np.clip(overlay_strength, 0.0, 1.0))
    if overlay_strength <= 0.0:
        return panel
    panel_rgba = panel.convert("RGBA")
    fill = Image.new("RGBA", panel_rgba.size, tuple(int(v) for v in SEGMENT_COLOR) + (0,))
    fill_alpha = int(round(35 + 105 * overlay_strength))
    fill.putalpha(mask_panel.point(lambda value: fill_alpha if value > 0 else 0))
    panel_rgba = Image.alpha_composite(panel_rgba, fill)

    mask_arr = np.asarray(mask_panel) > 0
    border = mask_arr & ~ndi.binary_erosion(mask_arr, iterations=2)
    border_img = Image.fromarray((border.astype(np.uint8) * 255), mode="L")
    outline = Image.new("RGBA", panel_rgba.size, tuple(int(v) for v in SEGMENT_COLOR) + (255,))
    outline_alpha = int(round(85 + 170 * overlay_strength))
    outline.putalpha(border_img.point(lambda value: outline_alpha if value > 0 else 0))
    return Image.alpha_composite(panel_rgba, outline).convert("RGB")


def segmented_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    angle_rad: float,
    depth_scale: float,
    overlay_strength: float,
    overlay_mode: str,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh_vertices = vertices.astype(np.float32).copy()
    mesh_vertices = mesh_vertices - (mesh_vertices.min(axis=0) + mesh_vertices.max(axis=0)) / 2.0
    mesh_vertices[:, 1] *= -1.0
    mesh_vertices[:, 2] *= depth_scale

    mesh_colors = colors.astype(np.float32).copy()
    overlay_strength = float(np.clip(overlay_strength, 0.0, 1.0))
    if overlay_mode in {"pulse", "pulse_outline", "pulse_screen_outline"} and overlay_strength > 0.0:
        flat_mask = mask.reshape(-1)
        blend = 0.70 * overlay_strength
        mesh_colors[flat_mask] = np.clip((1.0 - blend) * mesh_colors[flat_mask] + blend * SEGMENT_COLOR, 0, 255)
        if overlay_mode == "pulse_outline":
            boundary = mask & ~ndi.binary_erosion(mask, iterations=2)
            boundary = morphology.binary_dilation(boundary, morphology.disk(1))
            flat_boundary = boundary.reshape(-1)
            boundary_blend = 0.80 + 0.15 * overlay_strength
            mesh_colors[flat_boundary] = np.clip(
                (1.0 - boundary_blend) * mesh_colors[flat_boundary] + boundary_blend * SEGMENT_COLOR,
                0,
                255,
            )
    elif overlay_mode == "subtle_boundary_outline" and overlay_strength > 0.0:
        flat_mask = mask.reshape(-1)
        blend = 0.30 * overlay_strength
        mesh_colors[flat_mask] = np.clip((1.0 - blend) * mesh_colors[flat_mask] + blend * SEGMENT_COLOR, 0, 255)
    elif overlay_mode == "boundary_only":
        boundary = mask & ~ndi.binary_erosion(mask, iterations=2)
        flat_boundary = boundary.reshape(-1)
        mesh_colors[flat_boundary] = SEGMENT_COLOR
    elif overlay_mode == "outline":
        boundary = mask & ~ndi.binary_erosion(mask, iterations=2)
        boundary = morphology.binary_dilation(boundary, morphology.disk(1))
        flat_boundary = boundary.reshape(-1)
        blend = 0.70 + 0.20 * overlay_strength
        mesh_colors[flat_boundary] = np.clip(
            (1.0 - blend) * mesh_colors[flat_boundary] + blend * SEGMENT_COLOR,
            0,
            255,
        )

    mesh_faces = np.vstack([faces, faces[:, ::-1]])
    mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        vertex_colors=base.rgba(mesh_colors.astype(np.uint8)),
        process=False,
    )
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_rad, [0.0, 1.0, 0.0]))
    return mesh, colors


def render_segmented_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    angle_rad: float,
    depth_scale: float,
    overlay_strength: float,
    overlay_mode: str,
    surface_outline_width: int,
) -> Image.Image:
    def render_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        scene = pyrender.Scene(bg_color=[*base.BACKGROUND, 255], ambient_light=[0.23, 0.23, 0.23])
        scene.add(base.glossy_vertex_mesh(mesh))

        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[:3, 3] = [0.0, 0.0, 3.2]
        scene.add(pyrender.PerspectiveCamera(yfov=math.radians(36.0), znear=0.01, zfar=10.0), pose=camera_pose)
        base.add_surface_lights(scene)

        renderer = pyrender.OffscreenRenderer(viewport_width=base.PANEL_WIDTH, viewport_height=base.PANEL_HEIGHT)
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        return color[:, :, :3], depth

    base_mesh, reference_colors = segmented_mesh(vertices, faces, colors, mask, angle_rad, depth_scale, 0.0, "none")
    base_rgb, base_depth = render_mesh(base_mesh)
    base_matched = base.match_rendered_color_statistics(base_rgb, base_depth, reference_colors)
    if overlay_mode == "none" or overlay_strength <= 0.02:
        return Image.fromarray(base_matched).convert("RGB")

    if overlay_mode == "subtle_boundary_outline":
        fill_mesh, _reference_colors = segmented_mesh(
            vertices,
            faces,
            colors,
            mask,
            angle_rad,
            depth_scale,
            overlay_strength,
            overlay_mode,
        )
        fill_rgb, fill_depth = render_mesh(fill_mesh)
        foreground = base.foreground_mask(base_depth) | base.foreground_mask(fill_depth)
        fill_delta = np.linalg.norm(fill_rgb.astype(np.int16) - base_rgb.astype(np.int16), axis=2)

        output = base_matched.copy()
        fill_changed = foreground & (fill_delta > 2.0)
        output[fill_changed] = fill_rgb[fill_changed]

        boundary_changed = fill_changed & ~ndi.binary_erosion(fill_changed, iterations=max(1, int(surface_outline_width)))
        output[boundary_changed] = SEGMENT_COLOR.astype(np.uint8)
        return Image.fromarray(output).convert("RGB")

    overlay_mesh, _reference_colors = segmented_mesh(
        vertices,
        faces,
        colors,
        mask,
        angle_rad,
        depth_scale,
        overlay_strength,
        overlay_mode,
    )
    overlay_rgb, overlay_depth = render_mesh(overlay_mesh)
    foreground = base.foreground_mask(base_depth) | base.foreground_mask(overlay_depth)
    color_delta = np.linalg.norm(overlay_rgb.astype(np.int16) - base_rgb.astype(np.int16), axis=2)
    changed = foreground & (color_delta > 4.0)
    changed = ndi.binary_dilation(changed, iterations=1)

    output = base_matched.copy()
    output[changed] = overlay_rgb[changed]
    if overlay_mode == "pulse_screen_outline" and surface_outline_width > 0:
        outline_width = max(1, int(surface_outline_width))
        inner = ndi.binary_erosion(changed, iterations=outline_width)
        outline = changed & ~inner
        output[outline] = SEGMENT_COLOR.astype(np.uint8)
    return Image.fromarray(output).convert("RGB")


def compose_frame(rows: list[tuple[Image.Image, Image.Image, Image.Image]]) -> np.ndarray:
    row_images = []
    for original_panel, depth_panel, surface_panel in rows:
        row = Image.new("RGB", (base.OUTPUT_WIDTH, base.PANEL_HEIGHT), base.BACKGROUND)
        row.paste(original_panel, (0, 0))
        row.paste(depth_panel, (base.PANEL_WIDTH + base.PANEL_GAP, 0))
        row.paste(surface_panel, ((base.PANEL_WIDTH + base.PANEL_GAP) * 2, 0))
        row_images.append(row)
    return base.build_frame(row_images)


def gif_frame_durations_ms(frame_count: int, fps: int) -> list[int]:
    if frame_count <= 0:
        return []
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    cumulative_centiseconds = [int(round(index * 100 / fps)) for index in range(frame_count + 1)]
    durations = []
    for start, stop in zip(cumulative_centiseconds[:-1], cumulative_centiseconds[1:], strict=True):
        durations.append(max(1, stop - start) * 10)
    return durations


def gif_duration_arg(frame_count: int, fps: int) -> int | list[int]:
    durations = gif_frame_durations_ms(frame_count, fps)
    if len(durations) == 1:
        return durations[0]
    return durations


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "image_path", "surface_path", "mask_path", "mask_pixels", "components", "method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gif_output_cell(gif_path: Path) -> nbf.NotebookNode:
    encoded = base64.b64encode(gif_path.read_bytes()).decode("ascii")
    return nbf.v4.new_code_cell(
        source="",
        execution_count=None,
        metadata={"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
        outputs=[
            nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/gif": encoded,
                    "text/plain": f"{base.root_relative(gif_path)}",
                },
                metadata={},
            )
        ],
    )


def plotly_output_cell(sample: dict[str, object]) -> nbf.NotebookNode:
    vertices = np.asarray(sample["vertices"], dtype=np.float32)
    faces = np.asarray(sample["faces"], dtype=np.int32)
    colors = np.asarray(sample["colors"], dtype=np.uint8).copy()
    mask = np.asarray(sample["mask"], dtype=bool).reshape(-1)
    colors[mask] = np.clip(0.30 * colors[mask].astype(np.float32) + 0.70 * SEGMENT_COLOR, 0, 255).astype(np.uint8)
    vertexcolor = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=-vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertexcolor,
                flatshading=False,
                lighting=dict(ambient=0.48, diffuse=0.72, specular=0.36, roughness=0.34, fresnel=0.08),
                lightposition=dict(x=-120, y=-160, z=260),
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        title=f"{sample['sample_id']} segmented Depth Pro surface",
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=0.0, y=-0.15, z=2.25)),
        ),
        width=760,
        height=620,
        margin=dict(l=0, r=0, t=48, b=0),
    )
    plotly_json = json.loads(pio.to_json(fig, validate=False))
    return nbf.v4.new_code_cell(
        source="",
        execution_count=None,
        metadata={"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
        outputs=[
            nbf.v4.new_output(
                output_type="display_data",
                data={
                    "application/vnd.plotly.v1+json": plotly_json,
                    "text/plain": f"{sample['sample_id']} segmented surface",
                },
                metadata={},
            )
        ],
    )


def write_notebook(path: Path, gif_path: Path, samples: list[dict[str, object]], manifest_rows: list[dict[str, object]]) -> None:
    mask_summary = "\n".join(
        f"- `{row['sample_id']}`: `{row['components']}` components, `{row['mask_pixels']}` mask pixels"
        for row in manifest_rows
    )
    manual_masks = all(str(row["method"]).startswith("manual_polygon_mask:") for row in manifest_rows)
    title = (
        "Fitzpatrick Depth Pro Manual Segmented Lesion Blink"
        if manual_masks
        else "Fitzpatrick Depth Pro Segmented Lesion Blink"
    )
    source_line = (
        "Manual polygon lesion masks are projected onto the Depth Pro surfaces."
        if manual_masks
        else "Generated preview GIF and interactive segmented 3D surfaces."
    )
    cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            f"{source_line} Code cells have empty sources; "
            "outputs were generated by the repository script.\n\n"
            f"{mask_summary}"
        ),
        gif_output_cell(gif_path),
    ]
    cells.extend(plotly_output_cell(sample) for sample in samples)
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def build_gif(
    sample_ids: tuple[str, ...],
    output_path: Path,
    mask_root: Path,
    manifest_path: Path,
    notebook_path: Path,
    manual_mask_root: Path | None,
    frames: int,
    fps: int,
    depth_scale: float,
    front_yaw_degrees: float,
    render_side: int,
    pulse_cycles: float,
    depth_overlay: bool,
    surface_overlay_mode: str,
    legacy_gif_timing: bool,
    blink_mode: str,
    surface_outline_width: int,
    overlay_min_strength: float,
    overlay_max_strength: float,
) -> None:
    samples = []
    manifest_rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        image_path = base.FITZPATRICK_PLOTLY_ROOT / "images" / f"{sample_id}.jpg"
        surface_path = base.FITZPATRICK_PLOTLY_ROOT / "surfaces" / f"{sample_id}_depthpro_surface_64.npz"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing Fitzpatrick image: {image_path}")
        if not surface_path.exists():
            raise FileNotFoundError(f"Missing Fitzpatrick surface: {surface_path}")

        if manual_mask_root is None:
            mask, vertices, faces, colors = lesion_mask_from_depth_and_texture(surface_path, image_path, render_side)
            image_mask = mask
            method = "depth_pro_relief_plus_lab_color_connected_components"
        else:
            vertices, faces, colors = base.resampled_surface(surface_path, image_path, render_side)
            image_mask, mask, source_mask_path = load_manual_polygon_mask(manual_mask_root, sample_id, image_path, render_side)
            method = f"manual_polygon_mask:{base.root_relative(source_mask_path)}"

        mask_path = mask_root / f"{sample_id}_depthpro_lesion_mask_{render_side}.png"
        save_mask(mask, mask_path)
        labels = measure.label(image_mask)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "image_path": base.root_relative(image_path),
                "surface_path": base.root_relative(surface_path),
                "mask_path": base.root_relative(mask_path),
                "mask_pixels": int(image_mask.sum()),
                "components": int(labels.max()),
                "method": method,
            }
        )
        samples.append(
            {
                "sample_id": sample_id,
                "original_panel": base.original_image_panel(image_path),
                "depth_panel": base.depth_panel(surface_path),
                "image_mask_panel": mask_for_original_panel(image_mask, image_path),
                "full_mask_panel": mask_for_full_panel(mask),
                "vertices": vertices,
                "faces": faces,
                "colors": colors,
                "mask": mask,
            }
        )

    images = []
    yaw_rad = math.radians(front_yaw_degrees)
    overlay_min_strength = float(np.clip(overlay_min_strength, 0.0, 1.0))
    overlay_max_strength = float(np.clip(overlay_max_strength, overlay_min_strength, 1.0))
    for frame_index in range(frames):
        angle = yaw_rad * math.sin(2.0 * math.pi * frame_index / frames)
        if blink_mode == "hard":
            cycle_position = (pulse_cycles * frame_index / frames) % 1.0
            overlay_strength = overlay_max_strength if cycle_position < 0.5 else overlay_min_strength
        else:
            pulse = 0.5 - 0.5 * math.cos(2.0 * math.pi * pulse_cycles * frame_index / frames)
            overlay_strength = overlay_min_strength + (overlay_max_strength - overlay_min_strength) * pulse
        rows = []
        for sample in samples:
            original_panel = overlay_mask(sample["original_panel"], sample["image_mask_panel"], overlay_strength)
            depth_panel = (
                overlay_mask(sample["depth_panel"], sample["full_mask_panel"], overlay_strength)
                if depth_overlay
                else sample["depth_panel"]
            )
            surface_strength = (
                overlay_strength
                if surface_overlay_mode in {"pulse", "pulse_outline", "pulse_screen_outline", "subtle_boundary_outline", "outline"}
                else 0.0
            )
            surface_panel = render_segmented_surface(
                sample["vertices"],
                sample["faces"],
                sample["colors"],
                sample["mask"],
                angle,
                depth_scale,
                surface_strength,
                surface_overlay_mode,
                surface_outline_width,
            )
            rows.append((original_panel, depth_panel, surface_panel))
        images.append(compose_frame(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_gif_timing:
        imageio.mimsave(output_path, images, duration=1 / fps, loop=0)
    else:
        imageio.mimsave(output_path, images, duration=gif_duration_arg(len(images), fps), loop=0)
    write_manifest(manifest_path, manifest_rows)
    write_notebook(notebook_path, output_path, samples, manifest_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--notebook", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--manual-polygon-masks",
        action="store_true",
        help="Use the real manually drawn Fitzpatrick polygon masks and write to manual-specific defaults.",
    )
    parser.add_argument("--manual-mask-root", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.85)
    parser.add_argument("--front-yaw-degrees", type=float, default=14.0)
    parser.add_argument("--render-side", type=int, default=base.RENDER_SURFACE_SIDE)
    parser.add_argument("--pulse-cycles", type=float, default=2.0)
    depth_overlay_group = parser.add_mutually_exclusive_group()
    depth_overlay_group.add_argument(
        "--depth-overlay",
        dest="depth_overlay",
        action="store_true",
        help="Overlay masks on the depth-map column.",
    )
    depth_overlay_group.add_argument(
        "--no-depth-overlay",
        dest="depth_overlay",
        action="store_false",
        help="Leave the depth-map column unsegmented.",
    )
    parser.set_defaults(depth_overlay=None)
    parser.add_argument(
        "--surface-overlay-mode",
        choices=("pulse", "pulse_outline", "pulse_screen_outline", "subtle_boundary_outline", "outline", "none"),
        default="pulse",
        help="Use pulsed lesion tint, boundary outline, or original 3D colors on the 3D surface.",
    )
    parser.add_argument(
        "--surface-outline-width",
        type=int,
        default=3,
        help="Fixed screen-space outline width in pixels for pulse_screen_outline mode.",
    )
    parser.add_argument(
        "--blink-mode",
        choices=("pulse", "hard"),
        default="pulse",
        help="Use a smooth sinusoidal pulse or hard on/off blink.",
    )
    parser.add_argument("--overlay-min-strength", type=float, default=0.18)
    parser.add_argument("--overlay-max-strength", type=float, default=1.0)
    parser.add_argument(
        "--legacy-gif-timing",
        action="store_true",
        help="Write GIF timing the same way as the original README GIF for playback-speed matching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_mode = args.manual_polygon_masks or args.manual_mask_root is not None
    sample_ids = tuple(args.sample_ids) if args.sample_ids else base.DEFAULT_SAMPLE_IDS
    output_path = args.output or (DEFAULT_MANUAL_OUTPUT if manual_mode else DEFAULT_OUTPUT)
    notebook_path = args.notebook or (DEFAULT_MANUAL_NOTEBOOK if manual_mode else DEFAULT_NOTEBOOK)
    mask_root = args.mask_root or (DEFAULT_MANUAL_RESAMPLED_MASK_ROOT if manual_mode else DEFAULT_MASK_ROOT)
    manifest_path = args.manifest or (DEFAULT_MANUAL_MANIFEST if manual_mode else DEFAULT_MANIFEST)
    manual_mask_root = args.manual_mask_root or (DEFAULT_MANUAL_POLYGON_MASK_ROOT if manual_mode else None)
    fps = args.fps if args.fps is not None else (40 if manual_mode else 12)
    depth_overlay = args.depth_overlay if args.depth_overlay is not None else not manual_mode
    build_gif(
        sample_ids=sample_ids,
        output_path=output_path,
        mask_root=mask_root,
        manifest_path=manifest_path,
        notebook_path=notebook_path,
        manual_mask_root=manual_mask_root,
        frames=args.frames,
        fps=fps,
        depth_scale=args.depth_scale,
        front_yaw_degrees=args.front_yaw_degrees,
        render_side=args.render_side,
        pulse_cycles=args.pulse_cycles,
        depth_overlay=depth_overlay,
        surface_overlay_mode=args.surface_overlay_mode,
        legacy_gif_timing=args.legacy_gif_timing,
        blink_mode=args.blink_mode,
        surface_outline_width=args.surface_outline_width,
        overlay_min_strength=args.overlay_min_strength,
        overlay_max_strength=args.overlay_max_strength,
    )
    print(f"Wrote {base.root_relative(output_path)}")
    print(f"Wrote {base.root_relative(manifest_path)}")
    print(f"Wrote {base.root_relative(notebook_path)}")


if __name__ == "__main__":
    main()

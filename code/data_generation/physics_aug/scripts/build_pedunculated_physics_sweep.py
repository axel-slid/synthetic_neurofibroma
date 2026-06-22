#!/usr/bin/env python3
"""Generate 10 flat-to-pedunculated physics-style lesion sweeps."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from plotly.utils import PlotlyJSONEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_constant_radius_gravity_comparison import (  # noqa: E402
    interpolated_lesion_colors,
    quadratic_skin_points,
)
from build_physics_aug_growth import (  # noqa: E402
    HSR_MESH_ROOT,
    ROOT,
    GrowthFrame,
    combine_base_and_lesion,
    compute_vertex_normals,
    crop_mesh_to_target,
    pick_target_vertex,
    read_colored_ply,
    remove_degenerate_faces,
    rgb_strings,
    sample_skin_and_color,
    simulate_soft_body_surface,
    target_basis,
    write_colored_ply,
)

DATASET_ROOT = ROOT / "data" / "synthetic" / "physics_aug_pedunculated_sweep"
COMBINED_NOTEBOOK_NAME = "pedunculated_physics_sweep_all_methods.ipynb"


@dataclass(frozen=True)
class PedunculationMethod:
    method_id: str
    label: str
    implementation: str
    final_height: float
    support_radius: float
    neck_radius: float
    bulb_radius: float
    stalk_fraction: float
    bend: float
    bend_power: float
    gravity_scale: float
    shape_memory: float
    contact_adhesion: float
    lateral: float
    bulb_power: float
    pear_bias: float
    lobe_amp: float
    twist: float
    vertical_compression: float
    notes: str


def smoothstep(value: float | np.ndarray) -> float | np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, stop: float, amount: float | np.ndarray) -> float | np.ndarray:
    return start + (stop - start) * amount


def method_plan() -> list[PedunculationMethod]:
    return [
        PedunculationMethod(
            method_id="m00_axisymmetric_control",
            label="00 axisymmetric control",
            implementation="analytic_blended_dome_to_stalk",
            final_height=0.092,
            support_radius=0.046,
            neck_radius=0.016,
            bulb_radius=0.030,
            stalk_fraction=0.46,
            bend=0.10,
            bend_power=1.8,
            gravity_scale=0.00,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=0.00,
            bulb_power=0.78,
            pear_bias=0.10,
            lobe_amp=0.00,
            twist=0.00,
            vertical_compression=0.00,
            notes="Clean analytic baseline; shows growth and necking without meaningful gravity.",
        ),
        PedunculationMethod(
            method_id="m01_gravity_bent_stalk",
            label="01 gravity bent stalk",
            implementation="analytic_centerline_gravity_bend",
            final_height=0.104,
            support_radius=0.046,
            neck_radius=0.013,
            bulb_radius=0.032,
            stalk_fraction=0.44,
            bend=0.48,
            bend_power=1.65,
            gravity_scale=1.00,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=0.00,
            bulb_power=0.78,
            pear_bias=0.18,
            lobe_amp=0.00,
            twist=0.00,
            vertical_compression=0.05,
            notes="Closed-form centerline bend along local gravity; readable but idealized.",
        ),
        PedunculationMethod(
            method_id="m02_long_thin_stalk",
            label="02 long thin stalk",
            implementation="analytic_long_stalk_small_neck",
            final_height=0.120,
            support_radius=0.044,
            neck_radius=0.009,
            bulb_radius=0.027,
            stalk_fraction=0.58,
            bend=0.42,
            bend_power=1.30,
            gravity_scale=1.20,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=0.02,
            bulb_power=0.92,
            pear_bias=0.22,
            lobe_amp=0.00,
            twist=0.10,
            vertical_compression=0.03,
            notes="Thin neck and longer stalk; deliberately strongly pedunculated.",
        ),
        PedunculationMethod(
            method_id="m03_heavy_pear_bulb",
            label="03 heavy pear bulb",
            implementation="analytic_pear_bulb_gravity",
            final_height=0.108,
            support_radius=0.046,
            neck_radius=0.012,
            bulb_radius=0.038,
            stalk_fraction=0.36,
            bend=0.58,
            bend_power=1.85,
            gravity_scale=1.35,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=-0.02,
            bulb_power=0.62,
            pear_bias=0.52,
            lobe_amp=0.00,
            twist=0.00,
            vertical_compression=0.10,
            notes="Larger pear-shaped terminal bulb with a visible gravity offset.",
        ),
        PedunculationMethod(
            method_id="m04_mass_spring_soft",
            label="04 mass-spring soft",
            implementation="pbd_mass_spring_low_gravity",
            final_height=0.102,
            support_radius=0.046,
            neck_radius=0.014,
            bulb_radius=0.032,
            stalk_fraction=0.42,
            bend=0.34,
            bend_power=1.60,
            gravity_scale=1.20,
            shape_memory=0.075,
            contact_adhesion=0.08,
            lateral=0.03,
            bulb_power=0.82,
            pear_bias=0.20,
            lobe_amp=0.00,
            twist=0.18,
            vertical_compression=0.04,
            notes="Position-based mass-spring relaxation with mild gravity; usually the most conservative.",
        ),
        PedunculationMethod(
            method_id="m05_mass_spring_heavy_contact",
            label="05 mass-spring contact",
            implementation="pbd_mass_spring_heavy_contact",
            final_height=0.106,
            support_radius=0.046,
            neck_radius=0.012,
            bulb_radius=0.035,
            stalk_fraction=0.38,
            bend=0.55,
            bend_power=1.50,
            gravity_scale=3.60,
            shape_memory=0.030,
            contact_adhesion=0.42,
            lateral=-0.02,
            bulb_power=0.72,
            pear_bias=0.30,
            lobe_amp=0.00,
            twist=0.20,
            vertical_compression=0.09,
            notes="Mass-spring with stronger gravity and skin-plane contact; tests plop/settle behavior.",
        ),
        PedunculationMethod(
            method_id="m06_catenary_droop",
            label="06 catenary droop",
            implementation="analytic_catenary_like_stalk",
            final_height=0.116,
            support_radius=0.045,
            neck_radius=0.011,
            bulb_radius=0.033,
            stalk_fraction=0.52,
            bend=0.72,
            bend_power=1.15,
            gravity_scale=1.55,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=0.00,
            bulb_power=0.80,
            pear_bias=0.36,
            lobe_amp=0.00,
            twist=0.00,
            vertical_compression=0.13,
            notes="Catenary-like centerline bend, useful for exaggerated hanging morphology.",
        ),
        PedunculationMethod(
            method_id="m07_asymmetric_lobulated",
            label="07 asymmetric lobulated",
            implementation="analytic_lobulated_asymmetric",
            final_height=0.100,
            support_radius=0.046,
            neck_radius=0.014,
            bulb_radius=0.034,
            stalk_fraction=0.40,
            bend=0.44,
            bend_power=1.60,
            gravity_scale=1.00,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=0.08,
            bulb_power=0.70,
            pear_bias=0.24,
            lobe_amp=0.115,
            twist=1.20,
            vertical_compression=0.04,
            notes="Adds low-frequency lobulation and lateral asymmetry.",
        ),
        PedunculationMethod(
            method_id="m08_volume_preserving_pear",
            label="08 volume-preserving pear",
            implementation="analytic_volume_preserving_pear",
            final_height=0.110,
            support_radius=0.046,
            neck_radius=0.012,
            bulb_radius=0.036,
            stalk_fraction=0.34,
            bend=0.36,
            bend_power=1.80,
            gravity_scale=0.85,
            shape_memory=1.00,
            contact_adhesion=0.00,
            lateral=0.02,
            bulb_power=0.58,
            pear_bias=0.58,
            lobe_amp=0.025,
            twist=0.35,
            vertical_compression=0.06,
            notes="Keeps a fuller lower bulb as the neck narrows; often plausible for mature lesions.",
        ),
        PedunculationMethod(
            method_id="m09_hybrid_realistic",
            label="09 hybrid realistic",
            implementation="hybrid_analytic_plus_mild_pbd",
            final_height=0.104,
            support_radius=0.046,
            neck_radius=0.013,
            bulb_radius=0.034,
            stalk_fraction=0.41,
            bend=0.40,
            bend_power=1.65,
            gravity_scale=1.65,
            shape_memory=0.055,
            contact_adhesion=0.16,
            lateral=0.03,
            bulb_power=0.72,
            pear_bias=0.30,
            lobe_amp=0.035,
            twist=0.45,
            vertical_compression=0.05,
            notes="Balanced analytic neck/bulb with mild mass-spring relaxation; likely best first-pass candidate.",
        ),
    ]


def clear_output_dirs(root: Path) -> tuple[Path, Path]:
    data_root = root / "data"
    visualization_root = root / "visualizations"
    for child in (
        data_root / "lesion_meshes",
        data_root / "final_combined_meshes",
        data_root / "metadata",
        visualization_root / "gifs",
        visualization_root / "plotly",
    ):
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)
    return data_root, visualization_root


def mesh_faces_for_rings(ring_count: int, angular_segments: int, has_base_center: bool = True) -> np.ndarray:
    offset = 1 if has_base_center else 0
    faces: list[list[int]] = []
    if has_base_center:
        for step in range(angular_segments):
            faces.append([0, offset + ((step + 1) % angular_segments), offset + step])
    for ring in range(ring_count - 1):
        current = offset + ring * angular_segments
        next_ring = offset + (ring + 1) * angular_segments
        for step in range(angular_segments):
            a = current + step
            b = current + ((step + 1) % angular_segments)
            c = next_ring + step
            d = next_ring + ((step + 1) % angular_segments)
            faces.append([a, c, b])
            faces.append([b, c, d])
    top_idx = offset + ring_count * angular_segments
    last = offset + (ring_count - 1) * angular_segments
    for step in range(angular_segments):
        faces.append([last + step, top_idx, last + ((step + 1) % angular_segments)])
    return np.asarray(faces, dtype=np.int32)


def build_local_shape(
    method: PedunculationMethod,
    frame_index: int,
    frame_count: int,
    gravity_direction_2d: np.ndarray,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = frame_index / max(frame_count - 1, 1)
    growth = smoothstep(t)
    ped = smoothstep((t - 0.18) / 0.82)
    height = float(lerp(0.0025, method.final_height, growth))
    neck_radius = float(lerp(method.support_radius * 0.72, method.neck_radius, ped))
    attachment_radius = float(lerp(method.support_radius, method.neck_radius * 1.15, ped**0.88))
    bulb_radius = float(lerp(method.support_radius * 0.42, method.bulb_radius, ped))

    gravity_direction = gravity_direction_2d.astype(np.float32)
    norm = float(np.linalg.norm(gravity_direction))
    if norm <= 1e-8:
        gravity_direction = np.array([0.0, 1.0], dtype=np.float32)
    else:
        gravity_direction /= norm
    lateral_direction = np.array([-gravity_direction[1], gravity_direction[0]], dtype=np.float32)

    ring_s = np.linspace(0.0, 0.96, radial_segments, dtype=np.float32)
    vertices = [[0.0, 0.0, 0.0]]
    radial_weight = [1.0]
    for s in ring_s:
        dome_radius = method.support_radius * np.sqrt(max(0.0, 1.0 - s**1.58))
        dome_radius *= 1.0 - 0.12 * growth
        dome_y = height * np.power(s, 0.72 + 0.35 * growth)

        taper = smoothstep(s / 0.20)
        stalk_radius = float(lerp(attachment_radius, neck_radius, taper))
        if s < method.stalk_fraction:
            ped_radius = stalk_radius
        else:
            q = float(np.clip((s - method.stalk_fraction) / max(1.0 - method.stalk_fraction, 1e-6), 0.0, 1.0))
            bulb = math.sin(math.pi * q) ** method.bulb_power
            pear = 1.0 + method.pear_bias * (1.0 - q) * (1.0 - 0.45 * q)
            ped_radius = bulb_radius * bulb * pear + neck_radius * (1.0 - q) ** 2

        radius = float(lerp(dome_radius, ped_radius, ped))
        y = float(lerp(dome_y, height * s * (1.0 - method.vertical_compression * ped * s), ped))
        bend_curve = np.power(s, method.bend_power)
        catenary_extra = 0.45 * method.bend * method.support_radius * ped * np.power(max(s - 0.25, 0.0), 2.0)
        gravity_offset = (
            method.bend
            * method.gravity_scale
            * method.support_radius
            * ped
            * bend_curve
            + catenary_extra
        )
        lateral_offset = method.lateral * method.support_radius * ped * math.sin(math.pi * s)
        center = gravity_offset * gravity_direction + lateral_offset * lateral_direction
        twist = method.twist * ped * s
        for step in range(angular_segments):
            theta = 2.0 * math.pi * step / angular_segments + twist
            lobe = 1.0 + method.lobe_amp * ped * math.sin(3.0 * theta + 4.0 * s)
            ring_radius = max(radius * lobe, 0.001)
            local_xz = center + ring_radius * (
                math.cos(theta) * lateral_direction + math.sin(theta) * gravity_direction
            )
            vertices.append([float(local_xz[0]), y, float(local_xz[1])])
            radial_weight.append(float(np.clip(radius / max(method.support_radius, 1e-6), 0.0, 1.7)))

    top_offset = method.bend * method.gravity_scale * method.support_radius * ped
    top_center = top_offset * gravity_direction + method.lateral * 0.35 * method.support_radius * ped * lateral_direction
    vertices.append([float(top_center[0]), height * (1.0 - 0.30 * method.vertical_compression * ped), float(top_center[1])])
    radial_weight.append(0.0)

    xyz = np.asarray(vertices, dtype=np.float32)
    faces = mesh_faces_for_rings(radial_segments, angular_segments)
    radial_weight_arr = np.asarray(radial_weight, dtype=np.float32)
    pin_mask = np.zeros(len(xyz), dtype=bool)
    pin_mask[0 : 1 + angular_segments] = True

    if method.implementation.startswith("pbd") or method.implementation.startswith("hybrid"):
        frame = GrowthFrame(
            index=frame_index,
            phase=method.label,
            phase_slug=method.method_id,
            growth_t=float(t),
            height=float(height),
            support_radius=float(method.support_radius),
            roundness=1.0,
            lesion_blend=0.5,
            gravity_scale=float(method.gravity_scale),
            shape_memory=float(method.shape_memory),
            contact_adhesion=float(method.contact_adhesion),
        )
        xyz = simulate_soft_body_surface(
            xyz,
            faces,
            pin_mask,
            frame,
            np.array([gravity_direction[0], 0.0, gravity_direction[1]], dtype=np.float32),
            support_radius=method.support_radius,
        )

    return xyz, faces, radial_weight_arr, pin_mask


def world_lesion_from_local(
    local_xyz: np.ndarray,
    radial_weight: np.ndarray,
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    base_rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    local_points = local_xyz[:, [0, 2]].astype(np.float32)
    heights = np.maximum(local_xyz[:, 1], 0.0).astype(np.float32)
    skin_points = quadratic_skin_points(
        local_points,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        base_xyz,
        base_faces,
        support_radius=max(support_radius, float(np.max(np.linalg.norm(local_points, axis=1))) * 0.6),
    )
    _, skin_colors = sample_skin_and_color(
        local_points,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        base_xyz,
        base_faces,
        base_rgb,
        max(support_radius, float(np.max(np.linalg.norm(local_points, axis=1))) * 0.6),
    )
    xyz = skin_points + heights[:, None] * normal
    rgb = interpolated_lesion_colors(skin_colors, heights, np.clip(radial_weight, 0.0, 1.0))
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def localize_world(
    xyz: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    offsets = xyz - anchor
    return np.column_stack([offsets @ tangent_u, offsets @ normal, offsets @ tangent_v]).astype(np.float32)


def surface_area(xyz: np.ndarray, faces: np.ndarray) -> float:
    triangles = xyz[faces]
    return float(np.sum(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2.0))


def frame_metrics(local_xyz: np.ndarray, faces: np.ndarray, method: PedunculationMethod, frame_index: int, frame_count: int) -> dict[str, float | int | str]:
    heights = np.maximum(local_xyz[:, 1], 0.0)
    local_xz = local_xyz[:, [0, 2]]
    radial = np.linalg.norm(local_xz, axis=1)
    return {
        "method_id": method.method_id,
        "frame_index": frame_index,
        "growth_t": frame_index / max(frame_count - 1, 1),
        "peak_height_m": float(np.max(heights)),
        "mean_height_m": float(np.mean(heights)),
        "max_radial_extent_m": float(np.max(radial)),
        "surface_area_m2": surface_area(local_xyz, faces),
    }


def make_patch_figure(
    skin_local: np.ndarray,
    skin_faces: np.ndarray,
    skin_rgb: np.ndarray,
    lesion_local: np.ndarray,
    lesion_faces: np.ndarray,
    lesion_rgb: np.ndarray,
    title: str,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=skin_local[:, 0],
                y=skin_local[:, 1],
                z=skin_local[:, 2],
                i=skin_faces[:, 0],
                j=skin_faces[:, 1],
                k=skin_faces[:, 2],
                vertexcolor=rgb_strings(skin_rgb),
                flatshading=False,
                lighting=dict(ambient=0.96, diffuse=0.52, specular=0.025, roughness=0.96),
                hoverinfo="skip",
                name="skin",
            ),
            go.Mesh3d(
                x=lesion_local[:, 0],
                y=lesion_local[:, 1],
                z=lesion_local[:, 2],
                i=lesion_faces[:, 0],
                j=lesion_faces[:, 1],
                k=lesion_faces[:, 2],
                vertexcolor=rgb_strings(lesion_rgb),
                flatshading=False,
                lighting=dict(ambient=0.92, diffuse=0.62, specular=0.05, roughness=0.88),
                hoverinfo="skip",
                name="lesion",
            ),
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[-half_width, half_width]),
            yaxis=dict(visible=False, range=[-0.020, depth_after]),
            zaxis=dict(visible=False, range=[-half_height, half_height]),
            bgcolor="rgb(244,246,249)",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=0.56, z=1.0),
            camera=dict(
                eye=dict(x=1.80, y=0.28, z=0.35),
                center=dict(x=0.0, y=0.040, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        width=900,
        height=720,
        margin=dict(l=0, r=0, t=54, b=0),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def annotate_png(path: Path, method: PedunculationMethod, frame_index: int, frame_count: int, metrics: dict[str, object]) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    label = (
        f"{method.label} | {frame_index + 1:02d}/{frame_count} | "
        f"height {float(metrics['peak_height_m']) * 1000:.0f} mm"
    )
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.rounded_rectangle((18, 18, 36 + width, 36 + height), radius=7, fill=(255, 255, 255, 224))
    draw.text((27, 25), label, font=font, fill=(36, 39, 44, 255))
    image.save(path)


def render_variant_gif(
    method: PedunculationMethod,
    frame_records: list[dict[str, object]],
    skin_local: np.ndarray,
    skin_faces: np.ndarray,
    skin_rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_after: float,
    gif_path: Path,
) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    with tempfile.TemporaryDirectory(prefix=f"{method.method_id}_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for idx, record in enumerate(frame_records):
            lesion_xyz, lesion_faces, lesion_rgb = read_colored_ply(DATASET_ROOT / str(record["lesion_mesh"]))
            lesion_local = localize_world(lesion_xyz, anchor, normal, tangent_u, tangent_v)
            fig = make_patch_figure(
                skin_local,
                skin_faces,
                skin_rgb,
                lesion_local,
                lesion_faces,
                lesion_rgb,
                method.label,
                half_width,
                half_height,
                depth_after,
            )
            png_path = tmp_dir / f"frame_{idx:03d}.png"
            fig.write_image(png_path, scale=1)
            annotate_png(png_path, method, idx, len(frame_records), record["metrics"])
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=0.18, loop=0)


def make_variant_figure(
    method: PedunculationMethod,
    frame_records: list[dict[str, object]],
    skin_local: np.ndarray,
    skin_faces: np.ndarray,
    skin_rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> go.Figure:
    first = frame_records[0]
    lesion_xyz, lesion_faces, lesion_rgb = read_colored_ply(DATASET_ROOT / str(first["lesion_mesh"]))
    lesion_local = localize_world(lesion_xyz, anchor, normal, tangent_u, tangent_v)
    fig = make_patch_figure(
        skin_local,
        skin_faces,
        skin_rgb,
        lesion_local,
        lesion_faces,
        lesion_rgb,
        method.label,
        half_width,
        half_height,
        depth_after,
    )
    frames = []
    for record in frame_records:
        lesion_xyz, lesion_faces, lesion_rgb = read_colored_ply(DATASET_ROOT / str(record["lesion_mesh"]))
        lesion_local = localize_world(lesion_xyz, anchor, normal, tangent_u, tangent_v)
        metrics = record["metrics"]
        frames.append(
            go.Frame(
                name=f"{int(record['frame_index']) + 1:02d}",
                data=[
                    go.Mesh3d(
                        x=lesion_local[:, 0],
                        y=lesion_local[:, 1],
                        z=lesion_local[:, 2],
                        i=lesion_faces[:, 0],
                        j=lesion_faces[:, 1],
                        k=lesion_faces[:, 2],
                        vertexcolor=rgb_strings(lesion_rgb),
                        flatshading=False,
                        lighting=dict(ambient=0.92, diffuse=0.62, specular=0.05, roughness=0.88),
                        hoverinfo="skip",
                        name="lesion",
                    )
                ],
                traces=[1],
                layout=go.Layout(
                    title_text=(
                        f"{method.label} - frame {int(record['frame_index']) + 1:02d}/"
                        f"{len(frame_records)} - height {float(metrics['peak_height_m']) * 1000:.0f} mm"
                    )
                ),
            )
        )
    steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]
    fig.frames = frames
    fig.update_layout(
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
                                "frame": {"duration": 150, "redraw": True},
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


def plotly_output_cell(figure: go.Figure, label: str) -> nbf.NotebookNode:
    """Create a code-free output cell with the Plotly figure already embedded."""
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    cell = nbf.v4.new_code_cell(
        source="",
        execution_count=None,
        metadata={"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
        outputs=[
            nbf.v4.new_output(
                output_type="display_data",
                data={
                    "application/vnd.plotly.v1+json": payload,
                    "text/plain": f"<Plotly Figure: {label}>",
                },
                metadata={},
            )
        ],
    )
    return cell


def write_combined_notebook(
    notebook_path: Path,
    method_figures: list[tuple[PedunculationMethod, go.Figure]],
) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        nbf.v4.new_markdown_cell("# Pedunculated physics sweep"),
        nbf.v4.new_markdown_cell(
            "Ten 30-frame implementations. Each output is a pre-generated Plotly slider; code cells are intentionally empty."
        ),
    ]
    for method, figure in method_figures:
        cells.append(nbf.v4.new_markdown_cell(f"## {method.label}\n\n{method.notes}"))
        cells.append(plotly_output_cell(figure, method.label))

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", pygments_lexer="ipython3"),
        )
    )
    nbf.write(notebook, notebook_path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace) -> None:
    scan_id = args.scan_id
    base_mesh_path = HSR_MESH_ROOT / f"{scan_id}_closed_textured_mesh.ply"
    if not base_mesh_path.exists():
        raise FileNotFoundError(f"Missing closed HSR mesh: {base_mesh_path}")

    data_root, visualization_root = clear_output_dirs(DATASET_ROOT)
    base_xyz, base_faces, base_rgb = read_colored_ply(base_mesh_path)
    normals = compute_vertex_normals(base_xyz, base_faces)
    target_index = pick_target_vertex(base_xyz, args.target_x, args.target_z, args.target_y, args.target_window)
    anchor, normal, tangent_u, tangent_v = target_basis(base_xyz, normals, target_index)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_local = np.array(
        [
            float(np.dot(gravity_world, tangent_u)),
            float(np.dot(gravity_world, normal)),
            float(np.dot(gravity_world, tangent_v)),
        ],
        dtype=np.float32,
    )
    gravity_2d = np.array([gravity_local[0], gravity_local[2]], dtype=np.float32)

    skin_local, skin_faces, skin_rgb = crop_mesh_to_target(
        base_xyz,
        base_faces,
        base_rgb,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width=args.view_half_width,
        half_height=args.view_half_height,
        depth_before=0.025,
        depth_after=args.view_depth_after,
    )
    skin_patch_path = data_root / "metadata" / f"{scan_id}_local_skin_patch.ply"
    write_colored_ply(skin_patch_path, skin_local, skin_faces, skin_rgb)

    all_records: list[dict[str, object]] = []
    final_metric_rows: list[dict[str, object]] = []
    frame_metric_rows: list[dict[str, object]] = []
    method_figures: list[tuple[PedunculationMethod, go.Figure]] = []
    methods = method_plan()
    combined_notebook_relative = f"visualizations/plotly/{COMBINED_NOTEBOOK_NAME}"
    metadata_record: dict[str, object] = {
        "dataset": "physics_aug_pedunculated_sweep",
        "scan_id": scan_id,
        "source_mesh": str(base_mesh_path.relative_to(ROOT)),
        "target_area": {
            "target_vertex_index": target_index,
            "requested_target_x": args.target_x,
            "requested_target_y": args.target_y,
            "requested_target_z": args.target_z,
            "anchor": anchor.astype(float).tolist(),
            "normal": normal.astype(float).tolist(),
            "tangent_u": tangent_u.astype(float).tolist(),
            "tangent_v": tangent_v.astype(float).tolist(),
            "gravity_local": gravity_local.astype(float).tolist(),
        },
        "simulation": {
            "frame_count": args.frames,
            "radial_segments": args.radial_segments,
            "angular_segments": args.angular_segments,
            "description": "Ten procedural/physics-inspired implementations grow from flat skin elevation to strongly pedunculated lesions under different neck, bulb, gravity, and soft-body assumptions.",
        },
        "coloring": {
            "method": "local skin color interpolation",
            "description": "Lesion colors are barycentrically sampled from nearby skin triangles and given only a subtle height-dependent warm tint.",
        },
        "visualization": {
            "skin_patch": str(skin_patch_path.relative_to(DATASET_ROOT)),
        },
        "methods": [asdict(method) for method in methods],
    }

    for method in methods:
        variant_records: list[dict[str, object]] = []
        method_dir = data_root / "lesion_meshes" / method.method_id
        method_dir.mkdir(parents=True, exist_ok=True)
        final_world = final_faces = final_rgb = None
        final_metrics = None
        for frame_index in range(args.frames):
            local_xyz, faces, radial_weight, _ = build_local_shape(
                method,
                frame_index,
                args.frames,
                gravity_2d,
                args.radial_segments,
                args.angular_segments,
            )
            faces = remove_degenerate_faces(local_xyz, faces)
            lesion_xyz, lesion_rgb = world_lesion_from_local(
                local_xyz,
                radial_weight,
                base_xyz,
                base_faces,
                base_rgb,
                anchor,
                normal,
                tangent_u,
                tangent_v,
                method.support_radius,
            )
            metrics = frame_metrics(local_xyz, faces, method, frame_index, args.frames)
            frame_metric_rows.append(metrics)
            stem = f"{scan_id}_{method.method_id}_frame_{frame_index:03d}"
            lesion_path = method_dir / f"{stem}_lesion.ply"
            write_colored_ply(lesion_path, lesion_xyz, faces, lesion_rgb)
            record = {
                "scan_id": scan_id,
                "method_id": method.method_id,
                "label": method.label,
                "frame_index": frame_index,
                "lesion_mesh": str(lesion_path.relative_to(DATASET_ROOT)),
                "metrics": metrics,
            }
            variant_records.append(record)
            all_records.append(record)
            if frame_index == args.frames - 1:
                final_world = lesion_xyz
                final_faces = faces
                final_rgb = lesion_rgb
                final_metrics = metrics

        assert final_world is not None and final_faces is not None and final_rgb is not None and final_metrics is not None
        combined_xyz, combined_faces, combined_rgb = combine_base_and_lesion(
            base_xyz,
            base_faces,
            base_rgb,
            final_world,
            final_faces,
            final_rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            method.support_radius,
            float(final_metrics["peak_height_m"]),
        )
        combined_path = data_root / "final_combined_meshes" / f"{scan_id}_{method.method_id}_final_hsr_lesion.ply"
        write_colored_ply(combined_path, combined_xyz, combined_faces, combined_rgb)
        final_row = {
            "method_id": method.method_id,
            "label": method.label,
            "implementation": method.implementation,
            "final_combined_mesh": str(combined_path.relative_to(DATASET_ROOT)),
            "gif": f"visualizations/gifs/{method.method_id}.gif",
            "notebook": combined_notebook_relative,
            "notes": method.notes,
        }
        final_row.update(final_metrics)
        final_metric_rows.append(final_row)

        gif_path = visualization_root / "gifs" / f"{method.method_id}.gif"
        render_variant_gif(
            method,
            variant_records,
            skin_local,
            skin_faces,
            skin_rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            half_width=args.view_half_width,
            half_height=args.view_half_height,
            depth_after=args.view_depth_after,
            gif_path=gif_path,
        )
        figure = make_variant_figure(
            method,
            variant_records,
            skin_local,
            skin_faces,
            skin_rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            half_width=args.view_half_width,
            half_height=args.view_half_height,
            depth_after=args.view_depth_after,
        )
        method_figures.append((method, figure))
        print(
            f"{method.method_id:28s} frames={len(variant_records)} "
            f"final_height={float(final_metrics['peak_height_m']) * 1000:.1f}mm "
            f"extent={float(final_metrics['max_radial_extent_m']) * 1000:.1f}mm"
        )

    write_combined_notebook(visualization_root / "plotly" / COMBINED_NOTEBOOK_NAME, method_figures)

    metadata_path = data_root / "metadata" / f"{scan_id}_pedunculated_sweep_metadata.json"
    metadata_path.write_text(json.dumps(metadata_record, indent=2), encoding="utf-8")
    write_csv(data_root / "final_metrics.csv", final_metric_rows)
    write_csv(data_root / "frame_metrics.csv", frame_metric_rows)

    manifest = {
        "dataset": "physics_aug_pedunculated_sweep",
        "scan_id": scan_id,
        "metadata": str(metadata_path.relative_to(DATASET_ROOT)),
        "final_metrics": "data/final_metrics.csv",
        "frame_metrics": "data/frame_metrics.csv",
        "plotly_notebook": combined_notebook_relative,
        "methods": final_metric_rows,
        "frames": all_records,
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (visualization_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(DATASET_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", default="HSR0018-Body-070")
    parser.add_argument("--target-x", type=float, default=-0.09)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=1.09)
    parser.add_argument("--target-window", type=float, default=0.040)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--radial-segments", type=int, default=28)
    parser.add_argument("--angular-segments", type=int, default=88)
    parser.add_argument("--view-half-width", type=float, default=0.145)
    parser.add_argument("--view-half-height", type=float, default=0.150)
    parser.add_argument("--view-depth-after", type=float, default=0.155)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()

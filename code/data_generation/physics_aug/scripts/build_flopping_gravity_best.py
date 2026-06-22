#!/usr/bin/env python3
"""Generate high-gravity flopping pedunculated lesion simulations."""

from __future__ import annotations

import argparse
import csv
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
import open3d as o3d
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
    combine_base_and_lesion,
    compute_vertex_normals,
    crop_mesh_to_target,
    read_colored_ply,
    remove_degenerate_faces,
    rgb_strings,
    sample_skin_and_color,
    visible_base_faces,
    write_colored_ply,
)

DATASET_ROOT = ROOT / "data" / "synthetic" / "physics_aug_flopping_gravity_best"
NOTEBOOK_NAME = "flopping_gravity_selected_and_10_lesions.ipynb"
LESION_MATERIAL_COLORS = [
    "rgb(184,119,83)",
    "rgb(193,130,91)",
    "rgb(174,105,78)",
    "rgb(201,139,98)",
    "rgb(181,112,88)",
    "rgb(196,126,86)",
    "rgb(169,101,75)",
    "rgb(207,147,104)",
    "rgb(187,118,94)",
    "rgb(176,108,82)",
]


@dataclass(frozen=True)
class FlopVariant:
    lesion_id: str
    label: str
    target_x: float
    target_z: float
    target_y: float | None
    final_height: float
    support_radius: float
    neck_radius: float
    bulb_radius: float
    stalk_fraction: float
    gravity_scale: float
    flop_distance: float
    arch_height: float
    distal_center_height: float
    sag: float
    lateral: float
    twist: float
    lobe_amp: float
    pear_bias: float
    growth_delay: float
    growth_power: float
    notes: str


def smoothstep(value: float | np.ndarray) -> float | np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, stop: float, amount: float | np.ndarray) -> float | np.ndarray:
    return start + (stop - start) * amount


def variant_plan(target_x: float, target_z: float, target_y: float | None) -> list[FlopVariant]:
    offsets = [
        (0.000, 0.000),
        (0.040, 0.060),
        (-0.050, -0.050),
        (0.080, -0.020),
        (-0.080, 0.100),
        (0.120, 0.130),
        (0.085, -0.065),
        (-0.020, 0.160),
        (0.165, 0.005),
        (-0.120, 0.020),
    ]
    settings = [
        (0.108, 0.046, 0.0125, 0.034, 0.39, 8.8, 0.088, 0.047, 0.026, 0.014, 0.010, 0.35, 0.020, 0.30, 0.00, 1.00),
        (0.103, 0.044, 0.0115, 0.032, 0.42, 7.9, 0.094, 0.044, 0.024, 0.016, -0.014, 0.48, 0.018, 0.25, 0.06, 0.78),
        (0.116, 0.047, 0.0100, 0.035, 0.46, 9.6, 0.104, 0.050, 0.022, 0.020, 0.006, 0.20, 0.015, 0.32, 0.11, 1.18),
        (0.100, 0.043, 0.0130, 0.031, 0.37, 8.2, 0.078, 0.043, 0.027, 0.011, -0.008, 0.65, 0.030, 0.22, 0.02, 0.92),
        (0.112, 0.046, 0.0110, 0.038, 0.36, 9.1, 0.098, 0.049, 0.029, 0.018, 0.018, 0.30, 0.012, 0.42, 0.15, 0.88),
        (0.106, 0.045, 0.0120, 0.033, 0.41, 10.0, 0.106, 0.045, 0.023, 0.021, -0.004, 0.55, 0.024, 0.28, 0.08, 1.35),
        (0.109, 0.048, 0.0135, 0.036, 0.40, 7.6, 0.086, 0.048, 0.030, 0.013, 0.014, 0.12, 0.010, 0.38, 0.18, 0.72),
        (0.102, 0.044, 0.0105, 0.030, 0.48, 9.4, 0.100, 0.042, 0.021, 0.019, -0.018, 0.72, 0.018, 0.24, 0.04, 1.10),
        (0.114, 0.047, 0.0120, 0.037, 0.35, 8.5, 0.092, 0.051, 0.028, 0.017, 0.004, 0.42, 0.035, 0.46, 0.13, 0.96),
        (0.105, 0.045, 0.0115, 0.034, 0.43, 9.8, 0.108, 0.046, 0.022, 0.022, -0.010, 0.28, 0.016, 0.34, 0.09, 1.24),
    ]
    variants = []
    for index, (dx, dz) in enumerate(offsets):
        (
            final_height,
            support_radius,
            neck_radius,
            bulb_radius,
            stalk_fraction,
            gravity_scale,
            flop_distance,
            arch_height,
            distal_center_height,
            sag,
            lateral,
            twist,
            lobe_amp,
            pear_bias,
            growth_delay,
            growth_power,
        ) = settings[index]
        variants.append(
            FlopVariant(
                lesion_id=f"lesion_{index:02d}",
                label=f"lesion {index:02d} high-gravity flop",
                target_x=target_x + dx,
                target_z=target_z + dz,
                target_y=target_y,
                final_height=final_height,
                support_radius=support_radius,
                neck_radius=neck_radius,
                bulb_radius=bulb_radius,
                stalk_fraction=stalk_fraction,
                gravity_scale=gravity_scale,
                flop_distance=flop_distance,
                arch_height=arch_height,
                distal_center_height=distal_center_height,
                sag=sag,
                lateral=lateral,
                twist=twist,
                lobe_amp=lobe_amp,
                pear_bias=pear_bias,
                growth_delay=growth_delay,
                growth_power=growth_power,
                notes=(
                    "Single continuous-gravity flopping model: narrow neck, heavy distal bulb, "
                    "and gravity applied throughout growth so the mass progressively hangs as it becomes pedunculated."
                ),
            )
        )
    return variants


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


def pick_back_target_vertex(
    xyz: np.ndarray,
    target_x: float,
    target_z: float,
    target_y: float | None,
    window: float,
) -> int:
    if target_y is not None:
        target = np.array([target_x, target_y, target_z], dtype=np.float32)
        return int(np.argmin(np.sum((xyz - target) ** 2, axis=1)))

    for multiplier in (1.0, 1.5, 2.25, 3.5):
        radius = window * multiplier
        mask = (np.abs(xyz[:, 0] - target_x) <= radius) & (np.abs(xyz[:, 2] - target_z) <= radius)
        if int(mask.sum()) > 0:
            candidates = np.flatnonzero(mask)
            return int(candidates[np.argmin(xyz[candidates, 1])])
    target = np.array([target_x, xyz[:, 1].min(), target_z], dtype=np.float32)
    return int(np.argmin(np.sum((xyz - target) ** 2, axis=1)))


def target_basis_on_back(
    xyz: np.ndarray,
    normals: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor = xyz[target_index].astype(np.float32)
    normal = normals[target_index].astype(np.float32)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-8:
        normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        normal = normal / normal_length

    body_center = xyz.mean(axis=0).astype(np.float32)
    if float(np.dot(normal, anchor - body_center)) < 0.0:
        normal = -normal
    if normal[1] > 0.0:
        normal = -normal

    vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tangent_v = vertical - float(np.dot(vertical, normal)) * normal
    if float(np.linalg.norm(tangent_v)) <= 1e-6:
        tangent_v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent_v = tangent_v / np.linalg.norm(tangent_v)
    tangent_u = np.cross(tangent_v, normal).astype(np.float32)
    tangent_u = tangent_u / np.linalg.norm(tangent_u)
    if tangent_u[0] < 0.0:
        tangent_u = -tangent_u
        tangent_v = -tangent_v
    tangent_v = np.cross(normal, tangent_u).astype(np.float32)
    tangent_v = tangent_v / np.linalg.norm(tangent_v)
    return anchor, normal.astype(np.float32), tangent_u.astype(np.float32), tangent_v.astype(np.float32)


def continuous_body_mesh_for_plot(
    xyz: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(faces) <= target_faces:
        return xyz.astype(np.float32), faces.astype(np.int32), np.zeros((len(xyz), 3), dtype=np.uint8)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyz.astype(np.float64))
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
    out_xyz = np.asarray(simplified.vertices, dtype=np.float32)
    out_faces = np.asarray(simplified.triangles, dtype=np.int32)
    return out_xyz, out_faces, np.zeros((len(out_xyz), 3), dtype=np.uint8)


def full_body_trace(
    body_xyz: np.ndarray,
    body_faces: np.ndarray,
    body_rgb: np.ndarray,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=body_xyz[:, 0],
        y=body_xyz[:, 1],
        z=body_xyz[:, 2],
        i=body_faces[:, 0],
        j=body_faces[:, 1],
        k=body_faces[:, 2],
        color="rgb(214,174,151)",
        flatshading=False,
        lighting=dict(ambient=0.88, diffuse=0.62, specular=0.018, roughness=0.96),
        hoverinfo="skip",
        name="body",
        opacity=0.72,
    )


def lesion_material_color(record: dict[str, object]) -> str:
    lesion_id = str(record.get("lesion_id", "lesion_00"))
    try:
        index = int(lesion_id.rsplit("_", 1)[-1])
    except ValueError:
        index = 0
    return LESION_MATERIAL_COLORS[index % len(LESION_MATERIAL_COLORS)]


def lesion_world_trace(record: dict[str, object], name: str | None = None) -> go.Mesh3d:
    lesion_xyz, lesion_faces, _ = read_colored_ply(DATASET_ROOT / str(record["lesion_mesh"]))
    return go.Mesh3d(
        x=lesion_xyz[:, 0],
        y=lesion_xyz[:, 1],
        z=lesion_xyz[:, 2],
        i=lesion_faces[:, 0],
        j=lesion_faces[:, 1],
        k=lesion_faces[:, 2],
        color=lesion_material_color(record),
        flatshading=False,
        lighting=dict(ambient=0.88, diffuse=0.70, specular=0.045, roughness=0.90),
        hoverinfo="skip",
        name=name or str(record["lesion_id"]),
    )


def full_body_layout(
    title: str,
    body_xyz: np.ndarray,
    width: int,
    height: int,
) -> go.Layout:
    pad = np.array([0.05, 0.08, 0.04], dtype=np.float32)
    xyz_min = body_xyz.min(axis=0) - pad
    xyz_max = body_xyz.max(axis=0) + pad
    return go.Layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[float(xyz_min[0]), float(xyz_max[0])]),
            yaxis=dict(visible=False, range=[float(xyz_min[1]), float(xyz_max[1])]),
            zaxis=dict(visible=False, range=[float(xyz_min[2]), float(xyz_max[2])]),
            bgcolor="rgb(244,246,249)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0.0, y=-2.25, z=0.38),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=54, b=0),
        paper_bgcolor="white",
        showlegend=False,
    )


def mesh_faces_for_rings(ring_count: int, angular_segments: int) -> np.ndarray:
    faces: list[list[int]] = []
    offset = 1
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


def local_directions(gravity_direction_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gravity_2d = gravity_direction_2d.astype(np.float32)
    norm = float(np.linalg.norm(gravity_2d))
    if norm <= 1e-8:
        gravity_2d = np.array([0.0, 1.0], dtype=np.float32)
    else:
        gravity_2d /= norm
    lateral_2d = np.array([-gravity_2d[1], gravity_2d[0]], dtype=np.float32)
    gravity_3d = np.array([gravity_2d[0], 0.0, gravity_2d[1]], dtype=np.float32)
    lateral_3d = np.array([lateral_2d[0], 0.0, lateral_2d[1]], dtype=np.float32)
    return gravity_2d, lateral_2d, gravity_3d, lateral_3d


def frame_state(variant: FlopVariant, frame_index: int, frame_count: int) -> dict[str, float]:
    t = frame_index / max(frame_count - 1, 1)
    adjusted_t = float(np.clip((t - variant.growth_delay) / max(1.0 - variant.growth_delay, 1e-6), 0.0, 1.0))
    growth_time = float(np.clip(adjusted_t ** variant.growth_power, 0.0, 1.0))
    growth = float(smoothstep(growth_time))
    pedunculation = float(smoothstep((growth_time - 0.12) / 0.68))
    gravity_drive = 0.16 + 0.84 * growth
    gravity_term = variant.gravity_scale * gravity_drive
    neck_release = 0.16 + 0.84 * pedunculation
    flop = float(smoothstep(gravity_drive * neck_release))
    return {
        "growth_t": float(t),
        "adjusted_growth_t": adjusted_t,
        "growth": growth,
        "pedunculation": pedunculation,
        "gravity_drive": gravity_drive,
        "gravity_term": gravity_term,
        "flop": flop,
        "height": float(lerp(0.0025, variant.final_height, growth)),
        "support_radius": float(lerp(variant.support_radius * 0.82, variant.support_radius, growth)),
        "neck_radius": float(lerp(variant.support_radius * 0.55, variant.neck_radius, pedunculation)),
        "bulb_radius": float(lerp(variant.support_radius * 0.34, variant.bulb_radius, pedunculation)),
    }


def centerline(
    s: float,
    variant: FlopVariant,
    state: dict[str, float],
    gravity_3d: np.ndarray,
    lateral_3d: np.ndarray,
) -> np.ndarray:
    height = state["height"]
    flop = state["flop"]
    gravity_drive = state["gravity_drive"]
    upright_y = height * s
    if s <= variant.stalk_fraction:
        q = s / max(variant.stalk_fraction, 1e-6)
        flopped_y = variant.arch_height * math.sin(0.5 * math.pi * smoothstep(q))
    else:
        q = (s - variant.stalk_fraction) / max(1.0 - variant.stalk_fraction, 1e-6)
        flopped_y = lerp(variant.arch_height, variant.distal_center_height, smoothstep(q))
        flopped_y -= variant.sag * gravity_drive * math.sin(math.pi * q)
    flopped_y = max(float(flopped_y), 0.010)
    y = float(lerp(upright_y, flopped_y, flop))
    gravity_offset = variant.flop_distance * flop * (s**1.30)
    lateral_offset = variant.lateral * variant.support_radius * state["pedunculation"] * math.sin(math.pi * s)
    return gravity_offset * gravity_3d + lateral_offset * lateral_3d + np.array([0.0, y, 0.0], dtype=np.float32)


def radius_profile(s: float, variant: FlopVariant, state: dict[str, float]) -> float:
    height = state["height"]
    support_radius = state["support_radius"]
    neck_radius = state["neck_radius"]
    bulb_radius = state["bulb_radius"]
    pedunculation = state["pedunculation"]
    dome_radius = support_radius * math.sqrt(max(0.0, 1.0 - s**1.72)) * (1.0 - 0.10 * state["growth"])
    attachment_radius = float(lerp(support_radius, neck_radius * 1.22, pedunculation**0.85))
    if s < variant.stalk_fraction:
        q = s / max(variant.stalk_fraction, 1e-6)
        ped_radius = float(lerp(attachment_radius, neck_radius, smoothstep(q)))
    else:
        q = (s - variant.stalk_fraction) / max(1.0 - variant.stalk_fraction, 1e-6)
        bulb = math.sin(math.pi * q) ** 0.62
        pear = 1.0 + variant.pear_bias * (1.0 - q) * (1.0 - 0.35 * q)
        ped_radius = bulb_radius * bulb * pear + neck_radius * (1.0 - q) ** 2
    ped_radius *= 1.0 + 0.08 * state["gravity_drive"] * math.sin(math.pi * s)
    return max(float(lerp(dome_radius, ped_radius, pedunculation)), min(0.0018, height * 0.08))


def mature_ring_axes(
    s: float,
    variant: FlopVariant,
    state: dict[str, float],
    gravity_3d: np.ndarray,
    lateral_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-3
    p0 = centerline(max(0.0, s - eps), variant, state, gravity_3d, lateral_3d)
    p1 = centerline(min(1.0, s + eps), variant, state, gravity_3d, lateral_3d)
    tangent = p1 - p0
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        tangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        tangent /= tangent_norm
    axis_a = lateral_3d.astype(np.float32)
    axis_a /= max(float(np.linalg.norm(axis_a)), 1e-8)
    axis_b = np.cross(tangent, axis_a).astype(np.float32)
    axis_b /= max(float(np.linalg.norm(axis_b)), 1e-8)
    return axis_a, axis_b


def build_local_shape(
    variant: FlopVariant,
    frame_index: int,
    frame_count: int,
    gravity_direction_2d: np.ndarray,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    _, _, gravity_3d, lateral_3d = local_directions(gravity_direction_2d)
    state = frame_state(variant, frame_index, frame_count)
    ring_s = np.linspace(0.0, 0.97, radial_segments, dtype=np.float32)
    vertices: list[list[float]] = [[0.0, 0.0, 0.0]]
    radial_weight: list[float] = [1.0]
    for s_raw in ring_s:
        s = float(s_raw)
        center = centerline(s, variant, state, gravity_3d, lateral_3d)
        axis_a, axis_b = mature_ring_axes(s, variant, state, gravity_3d, lateral_3d)
        radius = radius_profile(s, variant, state)
        dome_center = np.array([0.0, state["height"] * (s**0.78), 0.0], dtype=np.float32)
        dome_radius = state["support_radius"] * math.sqrt(max(0.0, 1.0 - s**1.72))
        dome_axis_a = lateral_3d
        dome_axis_b = gravity_3d
        for step in range(angular_segments):
            theta = 2.0 * math.pi * step / angular_segments + variant.twist * state["pedunculation"] * s
            lobe = 1.0 + variant.lobe_amp * state["pedunculation"] * math.sin(3.0 * theta + 5.5 * s)
            mature_vertex = center + max(radius * lobe, 0.001) * (
                math.cos(theta) * axis_a + math.sin(theta) * axis_b
            )
            dome_vertex = dome_center + dome_radius * (
                math.cos(theta) * dome_axis_a + math.sin(theta) * dome_axis_b
            )
            vertex = (1.0 - state["pedunculation"]) * dome_vertex + state["pedunculation"] * mature_vertex
            if vertex[1] < 0.0:
                vertex[1] = 0.0
            vertices.append(vertex.astype(float).tolist())
            radial_weight.append(float(np.clip(radius / max(variant.support_radius, 1e-6), 0.0, 1.6)))
    top_center = centerline(1.0, variant, state, gravity_3d, lateral_3d)
    dome_top = np.array([0.0, state["height"], 0.0], dtype=np.float32)
    top = (1.0 - state["pedunculation"]) * dome_top + state["pedunculation"] * top_center
    top[1] = max(float(top[1]), 0.0)
    vertices.append(top.astype(float).tolist())
    radial_weight.append(0.0)
    xyz = np.asarray(vertices, dtype=np.float32)
    faces = mesh_faces_for_rings(radial_segments, angular_segments)
    return xyz, faces, np.asarray(radial_weight, dtype=np.float32), state


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
    sample_radius = max(support_radius, float(np.max(np.linalg.norm(local_points, axis=1))) * 0.80)
    skin_points = quadratic_skin_points(
        local_points,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        base_xyz,
        base_faces,
        sample_radius,
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
        sample_radius,
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


def frame_metrics(
    local_xyz: np.ndarray,
    faces: np.ndarray,
    variant: FlopVariant,
    state: dict[str, float],
    frame_index: int,
    frame_count: int,
) -> dict[str, float | int | str]:
    heights = np.maximum(local_xyz[:, 1], 0.0)
    local_xz = local_xyz[:, [0, 2]]
    radial = np.linalg.norm(local_xz, axis=1)
    near_skin_fraction = float(np.mean(heights < 0.0025))
    return {
        "lesion_id": variant.lesion_id,
        "frame_index": frame_index,
        "growth_t": frame_index / max(frame_count - 1, 1),
        "adjusted_growth_t": state["adjusted_growth_t"],
        "pedunculation": state["pedunculation"],
        "gravity_drive": state["gravity_drive"],
        "gravity_term": state["gravity_term"],
        "flop": state["flop"],
        "peak_height_m": float(np.max(heights)),
        "mean_height_m": float(np.mean(heights)),
        "max_radial_extent_m": float(np.max(radial)),
        "near_skin_fraction": near_skin_fraction,
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
                lighting=dict(ambient=0.90, diffuse=0.66, specular=0.045, roughness=0.90),
                hoverinfo="skip",
                name="lesion",
            ),
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[-half_width, half_width]),
            yaxis=dict(visible=False, range=[-0.018, depth_after]),
            zaxis=dict(visible=False, range=[-half_height, half_height]),
            bgcolor="rgb(244,246,249)",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=0.52, z=1.0),
            camera=dict(
                eye=dict(x=1.72, y=0.23, z=0.48),
                center=dict(x=0.020, y=0.030, z=0.000),
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


def make_variant_figure(
    variant: FlopVariant,
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
        variant.label,
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
                        lighting=dict(ambient=0.90, diffuse=0.66, specular=0.045, roughness=0.90),
                        hoverinfo="skip",
                        name="lesion",
                    )
                ],
                traces=[1],
                layout=go.Layout(
                    title_text=(
                        f"{variant.label} - frame {int(record['frame_index']) + 1:02d}/"
                        f"{len(frame_records)} - gravity {float(metrics['gravity_term']):.1f} - "
                        f"height {float(metrics['peak_height_m']) * 1000:.0f} mm"
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


def shifted_mesh_trace(
    local_xyz: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    shift: np.ndarray,
    name: str,
    lighting: dict[str, float],
    opacity: float = 1.0,
) -> go.Mesh3d:
    shifted = local_xyz + shift[None, :]
    return go.Mesh3d(
        x=shifted[:, 0],
        y=shifted[:, 1],
        z=shifted[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=rgb_strings(rgb),
        flatshading=False,
        lighting=lighting,
        hoverinfo="skip",
        name=name,
        opacity=opacity,
    )


def make_multi_lesion_figure(
    variant_contexts: list[dict[str, object]],
    frame_count: int,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> go.Figure:
    columns = 5
    rows = int(math.ceil(len(variant_contexts) / columns))
    spacing_x = half_width * 2.40
    spacing_z = half_height * 2.35
    skin_lighting = dict(ambient=0.96, diffuse=0.52, specular=0.025, roughness=0.96)
    lesion_lighting = dict(ambient=0.90, diffuse=0.66, specular=0.045, roughness=0.90)
    shifts = []
    data = []
    lesion_trace_indices = []

    for index, context in enumerate(variant_contexts):
        row = index // columns
        column = index % columns
        shift = np.array(
            [
                (column - (columns - 1) / 2.0) * spacing_x,
                0.0,
                ((rows - 1) / 2.0 - row) * spacing_z,
            ],
            dtype=np.float32,
        )
        shifts.append(shift)
        variant = context["variant"]
        frame_records = context["frame_records"]
        anchor = context["anchor"]
        normal = context["normal"]
        tangent_u = context["tangent_u"]
        tangent_v = context["tangent_v"]
        first = frame_records[0]
        lesion_xyz, lesion_faces, lesion_rgb = read_colored_ply(DATASET_ROOT / str(first["lesion_mesh"]))
        lesion_local = localize_world(lesion_xyz, anchor, normal, tangent_u, tangent_v)
        data.append(
            shifted_mesh_trace(
                context["skin_local"],
                context["skin_faces"],
                context["skin_rgb"],
                shift,
                f"{variant.lesion_id} skin",
                skin_lighting,
                opacity=0.72,
            )
        )
        lesion_trace_indices.append(len(data))
        data.append(
            shifted_mesh_trace(
                lesion_local,
                lesion_faces,
                lesion_rgb,
                shift,
                variant.lesion_id,
                lesion_lighting,
            )
        )

    frames = []
    for frame_index in range(frame_count):
        frame_data = []
        gravity_terms = []
        for context, shift in zip(variant_contexts, shifts):
            frame_records = context["frame_records"]
            anchor = context["anchor"]
            normal = context["normal"]
            tangent_u = context["tangent_u"]
            tangent_v = context["tangent_v"]
            record = frame_records[frame_index]
            gravity_terms.append(float(record["metrics"]["gravity_term"]))
            lesion_xyz, lesion_faces, lesion_rgb = read_colored_ply(DATASET_ROOT / str(record["lesion_mesh"]))
            lesion_local = localize_world(lesion_xyz, anchor, normal, tangent_u, tangent_v)
            frame_data.append(
                shifted_mesh_trace(
                    lesion_local,
                    lesion_faces,
                    lesion_rgb,
                    shift,
                    str(record["lesion_id"]),
                    lesion_lighting,
                )
            )
        frames.append(
            go.Frame(
                name=f"{frame_index + 1:02d}",
                data=frame_data,
                traces=lesion_trace_indices,
                layout=go.Layout(
                    title_text=(
                        f"all lesions - frame {frame_index + 1:02d}/{frame_count} - "
                        f"gravity {float(np.mean(gravity_terms)):.1f}"
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
    max_x = ((columns - 1) / 2.0) * spacing_x + half_width
    max_z = ((rows - 1) / 2.0) * spacing_z + half_height
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title=dict(text="all lesions - continuous gravity flop", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[-max_x, max_x]),
            yaxis=dict(visible=False, range=[-0.018, depth_after]),
            zaxis=dict(visible=False, range=[-max_z, max_z]),
            bgcolor="rgb(244,246,249)",
            aspectmode="manual",
            aspectratio=dict(x=3.2, y=0.48, z=1.20),
            camera=dict(
                eye=dict(x=1.68, y=0.32, z=0.68),
                center=dict(x=0.0, y=0.028, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        width=1300,
        height=760,
        margin=dict(l=0, r=0, t=54, b=0),
        paper_bgcolor="white",
        showlegend=False,
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


def slider_steps(frames: list[go.Frame]) -> list[dict[str, object]]:
    return [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]


def play_pause_menu() -> list[dict[str, object]]:
    return [
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
    ]


def make_full_body_one_lesion_figure(
    variant: FlopVariant,
    frame_records: list[dict[str, object]],
    body_xyz: np.ndarray,
    body_faces: np.ndarray,
    body_rgb: np.ndarray,
) -> go.Figure:
    data = [
        full_body_trace(body_xyz, body_faces, body_rgb),
        lesion_world_trace(frame_records[0], variant.lesion_id),
    ]
    frames = []
    for record in frame_records:
        metrics = record["metrics"]
        frames.append(
            go.Frame(
                name=f"{int(record['frame_index']) + 1:03d}",
                data=[lesion_world_trace(record, variant.lesion_id)],
                traces=[1],
                layout=go.Layout(
                    title_text=(
                        f"one back lesion - frame {int(record['frame_index']) + 1:03d}/"
                        f"{len(frame_records)} - growth {float(metrics['adjusted_growth_t']):.2f} - "
                        f"gravity {float(metrics['gravity_term']):.1f}"
                    )
                ),
            )
        )
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(full_body_layout("one back lesion - continuous gravity", body_xyz, 960, 980))
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "len": 0.86,
                "steps": slider_steps(frames),
            }
        ],
        updatemenus=play_pause_menu(),
    )
    return fig


def make_full_body_multi_lesion_figure(
    variant_contexts: list[dict[str, object]],
    frame_count: int,
    body_xyz: np.ndarray,
    body_faces: np.ndarray,
    body_rgb: np.ndarray,
) -> go.Figure:
    data = [full_body_trace(body_xyz, body_faces, body_rgb)]
    for context in variant_contexts:
        variant = context["variant"]
        frame_records = context["frame_records"]
        data.append(lesion_world_trace(frame_records[0], variant.lesion_id))
    lesion_traces = list(range(1, len(data)))

    frames = []
    for frame_index in range(frame_count):
        frame_data = []
        growth_values = []
        gravity_terms = []
        for context in variant_contexts:
            frame_records = context["frame_records"]
            record = frame_records[frame_index]
            metrics = record["metrics"]
            growth_values.append(float(metrics["adjusted_growth_t"]))
            gravity_terms.append(float(metrics["gravity_term"]))
            frame_data.append(lesion_world_trace(record, str(record["lesion_id"])))
        frames.append(
            go.Frame(
                name=f"{frame_index + 1:03d}",
                data=frame_data,
                traces=lesion_traces,
                layout=go.Layout(
                    title_text=(
                        f"ten back lesions - frame {frame_index + 1:03d}/{frame_count} - "
                        f"mean growth {float(np.mean(growth_values)):.2f} - "
                        f"mean gravity {float(np.mean(gravity_terms)):.1f}"
                    )
                ),
            )
        )

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(full_body_layout("ten back lesions - varied growth speeds", body_xyz, 1080, 980))
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "len": 0.86,
                "steps": slider_steps(frames),
            }
        ],
        updatemenus=play_pause_menu(),
    )
    return fig


def render_figure_gif(figure: go.Figure, gif_path: Path, duration: float = 0.08) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    working = go.Figure(data=figure.data, layout=figure.layout)
    images = []
    with tempfile.TemporaryDirectory(prefix=f"{gif_path.stem}_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for frame_index, frame in enumerate(figure.frames):
            for trace_update, trace_index in zip(frame.data, frame.traces or []):
                working.data[trace_index].update(trace_update)
            if frame.layout and frame.layout.title:
                working.update_layout(title=frame.layout.title)
            png_path = tmp_dir / f"frame_{frame_index:03d}.png"
            working.write_image(png_path, scale=1)
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=duration, loop=0)


def combine_base_and_multiple_lesions(
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    base_rgb: np.ndarray,
    final_lesions: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible_faces = base_faces
    for lesion in final_lesions:
        visible_faces = visible_base_faces(
            base_xyz,
            visible_faces,
            lesion["anchor"],
            lesion["normal"],
            lesion["tangent_u"],
            lesion["tangent_v"],
            float(lesion["support_radius"]),
            float(lesion["max_height"]),
        )

    xyz_parts = [base_xyz]
    rgb_parts = [base_rgb]
    face_parts = [visible_faces]
    offset = len(base_xyz)
    for lesion in final_lesions:
        lesion_xyz = lesion["lesion_xyz"]
        lesion_faces = lesion["lesion_faces"]
        lesion_rgb = lesion["lesion_rgb"]
        xyz_parts.append(lesion_xyz)
        rgb_parts.append(lesion_rgb)
        face_parts.append(lesion_faces + offset)
        offset += len(lesion_xyz)
    return (
        np.vstack(xyz_parts).astype(np.float32),
        np.vstack(face_parts).astype(np.int32),
        np.vstack(rgb_parts).astype(np.uint8),
    )


def annotate_png(path: Path, variant: FlopVariant, frame_index: int, frame_count: int, metrics: dict[str, object]) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    label = (
        f"{variant.lesion_id} | {frame_index + 1:02d}/{frame_count} | "
        f"g {float(metrics['gravity_term']):.1f} | height {float(metrics['peak_height_m']) * 1000:.0f} mm"
    )
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.rounded_rectangle((18, 18, 36 + width, 36 + height), radius=7, fill=(255, 255, 255, 224))
    draw.text((27, 25), label, font=font, fill=(36, 39, 44, 255))
    image.save(path)


def render_variant_gif(
    variant: FlopVariant,
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
    with tempfile.TemporaryDirectory(prefix=f"{variant.lesion_id}_") as tmp_name:
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
                variant.label,
                half_width,
                half_height,
                depth_after,
            )
            png_path = tmp_dir / f"frame_{idx:03d}.png"
            fig.write_image(png_path, scale=1)
            annotate_png(png_path, variant, idx, len(frame_records), record["metrics"])
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=0.18, loop=0)


def plotly_output_cell(figure: go.Figure, label: str) -> nbf.NotebookNode:
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    payload = compact_plotly_payload(payload)
    return nbf.v4.new_code_cell(
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


def compact_plotly_payload(value: object) -> object:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [compact_plotly_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_plotly_payload(item) for key, item in value.items()}
    return value


def write_combined_notebook(
    notebook_path: Path,
    selected_figure: go.Figure,
    multi_lesion_figure: go.Figure,
) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        nbf.v4.new_markdown_cell("# Continuous-gravity flopping pedunculated lesion model"),
        nbf.v4.new_markdown_cell(
            "Two plots are included: one selected lesion simulation and one synchronized ten-lesion simulation. "
            "All Plotly figures are pre-generated; source cells are intentionally empty."
        ),
        nbf.v4.new_markdown_cell("## One lesion"),
        plotly_output_cell(selected_figure, "one lesion continuous-gravity flop"),
        nbf.v4.new_markdown_cell("## Multiple lesions"),
        plotly_output_cell(multi_lesion_figure, "ten lesions continuous-gravity flop"),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", pygments_lexer="ipython3"),
        ),
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
    body_plot_xyz, body_plot_faces, body_plot_rgb = continuous_body_mesh_for_plot(
        base_xyz,
        base_faces,
        args.body_target_faces,
    )
    normals = compute_vertex_normals(base_xyz, base_faces)
    variants = variant_plan(args.target_x, args.target_z, args.target_y)
    final_metric_rows: list[dict[str, object]] = []
    frame_metric_rows: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    variant_contexts: list[dict[str, object]] = []
    final_lesions: list[dict[str, object]] = []
    notebook_relative = f"visualizations/plotly/{NOTEBOOK_NAME}"
    one_gif_relative = "visualizations/gifs/one_back_lesion_full_body.gif"
    multi_gif_relative = "visualizations/gifs/ten_back_lesions_full_body.gif"

    metadata_record: dict[str, object] = {
        "dataset": "physics_aug_flopping_gravity_best",
        "scan_id": scan_id,
        "source_mesh": str(base_mesh_path.relative_to(ROOT)),
        "simulation": {
            "frame_count": args.frames,
            "radial_segments": args.radial_segments,
            "angular_segments": args.angular_segments,
            "body_plot_faces": int(len(body_plot_faces)),
            "body_visualization": "continuous quadric-simplified solid body mesh; no sparse face-sampled dotted mesh",
            "selected_model": "continuous_gravity_flopping_hybrid",
            "description": (
                "One physics method is used for all lesions on the back: a pedunculated narrow-neck stalk with a heavy distal bulb. "
                "Gravity is active throughout the 100-frame simulation and increases smoothly with growth, while per-lesion "
                "growth delays, growth rates, and morphology settings vary across the ten-lesion scene."
            ),
        },
        "coloring": {
            "method": "local skin color interpolation",
            "description": "Lesion colors are sampled from nearby skin triangles and only lightly warmed by height.",
        },
        "variants": [asdict(variant) for variant in variants],
    }

    for variant in variants:
        target_index = pick_back_target_vertex(
            base_xyz,
            variant.target_x,
            variant.target_z,
            variant.target_y,
            args.target_window,
        )
        anchor, normal, tangent_u, tangent_v = target_basis_on_back(base_xyz, normals, target_index)
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
            depth_before=0.030,
            depth_after=args.view_depth_after,
        )
        skin_patch_path = data_root / "metadata" / f"{scan_id}_{variant.lesion_id}_local_skin_patch.ply"
        write_colored_ply(skin_patch_path, skin_local, skin_faces, skin_rgb)

        variant_records: list[dict[str, object]] = []
        method_dir = data_root / "lesion_meshes" / variant.lesion_id
        method_dir.mkdir(parents=True, exist_ok=True)
        final_world = final_faces = final_rgb = None
        final_metrics = None
        for frame_index in range(args.frames):
            local_xyz, faces, radial_weight, state = build_local_shape(
                variant,
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
                variant.support_radius,
            )
            metrics = frame_metrics(local_xyz, faces, variant, state, frame_index, args.frames)
            frame_metric_rows.append(metrics)
            stem = f"{scan_id}_{variant.lesion_id}_frame_{frame_index:03d}"
            lesion_path = method_dir / f"{stem}_lesion.ply"
            write_colored_ply(lesion_path, lesion_xyz, faces, lesion_rgb)
            record = {
                "scan_id": scan_id,
                "lesion_id": variant.lesion_id,
                "label": variant.label,
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
            variant.support_radius,
            float(final_metrics["peak_height_m"]),
        )
        combined_path = data_root / "final_combined_meshes" / f"{scan_id}_{variant.lesion_id}_final_hsr_lesion.ply"
        write_colored_ply(combined_path, combined_xyz, combined_faces, combined_rgb)
        final_lesions.append(
            {
                "lesion_id": variant.lesion_id,
                "anchor": anchor,
                "normal": normal,
                "tangent_u": tangent_u,
                "tangent_v": tangent_v,
                "support_radius": variant.support_radius,
                "max_height": float(final_metrics["peak_height_m"]),
                "lesion_xyz": final_world,
                "lesion_faces": final_faces,
                "lesion_rgb": final_rgb,
            }
        )
        final_row = {
            "lesion_id": variant.lesion_id,
            "label": variant.label,
            "model": "continuous_gravity_flopping_hybrid",
            "gravity_scale": variant.gravity_scale,
            "growth_delay": variant.growth_delay,
            "growth_power": variant.growth_power,
            "final_combined_mesh": str(combined_path.relative_to(DATASET_ROOT)),
            "gif": multi_gif_relative,
            "notebook": notebook_relative,
            "target_vertex_index": target_index,
            "target_x": variant.target_x,
            "target_y": variant.target_y,
            "target_z": variant.target_z,
            "notes": variant.notes,
        }
        final_row.update(final_metrics)
        final_metric_rows.append(final_row)

        context = dict(
            variant=variant,
            frame_records=variant_records,
            anchor=anchor,
            normal=normal,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
        )
        variant_contexts.append(context)
        print(
            f"{variant.lesion_id:10s} frames={len(variant_records)} "
            f"gravity={float(final_metrics['gravity_term']):.1f} "
            f"flop={float(final_metrics['flop']):.2f} "
            f"height={float(final_metrics['peak_height_m']) * 1000:.1f}mm "
            f"extent={float(final_metrics['max_radial_extent_m']) * 1000:.1f}mm"
        )

    all_combined_xyz, all_combined_faces, all_combined_rgb = combine_base_and_multiple_lesions(
        base_xyz,
        base_faces,
        base_rgb,
        final_lesions,
    )
    all_combined_path = data_root / "final_combined_meshes" / f"{scan_id}_all_10_back_lesions_final_hsr_lesion.ply"
    write_colored_ply(all_combined_path, all_combined_xyz, all_combined_faces, all_combined_rgb)

    selected_context = variant_contexts[0]
    selected_figure = make_full_body_one_lesion_figure(
        selected_context["variant"],
        selected_context["frame_records"],
        body_plot_xyz,
        body_plot_faces,
        body_plot_rgb,
    )
    multi_lesion_figure = make_full_body_multi_lesion_figure(
        variant_contexts,
        args.frames,
        body_plot_xyz,
        body_plot_faces,
        body_plot_rgb,
    )
    write_combined_notebook(visualization_root / "plotly" / NOTEBOOK_NAME, selected_figure, multi_lesion_figure)
    render_figure_gif(selected_figure, DATASET_ROOT / one_gif_relative)
    render_figure_gif(multi_lesion_figure, DATASET_ROOT / multi_gif_relative)

    metadata_record["target_areas"] = [
        {
            "lesion_id": row["lesion_id"],
            "target_vertex_index": row["target_vertex_index"],
            "target_x": row["target_x"],
            "target_y": row["target_y"],
            "target_z": row["target_z"],
        }
        for row in final_metric_rows
    ]
    metadata_path = data_root / "metadata" / f"{scan_id}_flopping_gravity_metadata.json"
    metadata_path.write_text(json.dumps(metadata_record, indent=2), encoding="utf-8")
    write_csv(data_root / "final_metrics.csv", final_metric_rows)
    write_csv(data_root / "frame_metrics.csv", frame_metric_rows)
    manifest = {
        "dataset": "physics_aug_flopping_gravity_best",
        "scan_id": scan_id,
        "metadata": str(metadata_path.relative_to(DATASET_ROOT)),
        "final_metrics": "data/final_metrics.csv",
        "frame_metrics": "data/frame_metrics.csv",
        "plotly_notebook": notebook_relative,
        "one_lesion_gif": one_gif_relative,
        "multi_lesion_gif": multi_gif_relative,
        "all_10_final_combined_mesh": str(all_combined_path.relative_to(DATASET_ROOT)),
        "lesions": final_metric_rows,
        "frames": all_records,
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (visualization_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(DATASET_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", default="HSR0018-Body-070")
    parser.add_argument("--target-x", type=float, default=-0.08)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=1.12)
    parser.add_argument("--target-window", type=float, default=0.035)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--radial-segments", type=int, default=10)
    parser.add_argument("--angular-segments", type=int, default=32)
    parser.add_argument("--body-target-faces", type=int, default=8000)
    parser.add_argument("--view-half-width", type=float, default=0.205)
    parser.add_argument("--view-half-height", type=float, default=0.205)
    parser.add_argument("--view-depth-after", type=float, default=0.145)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()

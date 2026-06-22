#!/usr/bin/env python3
"""Build a compact multi-lesion physics dataset with Plotly progression views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "data" / "synthetic" / "multiple_lesion_physics"
HSR_MESH_ROOT = ROOT / "data" / "hsr" / "visualizations" / "meshes"

MODEL_NAME = "continuous_gravity_multi_lesion_flop"
COUNT_PRESETS = (10, 25, 50, 100)


@dataclass(frozen=True)
class LesionSpec:
    lesion_id: str
    target_x: float
    target_y: float
    target_z: float
    target_vertex_index: int
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
    growth_duration: float
    growth_power: float
    growth_rate: float
    color_rgb: tuple[int, int, int]


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATASET_ROOT))
    except ValueError:
        return str(path)


def smoothstep(value: float | np.ndarray) -> float | np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, stop: float, amount: float | np.ndarray) -> float | np.ndarray:
    return start + (stop - start) * amount


def read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    face = ply["face"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    if {"red", "green", "blue"}.issubset(vertex.dtype.names or ()):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), [210, 164, 139], dtype=np.uint8)
    faces = np.vstack(face["vertex_indices"]).astype(np.int32)
    return xyz, faces, rgb


def write_colored_ply(path: Path, xyz: np.ndarray, faces: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    face_array = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_array["vertex_indices"] = faces
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(face_array, "face")], text=False).write(path)


def rgb_strings(rgb: np.ndarray) -> list[str]:
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]


def rgb_string(color: tuple[int, int, int] | np.ndarray) -> str:
    red, green, blue = [int(value) for value in color]
    return f"rgb({red},{green},{blue})"


def compute_vertex_normals(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(xyz, dtype=np.float64)
    triangles = xyz[faces].astype(np.float64)
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    face_normals[valid] /= lengths[valid, None]
    for column in range(3):
        np.add.at(normals, faces[:, column], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, -1.0, 0.0])
    return normals.astype(np.float32)


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(faces) <= target_faces:
        return vertices.astype(np.float32), faces.astype(np.int32), rgb.astype(np.uint8)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(rgb.astype(np.float64) / 255.0, 0.0, 1.0))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    simplified_rgb = np.clip(np.rint(np.asarray(simplified.vertex_colors) * 255.0), 0, 255).astype(np.uint8)
    return (
        np.asarray(simplified.vertices, dtype=np.float32),
        np.asarray(simplified.triangles, dtype=np.int32),
        simplified_rgb,
    )


def remove_degenerate_faces(xyz: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(faces) == 0:
        return faces
    triangles = xyz[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return faces[areas > eps]


def pick_back_target_vertex(
    xyz: np.ndarray,
    target_x: float,
    target_z: float,
    window: float,
) -> int:
    for multiplier in (1.0, 1.5, 2.25, 3.5, 5.0):
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


def fit_skin_quadratic(
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    support_radius: float,
) -> np.ndarray:
    centroids = skin_vertices[skin_faces].mean(axis=1)
    offsets = centroids - anchor
    local_u = offsets @ tangent_u
    local_v = offsets @ tangent_v
    local_n = offsets @ normal
    radial = np.sqrt(local_u * local_u + local_v * local_v)
    candidate = (radial <= max(0.070, 2.65 * support_radius)) & (
        np.abs(local_n) <= max(0.045, 1.75 * support_radius)
    )
    if int(candidate.sum()) < 12:
        candidate = radial <= max(0.095, 3.5 * support_radius)
    if int(candidate.sum()) < 12:
        order = np.argsort(radial)
        candidate = np.zeros(len(skin_faces), dtype=bool)
        candidate[order[: min(180, len(order))]] = True

    fit_u = local_u[candidate]
    fit_v = local_v[candidate]
    fit_n = local_n[candidate]
    fit_radius = radial[candidate]
    if len(fit_n) < 6:
        return np.zeros(6, dtype=np.float32)

    weights = np.exp(-0.5 * (fit_radius / max(0.030, 1.45 * support_radius)) ** 2)
    design = np.column_stack(
        [
            np.ones(len(fit_n), dtype=np.float32),
            fit_u,
            fit_v,
            fit_u * fit_u,
            fit_u * fit_v,
            fit_v * fit_v,
        ]
    )
    try:
        coeffs, *_ = np.linalg.lstsq(design * np.sqrt(weights)[:, None], fit_n * np.sqrt(weights), rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(6, dtype=np.float32)
    return coeffs.astype(np.float32)


def evaluate_skin_points(
    local_points: np.ndarray,
    coeffs: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    u = local_points[:, 0]
    v = local_points[:, 1]
    fitted_n = coeffs[0] + coeffs[1] * u + coeffs[2] * v + coeffs[3] * u * u + coeffs[4] * u * v + coeffs[5] * v * v
    return (
        anchor + u[:, None] * tangent_u + v[:, None] * tangent_v + fitted_n[:, None] * normal
    ).astype(np.float32)


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


def local_directions(gravity_direction_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gravity_2d = gravity_direction_2d.astype(np.float32)
    norm = float(np.linalg.norm(gravity_2d))
    if norm <= 1e-8:
        gravity_2d = np.array([0.0, -1.0], dtype=np.float32)
    else:
        gravity_2d /= norm
    lateral_2d = np.array([-gravity_2d[1], gravity_2d[0]], dtype=np.float32)
    gravity_3d = np.array([gravity_2d[0], 0.0, gravity_2d[1]], dtype=np.float32)
    lateral_3d = np.array([lateral_2d[0], 0.0, lateral_2d[1]], dtype=np.float32)
    return gravity_3d, lateral_3d


def frame_state(spec: LesionSpec, frame_index: int, frame_count: int) -> dict[str, float]:
    t = frame_index / max(frame_count - 1, 1)
    adjusted_t = float(np.clip((t - spec.growth_delay) / max(spec.growth_duration, 1e-6), 0.0, 1.0))
    growth_time = float(np.clip(adjusted_t ** spec.growth_power, 0.0, 1.0))
    growth = float(smoothstep(growth_time))
    pedunculation = float(smoothstep((growth_time - 0.10) / 0.72))
    gravity_drive = 0.10 + 0.90 * growth
    gravity_term = spec.gravity_scale * gravity_drive
    neck_release = 0.12 + 0.88 * pedunculation
    flop = float(smoothstep(gravity_drive * neck_release))
    return {
        "growth_t": float(t),
        "adjusted_growth_t": adjusted_t,
        "growth": growth,
        "pedunculation": pedunculation,
        "gravity_drive": gravity_drive,
        "gravity_term": gravity_term,
        "flop": flop,
        "height": float(lerp(0.0012, spec.final_height, growth)),
        "support_radius": float(lerp(spec.support_radius * 0.78, spec.support_radius, growth)),
        "neck_radius": float(lerp(spec.support_radius * 0.52, spec.neck_radius, pedunculation)),
        "bulb_radius": float(lerp(spec.support_radius * 0.34, spec.bulb_radius, pedunculation)),
    }


def centerline(
    s: float,
    spec: LesionSpec,
    state: dict[str, float],
    gravity_3d: np.ndarray,
    lateral_3d: np.ndarray,
) -> np.ndarray:
    height = state["height"]
    flop = state["flop"]
    gravity_drive = state["gravity_drive"]
    upright_y = height * s
    if s <= spec.stalk_fraction:
        q = s / max(spec.stalk_fraction, 1e-6)
        flopped_y = spec.arch_height * math.sin(0.5 * math.pi * float(smoothstep(q)))
    else:
        q = (s - spec.stalk_fraction) / max(1.0 - spec.stalk_fraction, 1e-6)
        flopped_y = lerp(spec.arch_height, spec.distal_center_height, float(smoothstep(q)))
        flopped_y -= spec.sag * gravity_drive * math.sin(math.pi * q)
    flopped_y = max(float(flopped_y), 0.0045)
    y = float(lerp(upright_y, flopped_y, flop))
    gravity_offset = spec.flop_distance * flop * (s**1.30)
    lateral_offset = spec.lateral * spec.support_radius * state["pedunculation"] * math.sin(math.pi * s)
    return gravity_offset * gravity_3d + lateral_offset * lateral_3d + np.array([0.0, y, 0.0], dtype=np.float32)


def radius_profile(s: float, spec: LesionSpec, state: dict[str, float]) -> float:
    support_radius = state["support_radius"]
    neck_radius = state["neck_radius"]
    bulb_radius = state["bulb_radius"]
    pedunculation = state["pedunculation"]
    dome_radius = support_radius * math.sqrt(max(0.0, 1.0 - s**1.72)) * (1.0 - 0.08 * state["growth"])
    attachment_radius = float(lerp(support_radius, neck_radius * 1.20, pedunculation**0.85))
    if s < spec.stalk_fraction:
        q = s / max(spec.stalk_fraction, 1e-6)
        ped_radius = float(lerp(attachment_radius, neck_radius, float(smoothstep(q))))
    else:
        q = (s - spec.stalk_fraction) / max(1.0 - spec.stalk_fraction, 1e-6)
        bulb = math.sin(math.pi * q) ** 0.64
        pear = 1.0 + spec.pear_bias * (1.0 - q) * (1.0 - 0.35 * q)
        ped_radius = bulb_radius * bulb * pear + neck_radius * (1.0 - q) ** 2
    ped_radius *= 1.0 + 0.07 * state["gravity_drive"] * math.sin(math.pi * s)
    return max(float(lerp(dome_radius, ped_radius, pedunculation)), min(0.0011, state["height"] * 0.09))


def mature_ring_axes(
    s: float,
    spec: LesionSpec,
    state: dict[str, float],
    gravity_3d: np.ndarray,
    lateral_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-3
    p0 = centerline(max(0.0, s - eps), spec, state, gravity_3d, lateral_3d)
    p1 = centerline(min(1.0, s + eps), spec, state, gravity_3d, lateral_3d)
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
    spec: LesionSpec,
    frame_index: int,
    frame_count: int,
    gravity_direction_2d: np.ndarray,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    gravity_3d, lateral_3d = local_directions(gravity_direction_2d)
    state = frame_state(spec, frame_index, frame_count)
    ring_s = np.linspace(0.0, 0.97, radial_segments, dtype=np.float32)
    vertices: list[list[float]] = [[0.0, 0.0, 0.0]]
    radial_weight: list[float] = [1.0]
    for s_raw in ring_s:
        s = float(s_raw)
        center = centerline(s, spec, state, gravity_3d, lateral_3d)
        axis_a, axis_b = mature_ring_axes(s, spec, state, gravity_3d, lateral_3d)
        radius = radius_profile(s, spec, state)
        dome_center = np.array([0.0, state["height"] * (s**0.78), 0.0], dtype=np.float32)
        dome_radius = state["support_radius"] * math.sqrt(max(0.0, 1.0 - s**1.72))
        dome_axis_a = lateral_3d
        dome_axis_b = gravity_3d
        for step in range(angular_segments):
            theta = 2.0 * math.pi * step / angular_segments + spec.twist * state["pedunculation"] * s
            lobe = 1.0 + spec.lobe_amp * state["pedunculation"] * math.sin(3.0 * theta + 5.5 * s)
            mature_vertex = center + max(radius * lobe, 0.0008) * (
                math.cos(theta) * axis_a + math.sin(theta) * axis_b
            )
            dome_vertex = dome_center + dome_radius * (math.cos(theta) * dome_axis_a + math.sin(theta) * dome_axis_b)
            vertex = (1.0 - state["pedunculation"]) * dome_vertex + state["pedunculation"] * mature_vertex
            if vertex[1] < 0.0:
                vertex[1] = 0.0
            vertices.append(vertex.astype(float).tolist())
            radial_weight.append(float(np.clip(radius / max(spec.support_radius, 1e-6), 0.0, 1.6)))
    top_center = centerline(1.0, spec, state, gravity_3d, lateral_3d)
    dome_top = np.array([0.0, state["height"], 0.0], dtype=np.float32)
    top = (1.0 - state["pedunculation"]) * dome_top + state["pedunculation"] * top_center
    top[1] = max(float(top[1]), 0.0)
    vertices.append(top.astype(float).tolist())
    radial_weight.append(0.0)
    xyz = np.asarray(vertices, dtype=np.float32)
    faces = mesh_faces_for_rings(radial_segments, angular_segments)
    return xyz, remove_degenerate_faces(xyz, faces), np.asarray(radial_weight, dtype=np.float32), state


def local_to_world(
    local_xyz: np.ndarray,
    coeffs: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    local_points = local_xyz[:, [0, 2]].astype(np.float32)
    heights = np.maximum(local_xyz[:, 1], 0.0).astype(np.float32)
    skin_points = evaluate_skin_points(local_points, coeffs, anchor, normal, tangent_u, tangent_v)
    return (skin_points + heights[:, None] * normal).astype(np.float32)


def lesion_vertex_colors(base_color: np.ndarray, heights: np.ndarray, radial_weight: np.ndarray) -> np.ndarray:
    warm = np.array([201, 126, 95], dtype=np.float32)
    color = 0.76 * base_color.astype(np.float32) + 0.24 * warm
    height_amount = np.clip(heights / max(float(np.max(heights)), 1e-6), 0.0, 1.0)
    rim = np.clip(radial_weight, 0.0, 1.0)
    vertex_rgb = color[None, :] * (0.88 + 0.12 * height_amount[:, None])
    vertex_rgb = vertex_rgb * (0.94 + 0.06 * (1.0 - rim[:, None]))
    return np.clip(vertex_rgb, 0, 255).astype(np.uint8)


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


def make_specs(
    count: int,
    xyz: np.ndarray,
    rgb: np.ndarray,
    normals: np.ndarray,
    seed: int,
) -> list[LesionSpec]:
    rng = np.random.default_rng(seed)
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    x_values = np.linspace(-0.205, 0.205, cols)
    z_values = np.linspace(0.83, 1.43, rows)
    specs: list[LesionSpec] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        jitter_x = float(rng.uniform(-0.012, 0.012))
        jitter_z = float(rng.uniform(-0.018, 0.018))
        target_x = float(x_values[col] + jitter_x)
        target_z = float(z_values[row] + jitter_z)
        target_index = pick_back_target_vertex(xyz, target_x, target_z, window=0.045)
        anchor, normal, _tangent_u, _tangent_v = target_basis_on_back(xyz, normals, target_index)
        base_color = rgb[target_index].astype(np.float32)
        color = np.clip(0.72 * base_color + 0.28 * np.array([202, 125, 94], dtype=np.float32), 0, 255).astype(np.uint8)
        scale = float(rng.uniform(0.0, 1.0))
        support_radius = float(rng.uniform(0.0115, 0.0225) * (1.0 - 0.14 * row / max(rows - 1, 1)))
        final_height = float(rng.uniform(0.036, 0.082) * (0.88 + 0.32 * scale))
        neck_radius = float(support_radius * rng.uniform(0.25, 0.42))
        bulb_radius = float(support_radius * rng.uniform(0.72, 1.18))
        stalk_fraction = float(rng.uniform(0.34, 0.56))
        gravity_scale = float(rng.uniform(6.2, 12.5))
        flop_distance = float(rng.uniform(0.018, 0.055) + 0.33 * final_height)
        arch_height = float(rng.uniform(0.010, 0.030) + 0.12 * final_height)
        distal_center_height = float(rng.uniform(0.0045, 0.020) + 0.10 * final_height)
        sag = float(rng.uniform(0.004, 0.015) + 0.05 * final_height)
        lateral = float(rng.uniform(-0.60, 0.60))
        twist = float(rng.uniform(-0.75, 0.95))
        lobe_amp = float(rng.uniform(0.004, 0.050))
        pear_bias = float(rng.uniform(0.16, 0.58))
        growth_delay = float(rng.uniform(0.0, 0.24))
        growth_duration = float(rng.uniform(0.58, 0.95))
        growth_power = float(rng.uniform(0.68, 1.48))
        growth_rate = float(1.0 / growth_duration)
        specs.append(
            LesionSpec(
                lesion_id=f"lesion_{index:03d}",
                target_x=float(anchor[0]),
                target_y=float(anchor[1]),
                target_z=float(anchor[2]),
                target_vertex_index=target_index,
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
                growth_duration=growth_duration,
                growth_power=growth_power,
                growth_rate=growth_rate,
                color_rgb=(int(color[0]), int(color[1]), int(color[2])),
            )
        )
    return specs


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
    for file_path in (
        data_root / "lesion_frame_vertices.npz",
        data_root / "lesion_parameters.csv",
        data_root / "frame_metrics.csv",
        data_root / "final_metrics.csv",
        data_root / "manifest.json",
        visualization_root / "manifest.json",
    ):
        if file_path.exists():
            file_path.unlink()
    return data_root, visualization_root


def combine_meshes(meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_parts = []
    face_parts = []
    rgb_parts = []
    offset = 0
    for mesh_xyz, mesh_faces, mesh_rgb in meshes:
        xyz_parts.append(mesh_xyz.astype(np.float32))
        face_parts.append(mesh_faces.astype(np.int32) + offset)
        rgb_parts.append(mesh_rgb.astype(np.uint8))
        offset += len(mesh_xyz)
    return np.vstack(xyz_parts), np.vstack(face_parts), np.vstack(rgb_parts)


def write_parameters_csv(path: Path, specs: list[LesionSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(specs[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            row = asdict(spec)
            row["color_rgb"] = json.dumps(row["color_rgb"])
            writer.writerow(row)


def write_metrics_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_body_trace(body_xyz: np.ndarray, body_faces: np.ndarray, body_rgb: np.ndarray) -> go.Mesh3d:
    return go.Mesh3d(
        x=np.round(body_xyz[:, 0], 5),
        y=np.round(body_xyz[:, 1], 5),
        z=np.round(body_xyz[:, 2], 5),
        i=body_faces[:, 0],
        j=body_faces[:, 1],
        k=body_faces[:, 2],
        vertexcolor=rgb_strings(body_rgb),
        opacity=0.78,
        flatshading=False,
        lighting=dict(ambient=0.88, diffuse=0.62, specular=0.018, roughness=0.96),
        hoverinfo="skip",
        name="body",
        showlegend=False,
    )


def build_lesion_trace(
    vertices: np.ndarray,
    faces: np.ndarray,
    spec: LesionSpec,
    show_legend: bool = False,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=np.round(vertices[:, 0], 5),
        y=np.round(vertices[:, 1], 5),
        z=np.round(vertices[:, 2], 5),
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=rgb_string(spec.color_rgb),
        opacity=0.98,
        flatshading=False,
        lighting=dict(ambient=0.90, diffuse=0.66, specular=0.035, roughness=0.91),
        hovertemplate=(
            f"{spec.lesion_id}<br>"
            f"growth rate {spec.growth_rate:.2f}<br>"
            f"delay {spec.growth_delay:.2f}<br>"
            f"gravity {spec.gravity_scale:.1f}<extra></extra>"
        ),
        name=spec.lesion_id,
        showlegend=show_legend,
    )


def build_lesion_frame_update(vertices: np.ndarray) -> go.Mesh3d:
    return go.Mesh3d(
        x=np.round(vertices[:, 0], 4),
        y=np.round(vertices[:, 1], 4),
        z=np.round(vertices[:, 2], 4),
    )


def camera_for_angle(angle: float) -> dict[str, dict[str, float]]:
    radius = 2.65
    return {
        "eye": {"x": radius * math.sin(angle), "y": -radius * math.cos(angle), "z": 0.55},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }


def make_progression_figure(
    body_xyz: np.ndarray,
    body_faces: np.ndarray,
    body_rgb: np.ndarray,
    lesion_vertices: np.ndarray,
    lesion_faces: np.ndarray,
    specs: list[LesionSpec],
    frame_metrics_by_index: list[list[dict[str, object]]],
) -> go.Figure:
    frame_count = lesion_vertices.shape[1]
    data: list[go.BaseTraceType] = [build_body_trace(body_xyz, body_faces, body_rgb)]
    lesion_trace_indices = []
    for lesion_index, spec in enumerate(specs):
        lesion_trace_indices.append(len(data))
        data.append(build_lesion_trace(lesion_vertices[lesion_index, 0], lesion_faces, spec))

    frames = []
    for frame_index in range(frame_count):
        frame_data = []
        growth_values = []
        gravity_values = []
        for lesion_index, spec in enumerate(specs):
            frame_data.append(build_lesion_frame_update(lesion_vertices[lesion_index, frame_index]))
            metrics = frame_metrics_by_index[frame_index][lesion_index]
            growth_values.append(float(metrics["adjusted_growth_t"]))
            gravity_values.append(float(metrics["gravity_term"]))
        frames.append(
            go.Frame(
                name=f"{frame_index + 1:03d}",
                data=frame_data,
                traces=lesion_trace_indices,
                layout=go.Layout(
                    title_text=(
                        f"100 back lesions - frame {frame_index + 1:03d}/{frame_count} - "
                        f"mean growth {np.mean(growth_values):.2f} - mean gravity {np.mean(gravity_values):.1f}"
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
    pad = np.array([0.08, 0.08, 0.04], dtype=np.float32)
    xyz_min = body_xyz.min(axis=0) - pad
    xyz_max = body_xyz.max(axis=0) + pad
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title=dict(text=f"100 back lesions - frame 001/{frame_count}", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[float(xyz_min[0]), float(xyz_max[0])]),
            yaxis=dict(visible=False, range=[float(xyz_min[1]), float(xyz_max[1])]),
            zaxis=dict(visible=False, range=[float(xyz_min[2]), float(xyz_max[2])]),
            bgcolor="rgb(244,246,249)",
            aspectmode="data",
            camera=camera_for_angle(0.0),
        ),
        width=1200,
        height=860,
        margin=dict(l=0, r=0, t=58, b=0),
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
                                "frame": {"duration": 100, "redraw": True},
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


def compact_payload(value: object) -> object:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [compact_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_payload(item) for key, item in value.items()}
    return value


def write_code_free_notebook(notebook_path: Path, figure: go.Figure) -> None:
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    payload = compact_payload(payload)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_code_cell(
                source="",
                execution_count=None,
                metadata={"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
                outputs=[
                    nbf.v4.new_output(
                        output_type="display_data",
                        data={
                            "application/vnd.plotly.v1+json": payload,
                            "text/plain": "<Plotly Figure: multiple lesion physics progression>",
                        },
                        metadata={},
                    )
                ],
            )
        ],
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", pygments_lexer="ipython3"),
        ),
    )
    nbf.write(notebook, notebook_path)


def render_progression_gif(figure: go.Figure, gif_path: Path, frame_count: int, gif_frames: int, fps: int) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = np.unique(np.linspace(0, frame_count - 1, gif_frames, dtype=np.int32))
    working = go.Figure(data=figure.data, layout=figure.layout)
    images = []
    with tempfile.TemporaryDirectory(prefix=f"{gif_path.stem}_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for output_index, frame_index in enumerate(sample_indices):
            frame = figure.frames[int(frame_index)]
            if frame.data:
                for trace_index, trace_update in zip(frame.traces, frame.data):
                    working.data[int(trace_index)].update(trace_update)
            working.update_layout(title_text=frame.layout.title.text if frame.layout and frame.layout.title else None)
            angle = 2.0 * math.pi * output_index / max(len(sample_indices), 1)
            working.update_layout(scene_camera=camera_for_angle(float(angle)))
            png_path = tmp_dir / f"frame_{output_index:03d}.png"
            working.write_image(png_path, width=1050, height=780, scale=1)
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=1 / fps, loop=0)


def build_dataset(args: argparse.Namespace) -> None:
    data_root, visualization_root = clear_output_dirs(DATASET_ROOT)
    mesh_path = HSR_MESH_ROOT / f"{args.scan_id}_closed_textured_mesh.ply"
    base_xyz, base_faces, base_rgb = read_colored_ply(mesh_path)
    base_normals = compute_vertex_normals(base_xyz, base_faces)
    body_plot_xyz, body_plot_faces, body_plot_rgb = simplify_mesh(base_xyz, base_faces, base_rgb, args.body_plot_faces)

    specs = make_specs(args.lesion_count, base_xyz, base_rgb, base_normals, seed=args.seed)
    lesion_faces = mesh_faces_for_rings(args.radial_segments, args.angular_segments)
    lesion_faces = remove_degenerate_faces(
        np.zeros((1 + args.radial_segments * args.angular_segments + 1, 3), dtype=np.float32), lesion_faces, eps=-1.0
    )
    vertex_count = 1 + args.radial_segments * args.angular_segments + 1
    lesion_vertices = np.zeros((args.lesion_count, args.frame_count, vertex_count, 3), dtype=np.float32)
    lesion_colors = np.zeros((args.lesion_count, vertex_count, 3), dtype=np.uint8)
    frame_records: list[dict[str, object]] = []
    final_records: list[dict[str, object]] = []
    frame_metrics_by_index: list[list[dict[str, object]]] = [[] for _ in range(args.frame_count)]
    final_meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for lesion_index, spec in enumerate(specs):
        anchor, normal, tangent_u, tangent_v = target_basis_on_back(base_xyz, base_normals, spec.target_vertex_index)
        coeffs = fit_skin_quadratic(base_xyz, base_faces, anchor, normal, tangent_u, tangent_v, spec.support_radius)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_direction_2d = np.array([float(gravity_world @ tangent_u), float(gravity_world @ tangent_v)], dtype=np.float32)
        base_color = base_rgb[spec.target_vertex_index]

        last_local = None
        last_radial_weight = None
        last_state = None
        for frame_index in range(args.frame_count):
            local_xyz, local_faces, radial_weight, state = build_local_shape(
                spec,
                frame_index,
                args.frame_count,
                gravity_direction_2d,
                args.radial_segments,
                args.angular_segments,
            )
            world_xyz = local_to_world(local_xyz, coeffs, anchor, normal, tangent_u, tangent_v)
            lesion_vertices[lesion_index, frame_index] = world_xyz
            metrics = {
                "lesion_id": spec.lesion_id,
                "frame_index": frame_index,
                "growth_t": state["growth_t"],
                "adjusted_growth_t": state["adjusted_growth_t"],
                "growth": state["growth"],
                "pedunculation": state["pedunculation"],
                "gravity_drive": state["gravity_drive"],
                "gravity_term": state["gravity_term"],
                "flop": state["flop"],
                "peak_height_m": float(np.max(local_xyz[:, 1])),
                "mean_height_m": float(np.mean(local_xyz[:, 1])),
                "max_radial_extent_m": float(np.max(np.linalg.norm(local_xyz[:, [0, 2]], axis=1))),
                "surface_area_m2": surface_area(local_xyz, local_faces),
            }
            frame_records.append(metrics)
            frame_metrics_by_index[frame_index].append(metrics)
            last_local = local_xyz
            last_radial_weight = radial_weight
            last_state = state

        assert last_local is not None and last_radial_weight is not None and last_state is not None
        final_world = lesion_vertices[lesion_index, -1]
        final_heights = np.maximum(last_local[:, 1], 0.0)
        final_rgb = lesion_vertex_colors(base_color, final_heights, last_radial_weight)
        lesion_colors[lesion_index] = final_rgb
        final_meshes.append((final_world, lesion_faces, final_rgb))
        lesion_path = data_root / "lesion_meshes" / "final" / f"{args.scan_id}_{spec.lesion_id}_final_lesion.ply"
        write_colored_ply(lesion_path, final_world, lesion_faces, final_rgb)
        final_records.append(
            {
                **asdict(spec),
                "color_rgb": json.dumps(spec.color_rgb),
                "model": MODEL_NAME,
                "final_lesion_mesh": dataset_relative(lesion_path),
                "frame_index": args.frame_count - 1,
                "final_adjusted_growth_t": last_state["adjusted_growth_t"],
                "final_growth": last_state["growth"],
                "final_pedunculation": last_state["pedunculation"],
                "final_gravity_term": last_state["gravity_term"],
                "final_flop": last_state["flop"],
                "final_peak_height_m": float(np.max(last_local[:, 1])),
                "final_surface_area_m2": surface_area(last_local, lesion_faces),
            }
        )

    npz_path = data_root / "lesion_frame_vertices.npz"
    np.savez_compressed(
        npz_path,
        lesion_vertices=lesion_vertices,
        lesion_faces=lesion_faces.astype(np.int32),
        lesion_colors=lesion_colors,
        body_plot_vertices=body_plot_xyz,
        body_plot_faces=body_plot_faces,
        body_plot_colors=body_plot_rgb,
        count_presets=np.asarray(COUNT_PRESETS, dtype=np.int32),
        frame_count=np.asarray(args.frame_count, dtype=np.int32),
        model=np.asarray(MODEL_NAME),
        scan_id=np.asarray(args.scan_id),
    )

    write_parameters_csv(data_root / "lesion_parameters.csv", specs)
    write_metrics_csv(data_root / "frame_metrics.csv", frame_records)
    write_metrics_csv(data_root / "final_metrics.csv", final_records)

    combined_mesh_paths: dict[str, str] = {}
    for count in COUNT_PRESETS:
        if count > args.lesion_count:
            continue
        combined_xyz, combined_faces, combined_rgb = combine_meshes(
            [(base_xyz, base_faces, base_rgb)] + final_meshes[:count]
        )
        combined_path = (
            data_root
            / "final_combined_meshes"
            / f"{args.scan_id}_all_{count:03d}_back_lesions_final_hsr_lesion.ply"
        )
        write_colored_ply(combined_path, combined_xyz, combined_faces, combined_rgb)
        combined_mesh_paths[str(count)] = dataset_relative(combined_path)

    figure = make_progression_figure(
        body_plot_xyz,
        body_plot_faces,
        body_plot_rgb,
        lesion_vertices,
        lesion_faces,
        specs,
        frame_metrics_by_index,
    )
    notebook_path = visualization_root / "plotly" / "multiple_lesion_physics_progression.ipynb"
    gif_path = visualization_root / "gifs" / "multiple_lesion_physics_100_lesions_progression.gif"
    write_code_free_notebook(notebook_path, figure)
    render_progression_gif(figure, gif_path, args.frame_count, args.gif_frames, args.fps)

    metadata = {
        "dataset": "multiple_lesion_physics",
        "scan_id": args.scan_id,
        "source_mesh": root_relative(mesh_path),
        "model": MODEL_NAME,
        "description": (
            "Continuous gravity is applied at every growth step. Each lesion has a different growth delay, "
            "duration, power, gravity scale, stalk size, bulb size, and flop distance."
        ),
        "body_visualization": "continuous quadric-simplified textured body mesh; no sparse/dotted full-body rendering",
        "lesion_count": args.lesion_count,
        "frame_count": args.frame_count,
        "radial_segments": args.radial_segments,
        "angular_segments": args.angular_segments,
        "count_presets": list(COUNT_PRESETS),
        "data_npz": dataset_relative(npz_path),
        "lesion_parameters": "data/lesion_parameters.csv",
        "frame_metrics": "data/frame_metrics.csv",
        "final_metrics": "data/final_metrics.csv",
        "final_combined_meshes": combined_mesh_paths,
        "plotly_notebook": dataset_relative(notebook_path),
        "gif": dataset_relative(gif_path),
        "lesions": [asdict(spec) for spec in specs],
    }
    metadata_path = data_root / "metadata" / f"{args.scan_id}_multiple_lesion_physics_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    viz_manifest = {
        "dataset": "multiple_lesion_physics",
        "metadata": dataset_relative(metadata_path),
        "plotly_notebook": dataset_relative(notebook_path),
        "gif": dataset_relative(gif_path),
        "data_npz": dataset_relative(npz_path),
        "lesion_count": args.lesion_count,
        "frame_count": args.frame_count,
        "count_presets": list(COUNT_PRESETS),
        "final_combined_meshes": combined_mesh_paths,
    }
    (data_root / "manifest.json").write_text(json.dumps(viz_manifest, indent=2), encoding="utf-8")
    (visualization_root / "manifest.json").write_text(json.dumps(viz_manifest, indent=2), encoding="utf-8")

    print(DATASET_ROOT)
    print(npz_path)
    print(notebook_path)
    print(gif_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", default="HSR0018-Body-070")
    parser.add_argument("--lesion-count", type=int, default=100)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--radial-segments", type=int, default=6)
    parser.add_argument("--angular-segments", type=int, default=16)
    parser.add_argument("--body-plot-faces", type=int, default=7200)
    parser.add_argument("--gif-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()

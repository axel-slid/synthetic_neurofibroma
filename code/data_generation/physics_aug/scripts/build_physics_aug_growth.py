#!/usr/bin/env python3
"""Build a physics-inspired lesion growth augmentation on a closed HSR scan."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from plotly.utils import PlotlyJSONEncoder
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "data" / "synthetic" / "single_lesion" / "body_parts" / "physics_aug_growth"
VISUALIZATION_ROOT = ROOT / "data" / "synthetic" / "single_lesion" / "visualization" / "physics_aug_growth"
HSR_MESH_ROOT = ROOT / "data" / "hsr" / "visualizations" / "meshes"


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class GrowthFrame:
    index: int
    phase: str
    phase_slug: str
    growth_t: float
    height: float
    support_radius: float
    roundness: float
    lesion_blend: float
    gravity_scale: float
    shape_memory: float
    contact_adhesion: float


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, stop: float, amount: float) -> float:
    return start + (stop - start) * amount


def growth_schedule(frame_count: int) -> list[GrowthFrame]:
    if frame_count < 14:
        raise ValueError("frame_count must be at least 14 to show all growth phases")

    frames = []
    for index in range(frame_count):
        growth_t = index / (frame_count - 1)
        if growth_t <= 0.12:
            phase = "flat"
            phase_slug = "flat"
            amount = smoothstep(growth_t / 0.12)
            height = lerp(0.0015, 0.008, amount)
            support_radius = lerp(0.041, 0.052, amount)
            roundness = lerp(3.4, 2.8, amount)
            lesion_blend = lerp(0.32, 0.38, amount)
            gravity_scale = lerp(0.05, 0.18, amount)
            shape_memory = lerp(0.080, 0.072, amount)
            contact_adhesion = 0.0
        elif growth_t <= 0.30:
            phase = "sessile growth"
            phase_slug = "sessile_growth"
            amount = smoothstep((growth_t - 0.12) / 0.18)
            height = lerp(0.008, 0.038, amount)
            support_radius = lerp(0.052, 0.046, amount)
            roundness = lerp(2.8, 1.75, amount)
            lesion_blend = lerp(0.38, 0.50, amount)
            gravity_scale = lerp(0.18, 0.45, amount)
            shape_memory = lerp(0.072, 0.060, amount)
            contact_adhesion = 0.0
        elif growth_t <= 0.48:
            phase = "globular"
            phase_slug = "globular"
            amount = smoothstep((growth_t - 0.30) / 0.18)
            height = lerp(0.038, 0.071, amount)
            support_radius = lerp(0.046, 0.034, amount)
            roundness = lerp(1.75, 0.82, amount)
            lesion_blend = lerp(0.50, 0.62, amount)
            gravity_scale = lerp(0.45, 0.90, amount)
            shape_memory = lerp(0.060, 0.048, amount)
            contact_adhesion = 0.04 * amount
        elif growth_t <= 0.66:
            phase = "pedunculated"
            phase_slug = "pedunculated"
            amount = smoothstep((growth_t - 0.48) / 0.18)
            height = lerp(0.071, 0.092, amount)
            support_radius = lerp(0.034, 0.038, amount)
            roundness = lerp(0.82, 0.70, amount)
            lesion_blend = lerp(0.62, 0.72, amount)
            gravity_scale = lerp(0.90, 2.40, amount)
            shape_memory = lerp(0.048, 0.032, amount)
            contact_adhesion = lerp(0.04, 0.18, amount)
        elif growth_t <= 0.84:
            phase = "gravity plop"
            phase_slug = "gravity_plop"
            amount = smoothstep((growth_t - 0.66) / 0.18)
            height = lerp(0.092, 0.056, amount)
            support_radius = lerp(0.038, 0.047, amount)
            roundness = lerp(0.70, 1.45, amount)
            lesion_blend = lerp(0.72, 0.60, amount)
            gravity_scale = lerp(2.40, 5.20, amount)
            shape_memory = lerp(0.032, 0.018, amount)
            contact_adhesion = lerp(0.18, 0.62, amount)
        else:
            phase = "sessile settle"
            phase_slug = "sessile_settle"
            amount = smoothstep((growth_t - 0.84) / 0.16)
            height = lerp(0.056, 0.044, amount)
            support_radius = lerp(0.047, 0.049, amount)
            roundness = lerp(1.45, 1.70, amount)
            lesion_blend = lerp(0.60, 0.52, amount)
            gravity_scale = lerp(5.20, 1.20, amount)
            shape_memory = lerp(0.018, 0.052, amount)
            contact_adhesion = lerp(0.62, 0.28, amount)

        frames.append(
            GrowthFrame(
                index=index,
                phase=phase,
                phase_slug=phase_slug,
                growth_t=float(growth_t),
                height=float(height),
                support_radius=float(support_radius),
                roundness=float(roundness),
                lesion_blend=float(lesion_blend),
                gravity_scale=float(gravity_scale),
                shape_memory=float(shape_memory),
                contact_adhesion=float(contact_adhesion),
            )
        )
    return frames


def read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    face = ply["face"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
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
    normals[~valid] = np.array([0.0, 1.0, 0.0])
    return normals.astype(np.float32)


def pick_target_vertex(
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
            return int(candidates[np.argmax(xyz[candidates, 1])])
    target = np.array([target_x, xyz[:, 1].max(), target_z], dtype=np.float32)
    return int(np.argmin(np.sum((xyz - target) ** 2, axis=1)))


def target_basis(xyz: np.ndarray, normals: np.ndarray, target_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor = xyz[target_index].astype(np.float32)
    normal = normals[target_index].astype(np.float32)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-8:
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        normal = normal / normal_length

    body_center = xyz.mean(axis=0).astype(np.float32)
    if float(np.dot(normal, anchor - body_center)) < 0.0:
        normal = -normal
    if normal[1] < 0.0:
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
    return anchor, normal, tangent_u, tangent_v


def remove_degenerate_faces(xyz: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(faces) == 0:
        return faces
    triangles = xyz[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2.0
    return faces[areas > eps]


def lesion_template(
    frame: GrowthFrame,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_points = [np.array([0.0, 0.0], dtype=np.float32)]
    profile_heights = [frame.height]
    radial_weight = [0.0]
    for ring in range(1, radial_segments + 1):
        rho = frame.support_radius * ring / radial_segments
        q = np.clip(rho / frame.support_radius, 0.0, 1.0)
        height = frame.height * np.power(np.clip(1.0 - q * q, 0.0, 1.0), frame.roundness)
        for step in range(angular_segments):
            theta = 2.0 * np.pi * step / angular_segments
            local_points.append(np.array([rho * np.cos(theta), rho * np.sin(theta)], dtype=np.float32))
            profile_heights.append(float(height))
            radial_weight.append(float(q))

    faces = []
    for step in range(angular_segments):
        faces.append([0, 1 + step, 1 + ((step + 1) % angular_segments)])
    for ring in range(1, radial_segments):
        prev_start = 1 + (ring - 1) * angular_segments
        next_start = 1 + ring * angular_segments
        for step in range(angular_segments):
            a = prev_start + step
            b = prev_start + ((step + 1) % angular_segments)
            c = next_start + step
            d = next_start + ((step + 1) % angular_segments)
            faces.append([a, c, b])
            faces.append([b, c, d])

    return (
        np.asarray(local_points, dtype=np.float32),
        np.asarray(profile_heights, dtype=np.float32),
        np.asarray(radial_weight, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def candidate_surface_faces(
    xyz: np.ndarray,
    faces: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    radius: float,
) -> np.ndarray:
    centroids = xyz[faces].mean(axis=1)
    offsets = centroids - anchor
    local_u = offsets @ tangent_u
    local_v = offsets @ tangent_v
    local_n = offsets @ normal
    radial = np.sqrt(local_u * local_u + local_v * local_v)
    mask = (radial <= max(0.12, 2.65 * radius)) & (np.abs(local_n) <= max(0.055, 1.45 * radius))
    if int(mask.sum()) < 12:
        mask = radial <= max(0.12, 3.2 * radius)
    if int(mask.sum()) < 12:
        order = np.argsort(radial)
        mask = np.zeros(len(faces), dtype=bool)
        mask[order[: min(250, len(order))]] = True
    return mask


def sample_skin_and_color(
    local_points: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    skin_rgb: np.ndarray,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_mask = candidate_surface_faces(
        skin_vertices, skin_faces, anchor, normal, tangent_u, tangent_v, support_radius
    )
    candidate_faces = skin_faces[candidate_mask]
    skin_triangles = skin_vertices[candidate_faces]
    skin_colors = skin_rgb[candidate_faces].astype(np.float32)
    skin_offsets = skin_triangles - anchor
    skin_triangle_plane = np.stack([skin_offsets @ tangent_u, skin_offsets @ tangent_v], axis=2)
    skin_plane = skin_triangle_plane.mean(axis=1)

    sampled = np.empty((len(local_points), 3), dtype=np.float32)
    sampled_rgb = np.empty((len(local_points), 3), dtype=np.float32)
    for start in range(0, len(local_points), 384):
        stop = start + 384
        delta = local_points[start:stop, None, :] - skin_plane[None, :, :]
        nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        targets = local_points[start:stop]
        tri_plane = skin_triangle_plane[nearest]
        tri_xyz = skin_triangles[nearest]
        tri_rgb = skin_colors[nearest]

        p0 = tri_plane[:, 0]
        edge0 = tri_plane[:, 1] - p0
        edge1 = tri_plane[:, 2] - p0
        target_offset = targets - p0
        d00 = np.sum(edge0 * edge0, axis=1)
        d01 = np.sum(edge0 * edge1, axis=1)
        d11 = np.sum(edge1 * edge1, axis=1)
        d20 = np.sum(target_offset * edge0, axis=1)
        d21 = np.sum(target_offset * edge1, axis=1)
        denom = d00 * d11 - d01 * d01

        weights = np.empty((len(targets), 3), dtype=np.float32)
        valid = np.abs(denom) > 1e-12
        weights[~valid] = 1.0 / 3.0
        v = np.zeros(len(targets), dtype=np.float32)
        w = np.zeros(len(targets), dtype=np.float32)
        v[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
        w[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
        weights[valid, 1] = v[valid]
        weights[valid, 2] = w[valid]
        weights[valid, 0] = 1.0 - v[valid] - w[valid]
        weights[valid] = np.clip(weights[valid], 0.0, None)
        weight_sums = weights.sum(axis=1, keepdims=True)
        empty = weight_sums[:, 0] <= 1e-8
        weights[empty] = 1.0 / 3.0
        weight_sums[empty] = 1.0
        weights /= weight_sums
        sampled[start:stop] = np.sum(tri_xyz * weights[:, :, None], axis=1)
        sampled_rgb[start:stop] = np.sum(tri_rgb * weights[:, :, None], axis=1)
    return sampled, sampled_rgb


def lesion_colors(skin_rgb: np.ndarray, profile: np.ndarray, radial_weight: np.ndarray, frame: GrowthFrame) -> np.ndarray:
    return np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(profile), 1))


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0).astype(np.int32)


def simulate_soft_body_surface(
    rest_local_xyz: np.ndarray,
    faces: np.ndarray,
    pin_mask: np.ndarray,
    frame: GrowthFrame,
    gravity_local: np.ndarray,
    support_radius: float,
) -> np.ndarray:
    pos = rest_local_xyz.astype(np.float32).copy()
    rest = rest_local_xyz.astype(np.float32)
    pins = pin_mask.astype(bool)
    free = ~pins
    edges = mesh_edges(faces)
    edge_a = edges[:, 0]
    edge_b = edges[:, 1]
    rest_lengths = np.linalg.norm(rest[edge_b] - rest[edge_a], axis=1)
    rest_lengths = np.maximum(rest_lengths, 1e-6)

    gravity = gravity_local.astype(np.float32)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm > 1e-8:
        gravity /= gravity_norm
    dt_gravity = 0.000032 * frame.gravity_scale
    spring_relaxation = 0.18
    iterations = 90
    constraint_passes = 2
    tangent_limit = max(0.13, support_radius * 3.4)
    height_limit = max(0.10, frame.height * 1.35)

    pin_indices = np.flatnonzero(pins)
    pos[pin_indices] = rest[pin_indices]
    low_side_threshold = -support_radius * (0.20 + 0.55 * frame.contact_adhesion)
    for _ in range(iterations):
        pos[free] += gravity * dt_gravity
        pos[free] += (rest[free] - pos[free]) * frame.shape_memory

        for _pass in range(constraint_passes):
            delta = pos[edge_b] - pos[edge_a]
            lengths = np.linalg.norm(delta, axis=1)
            valid = lengths > 1e-8
            correction = np.zeros_like(delta)
            correction[valid] = (
                delta[valid]
                * ((lengths[valid] - rest_lengths[valid]) / lengths[valid] * spring_relaxation)[:, None]
            )

            both_free = free[edge_a] & free[edge_b]
            a_free = free[edge_a] & pins[edge_b]
            b_free = pins[edge_a] & free[edge_b]

            accum = np.zeros_like(pos)
            counts = np.zeros(len(pos), dtype=np.float32)
            np.add.at(accum, edge_a[both_free], correction[both_free] * 0.5)
            np.add.at(accum, edge_b[both_free], -correction[both_free] * 0.5)
            np.add.at(counts, edge_a[both_free], 1.0)
            np.add.at(counts, edge_b[both_free], 1.0)
            np.add.at(accum, edge_a[a_free], correction[a_free])
            np.add.at(accum, edge_b[b_free], -correction[b_free])
            np.add.at(counts, edge_a[a_free], 1.0)
            np.add.at(counts, edge_b[b_free], 1.0)
            movable = free & (counts > 0)
            pos[movable] += accum[movable] / counts[movable, None]

            pos[pin_indices] = rest[pin_indices]

        below_skin = free & (pos[:, 1] < 0.0)
        if np.any(below_skin):
            pos[below_skin, 1] = 0.0
            pos[below_skin, 2] *= 0.992

        if frame.contact_adhesion > 0.0:
            lower_side = free & (pos[:, 2] < low_side_threshold)
            near_skin = lower_side & (pos[:, 1] < max(0.040, frame.height * 0.58))
            if np.any(near_skin):
                pull = 0.010 * frame.contact_adhesion
                pos[near_skin, 1] *= 1.0 - pull
                pos[near_skin, 2] -= 0.00010 * frame.gravity_scale * frame.contact_adhesion

        bad = ~np.isfinite(pos).all(axis=1)
        if np.any(bad):
            pos[bad] = rest[bad]
        pos[:, 0] = np.clip(pos[:, 0], -tangent_limit, tangent_limit)
        pos[:, 2] = np.clip(pos[:, 2], -tangent_limit * 1.55, tangent_limit)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, height_limit)

    pos[:, 1] = np.maximum(pos[:, 1], 0.0)
    pos[pin_indices] = rest[pin_indices]
    return pos


def build_pedunculated_template(
    frame: GrowthFrame,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neck_radius = max(0.010, frame.support_radius * 0.36)
    bulb_radius = frame.support_radius * 0.55
    bend_distance = frame.support_radius * 1.15
    stalk_height = frame.height * 0.48

    rings: list[tuple[float, float]] = [
        (0.000, neck_radius * 1.18),
        (frame.height * 0.09, neck_radius),
        (stalk_height, neck_radius * 0.92),
    ]
    for step in range(1, 8):
        amount = step / 8.0
        height = lerp(stalk_height, frame.height * 0.98, amount)
        bulb = math.sin(math.pi * amount) ** 0.78
        radius = max(0.004, min(bulb_radius, neck_radius * (1.0 - amount) * 0.55 + bulb_radius * bulb))
        rings.append((height, radius))

    vertices: list[list[float]] = []
    for height, radius in rings:
        bend = smoothstep(height / max(frame.height, 1e-6)) * bend_distance
        for step in range(angular_segments):
            theta = 2.0 * np.pi * step / angular_segments
            vertices.append([bend + radius * np.cos(theta), height, radius * np.sin(theta)])

    top_idx = len(vertices)
    vertices.append([bend_distance, frame.height, 0.0])

    faces: list[list[int]] = []
    for ring in range(len(rings) - 1):
        current = ring * angular_segments
        next_ring = (ring + 1) * angular_segments
        for step in range(angular_segments):
            a = current + step
            b = current + ((step + 1) % angular_segments)
            c = next_ring + step
            d = next_ring + ((step + 1) % angular_segments)
            faces.append([a, c, b])
            faces.append([b, c, d])

    last_ring = (len(rings) - 1) * angular_segments
    for step in range(angular_segments):
        faces.append([last_ring + step, top_idx, last_ring + ((step + 1) % angular_segments)])

    xyz = np.asarray(vertices, dtype=np.float32)
    pin_mask = np.zeros(len(xyz), dtype=bool)
    pin_mask[:angular_segments] = True
    face_arr = remove_degenerate_faces(xyz, np.asarray(faces, dtype=np.int32))
    return xyz, face_arr, pin_mask


def build_lesion_mesh(
    frame: GrowthFrame,
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    base_rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    gravity_local: np.ndarray,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.phase_slug in {"pedunculated", "gravity_plop"} and frame.contact_adhesion < 0.55:
        rest_local_xyz, faces, pin_mask = build_pedunculated_template(frame, angular_segments)
        simulated = simulate_soft_body_surface(
            rest_local_xyz,
            faces,
            pin_mask,
            frame,
            gravity_local,
            support_radius=frame.support_radius,
        )
        local_points = simulated[:, [0, 2]].astype(np.float32)
        profile_heights = simulated[:, 1].astype(np.float32)
        radial_weight = np.clip(np.linalg.norm(local_points, axis=1) / max(frame.support_radius, 1e-6), 0.0, 1.0)
    else:
        local_points, profile_heights, radial_weight, faces = lesion_template(frame, radial_segments, angular_segments)
        rest_local_xyz = np.column_stack([local_points[:, 0], profile_heights, local_points[:, 1]]).astype(np.float32)
        pin_mask = np.zeros(len(rest_local_xyz), dtype=bool)
        base_start = 1 + (radial_segments - 1) * angular_segments
        pin_mask[base_start : base_start + angular_segments] = True
        simulated = simulate_soft_body_surface(
            rest_local_xyz,
            faces,
            pin_mask,
            frame,
            gravity_local,
            support_radius=frame.support_radius,
        )
        local_points = simulated[:, [0, 2]].astype(np.float32)
        profile_heights = simulated[:, 1].astype(np.float32)
        radial_weight = np.clip(np.linalg.norm(local_points, axis=1) / max(frame.support_radius, 1e-6), 0.0, 1.0)

    skin_points, skin_colors = sample_skin_and_color(
        local_points,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        base_xyz,
        base_faces,
        base_rgb,
        frame.support_radius,
    )
    points = skin_points + profile_heights[:, None] * normal
    rgb = lesion_colors(skin_colors, profile_heights, radial_weight, frame)

    xyz = points.astype(np.float32)
    face_arr = remove_degenerate_faces(xyz, faces)
    return xyz, face_arr, rgb.astype(np.uint8)


def visible_base_faces(
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    support_radius: float,
    max_height: float,
) -> np.ndarray:
    centroids = base_xyz[base_faces].mean(axis=1)
    offsets = centroids - anchor
    local_u = offsets @ tangent_u
    local_v = offsets @ tangent_v
    local_n = offsets @ normal
    radial = np.sqrt(local_u * local_u + local_v * local_v)
    # Keep the scan surface near the lesion rim so close-up renderings do not expose
    # small background gaps where the independently generated cap meets curved skin.
    covered = (radial <= support_radius * 0.38) & (np.abs(local_n) <= max(0.032, max_height * 2.3))
    if int(covered.sum()) == 0:
        covered = radial <= support_radius * 0.985
    return base_faces[~covered]


def combine_base_and_lesion(
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    base_rgb: np.ndarray,
    lesion_xyz: np.ndarray,
    lesion_faces: np.ndarray,
    lesion_rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    support_radius: float,
    max_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible_faces = visible_base_faces(
        base_xyz,
        base_faces,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        support_radius,
        max_height,
    )
    combined_xyz = np.vstack([base_xyz, lesion_xyz])
    combined_rgb = np.vstack([base_rgb, lesion_rgb])
    combined_faces = np.vstack([visible_faces, lesion_faces + len(base_xyz)])
    return combined_xyz, combined_faces.astype(np.int32), combined_rgb.astype(np.uint8)


def localize_points(
    xyz: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
) -> np.ndarray:
    offsets = xyz - anchor
    return np.column_stack([offsets @ tangent_u, offsets @ normal, offsets @ tangent_v]).astype(np.float32)


def crop_mesh_to_target(
    xyz: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_before: float,
    depth_after: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_centroids = localize_points(xyz[faces].mean(axis=1), anchor, normal, tangent_u, tangent_v)
    mask = (
        (np.abs(local_centroids[:, 0]) <= half_width)
        & (local_centroids[:, 1] >= -depth_before)
        & (local_centroids[:, 1] <= depth_after)
        & (np.abs(local_centroids[:, 2]) <= half_height)
    )
    selected_faces = faces[mask]
    used = np.unique(selected_faces.ravel())
    remap = np.full(len(xyz), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    local_xyz = localize_points(xyz[used], anchor, normal, tangent_u, tangent_v)
    local_faces = remap[selected_faces]
    return local_xyz, local_faces.astype(np.int32), rgb[used]


def measurement_trace(frame: GrowthFrame, half_width: float, half_height: float) -> go.Scatter3d:
    x = half_width * 0.58
    z = -half_height * 0.58
    height_mm = frame.height * 1000.0
    return go.Scatter3d(
        x=[x, x, x],
        y=[0.0, frame.height, frame.height],
        z=[z, z, z],
        mode="lines+markers+text",
        line=dict(color="rgb(20,20,24)", width=8),
        marker=dict(color="rgb(20,20,24)", size=4),
        text=["", "", f"{height_mm:.0f} mm"],
        textposition="middle right",
        textfont=dict(color="rgb(20,20,24)", size=15),
        hoverinfo="skip",
        showlegend=False,
    )


def make_patch_figure(
    local_xyz: np.ndarray,
    local_faces: np.ndarray,
    local_rgb: np.ndarray,
    title: str,
    half_width: float,
    half_height: float,
    depth_after: float,
    frame: GrowthFrame | None = None,
) -> go.Figure:
    traces: list[go.BaseTraceType] = [
        go.Mesh3d(
            x=local_xyz[:, 0],
            y=local_xyz[:, 1],
            z=local_xyz[:, 2],
            i=local_faces[:, 0],
            j=local_faces[:, 1],
            k=local_faces[:, 2],
            vertexcolor=rgb_strings(local_rgb),
            flatshading=False,
            lighting=dict(ambient=0.92, diffuse=0.62, specular=0.035, roughness=0.92),
            hoverinfo="skip",
        )
    ]
    if frame is not None:
        traces.append(measurement_trace(frame, half_width, half_height))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[-half_width, half_width]),
            yaxis=dict(visible=False, range=[-0.025, depth_after]),
            zaxis=dict(visible=False, range=[-half_height, half_height]),
            bgcolor="rgb(244,246,249)",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=0.42, z=1.18),
            camera=dict(
                eye=dict(x=0.38, y=1.54, z=0.18),
                center=dict(x=0.0, y=0.02, z=0.0),
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


def annotate_png(path: Path, frame: GrowthFrame) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    label = f"{frame.index:02d}  {frame.phase}  d={frame.height * 1000:.0f} mm"
    text_box = draw.textbbox((0, 0), label, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.rounded_rectangle((18, 18, 36 + width, 36 + height), radius=7, fill=(255, 255, 255, 224))
    draw.text((27, 25), label, font=font, fill=(36, 39, 44, 255))
    image.save(path)


def render_growth_gif(
    frame_records: list[dict[str, object]],
    gif_path: Path,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="physics_aug_growth_") as tmp_name:
        tmp_dir = Path(tmp_name)
        png_paths = []
        for record in frame_records:
            frame = GrowthFrame(**record["frame"])
            xyz, faces, rgb = read_colored_ply(DATASET_ROOT / record["mesh"])
            local_xyz, local_faces, local_rgb = crop_mesh_to_target(
                xyz,
                faces,
                rgb,
                anchor,
                normal,
                tangent_u,
                tangent_v,
                half_width=half_width,
                half_height=half_height,
                depth_before=0.030,
                depth_after=depth_after,
            )
            title = "Physics augmentation: growth, pedunculation, gravity plop, and settle"
            fig = make_patch_figure(local_xyz, local_faces, local_rgb, title, half_width, half_height, depth_after, frame)
            png_path = tmp_dir / f"frame_{frame.index:03d}.png"
            fig.write_image(png_path, scale=1)
            annotate_png(png_path, frame)
            png_paths.append(png_path)

        images = [imageio.imread(path) for path in png_paths]
        imageio.mimsave(gif_path, images, duration=0.48, loop=0)


def make_notebook_figure(
    frame_records: list[dict[str, object]],
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> go.Figure:
    base_record = frame_records[0]
    base_xyz, base_faces, base_rgb = read_colored_ply(DATASET_ROOT / base_record["mesh"])
    local_xyz, local_faces, local_rgb = crop_mesh_to_target(
        base_xyz,
        base_faces,
        base_rgb,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width=half_width,
        half_height=half_height,
        depth_before=0.030,
        depth_after=depth_after,
    )
    fig = make_patch_figure(
        local_xyz,
        local_faces,
        local_rgb,
        "Interactive HSR physics lesion growth",
        half_width,
        half_height,
        depth_after,
        GrowthFrame(**base_record["frame"]),
    )
    fig.frames = []
    for record in frame_records:
        frame = GrowthFrame(**record["frame"])
        xyz, faces, rgb = read_colored_ply(DATASET_ROOT / record["mesh"])
        local_xyz, local_faces, local_rgb = crop_mesh_to_target(
            xyz,
            faces,
            rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            half_width=half_width,
            half_height=half_height,
            depth_before=0.030,
            depth_after=depth_after,
        )
        trace = go.Mesh3d(
            x=local_xyz[:, 0],
            y=local_xyz[:, 1],
            z=local_xyz[:, 2],
            i=local_faces[:, 0],
            j=local_faces[:, 1],
            k=local_faces[:, 2],
            vertexcolor=rgb_strings(local_rgb),
            flatshading=False,
            lighting=dict(ambient=0.92, diffuse=0.62, specular=0.035, roughness=0.92),
            hoverinfo="skip",
        )
        measure = measurement_trace(frame, half_width, half_height)
        fig.frames += (
            go.Frame(
                name=f"{frame.index:02d} {frame.phase}",
                data=[trace, measure],
                traces=[0, 1],
                layout=go.Layout(
                    title_text=f"Interactive HSR physics lesion growth - {frame.phase} - {frame.height * 1000:.0f} mm"
                ),
            ),
        )

    slider_steps = [
        {
            "args": [
                [frame.name],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
            ],
            "label": frame.name,
            "method": "animate",
        }
        for frame in fig.frames
    ]
    fig.update_layout(
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
                                "frame": {"duration": 430, "redraw": True},
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
        sliders=[
            {
                "active": 0,
                "x": 0.10,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "len": 0.84,
                "steps": slider_steps,
            }
        ],
    )
    return fig


def write_notebook(
    notebook_path: Path,
    frame_records: list[dict[str, object]],
    metadata_record: dict[str, object],
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    records_json = json.dumps(frame_records, indent=2)
    metadata_json = json.dumps(metadata_record, indent=2)
    setup_code = f"""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData

ROOT_CANDIDATES = []
for parent in (Path.cwd(), *Path.cwd().parents):
    ROOT_CANDIDATES.append(parent / 'data' / 'synthetic' / 'single_lesion' / 'body_parts' / 'physics_aug_growth')
    ROOT_CANDIDATES.append(parent / 'data' / 'synthetic' / 'physics_aug_growth')
ROOT_CANDIDATES.append(Path.cwd())
ROOT = next((path for path in ROOT_CANDIDATES if (path / 'data' / 'manifest.json').exists()), ROOT_CANDIDATES[0])
FRAME_RECORDS = {records_json}
METADATA = {metadata_json}
ANCHOR = np.array(METADATA['target_area']['anchor'], dtype=np.float32)
NORMAL = np.array(METADATA['target_area']['normal'], dtype=np.float32)
TANGENT_U = np.array(METADATA['target_area']['tangent_u'], dtype=np.float32)
TANGENT_V = np.array(METADATA['target_area']['tangent_v'], dtype=np.float32)

def _rgb_strings(rgb):
    return [f"rgb({{int(r)}},{{int(g)}},{{int(b)}})" for r, g, b in rgb]

def _read_colored_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    f = ply['face'].data
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    rgb = np.column_stack([v['red'], v['green'], v['blue']]).astype(np.uint8)
    faces = np.vstack(f['vertex_indices']).astype(np.int32)
    return xyz, faces, rgb

def _localize(xyz):
    offsets = xyz - ANCHOR
    return np.column_stack([offsets @ TANGENT_U, offsets @ NORMAL, offsets @ TANGENT_V]).astype(np.float32)

def _crop_mesh(path, half_width={half_width!r}, half_height={half_height!r}, depth_before=0.030, depth_after={depth_after!r}):
    xyz, faces, rgb = _read_colored_ply(path)
    local_centroids = _localize(xyz[faces].mean(axis=1))
    keep = (
        (np.abs(local_centroids[:, 0]) <= half_width)
        & (local_centroids[:, 1] >= -depth_before)
        & (local_centroids[:, 1] <= depth_after)
        & (np.abs(local_centroids[:, 2]) <= half_height)
    )
    faces = faces[keep]
    used = np.unique(faces.ravel())
    remap = np.full(len(xyz), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return _localize(xyz[used]), remap[faces], rgb[used]

def _measurement_trace(frame, half_width={half_width!r}, half_height={half_height!r}):
    x = half_width * 0.58
    z = -half_height * 0.58
    height_mm = frame['height'] * 1000.0
    return go.Scatter3d(
        x=[x, x, x],
        y=[0.0, frame['height'], frame['height']],
        z=[z, z, z],
        mode='lines+markers+text',
        line=dict(color='rgb(20,20,24)', width=8),
        marker=dict(color='rgb(20,20,24)', size=4),
        text=['', '', '{{:.0f}} mm'.format(height_mm)],
        textposition='middle right',
        textfont=dict(color='rgb(20,20,24)', size=15),
        hoverinfo='skip',
        showlegend=False,
    )

def make_growth_figure():
    first = FRAME_RECORDS[0]
    xyz, faces, rgb = _crop_mesh(ROOT / first['mesh'])
    trace = go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=_rgb_strings(rgb),
        flatshading=False,
        lighting=dict(ambient=0.92, diffuse=0.62, specular=0.035, roughness=0.92),
        hoverinfo='skip',
    )
    measure = _measurement_trace(first['frame'])
    frames = []
    for record in FRAME_RECORDS:
        xyz, faces, rgb = _crop_mesh(ROOT / record['mesh'])
        frame = record['frame']
        frames.append(go.Frame(
            name=f"{{frame['index']:02d}} {{frame['phase']}}",
            data=[go.Mesh3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                vertexcolor=_rgb_strings(rgb),
                flatshading=False,
                lighting=dict(ambient=0.92, diffuse=0.62, specular=0.035, roughness=0.92),
                hoverinfo='skip',
            ), _measurement_trace(frame)],
            traces=[0, 1],
            layout=go.Layout(
                title_text='Interactive HSR physics lesion growth - {{}} - {{:.0f}} mm'.format(
                    frame['phase'], frame['height'] * 1000.0
                )
            ),
        ))
    fig = go.Figure(data=[trace, measure], frames=frames)
    steps = [
        dict(
            args=[[frame.name], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))],
            label=frame.name,
            method='animate',
        )
        for frame in frames
    ]
    fig.update_layout(
        title='Interactive HSR physics lesion growth',
        scene=dict(
            xaxis=dict(visible=False, range=[-{half_width!r}, {half_width!r}]),
            yaxis=dict(visible=False, range=[-0.025, {depth_after!r}]),
            zaxis=dict(visible=False, range=[-{half_height!r}, {half_height!r}]),
            bgcolor='rgb(244,246,249)',
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=0.42, z=1.18),
            camera=dict(
                eye=dict(x=0.38, y=1.54, z=0.18),
                center=dict(x=0.0, y=0.02, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        width=900,
        height=720,
        margin=dict(l=0, r=0, t=54, b=0),
        paper_bgcolor='white',
        showlegend=False,
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            x=0.02, y=0.02, xanchor='left', yanchor='bottom',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=430, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(active=0, x=0.10, y=0.02, xanchor='left', yanchor='bottom', len=0.84, steps=steps)],
    )
    return fig
"""
    figure = make_notebook_figure(
        frame_records,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width,
        half_height,
        depth_after,
    )
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))

    cells = [
        nbf.v4.new_markdown_cell("# Physics augmentation lesion growth on HSR scan"),
        nbf.v4.new_markdown_cell(
            "This executed notebook contains an interactive Plotly mesh animation for the generated flat, sessile, globular, pedunculated, and final sessile lesion sequence."
        ),
        nbf.v4.new_code_cell(setup_code),
        nbf.v4.new_code_cell("fig = make_growth_figure()\nfig"),
    ]
    cells[2]["execution_count"] = 1
    cells[2]["outputs"] = []
    cells[3]["execution_count"] = 2
    cells[3]["outputs"] = [
        nbf.v4.new_output(
            output_type="display_data",
            data={
                "application/vnd.plotly.v1+json": payload,
                "text/plain": "<Plotly Figure: physics augmentation lesion growth>",
            },
            metadata={},
        )
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    nbf.write(notebook, notebook_path)


def clear_output_dirs(data_root: Path, visualization_root: Path) -> None:
    for child in (
        data_root / "meshes",
        data_root / "lesion_meshes",
        data_root / "metadata",
        visualization_root / "gifs",
        visualization_root / "plotly",
    ):
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)


def build_dataset(args: argparse.Namespace) -> None:
    scan_id = args.scan_id
    base_mesh_path = HSR_MESH_ROOT / f"{scan_id}_closed_textured_mesh.ply"
    if not base_mesh_path.exists():
        raise FileNotFoundError(f"Missing closed HSR mesh: {base_mesh_path}")

    data_root = DATASET_ROOT / "data"
    visualization_root = VISUALIZATION_ROOT
    clear_output_dirs(data_root, visualization_root)

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
    frames = growth_schedule(args.frames)
    max_height = max(frame.height for frame in frames)
    max_radius = max(frame.support_radius for frame in frames)

    frame_records: list[dict[str, object]] = []
    for frame in frames:
        lesion_xyz, lesion_faces, lesion_rgb = build_lesion_mesh(
            frame,
            base_xyz,
            base_faces,
            base_rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            gravity_local,
            radial_segments=args.radial_segments,
            angular_segments=args.angular_segments,
        )
        combined_xyz, combined_faces, combined_rgb = combine_base_and_lesion(
            base_xyz,
            base_faces,
            base_rgb,
            lesion_xyz,
            lesion_faces,
            lesion_rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            frame.support_radius,
            max_height,
        )
        stem = f"{scan_id}_physics_aug_growth_frame_{frame.index:03d}_{frame.phase_slug}"
        lesion_path = data_root / "lesion_meshes" / f"{stem}_lesion.ply"
        mesh_path = data_root / "meshes" / f"{stem}_hsr_lesion.ply"
        write_colored_ply(lesion_path, lesion_xyz, lesion_faces, lesion_rgb)
        write_colored_ply(mesh_path, combined_xyz, combined_faces, combined_rgb)
        frame_records.append(
            {
                "scan_id": scan_id,
                "stem": stem,
                "mesh": str(mesh_path.relative_to(DATASET_ROOT)),
                "lesion_mesh": str(lesion_path.relative_to(DATASET_ROOT)),
                "frame": asdict(frame),
            }
        )
        print(f"{frame.index:02d} {frame.phase:15s} height={frame.height:.4f} radius={frame.support_radius:.4f}")

    metadata_record: dict[str, object] = {
        "dataset": "physics_aug_growth",
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
            "max_support_radius": float(max_radius),
            "max_height": float(max_height),
        },
        "physics_model": {
            "type": "growth_pressure_mass_spring_gravity_contact",
            "description": (
                "A target-area soft-body mesh grows into pressure-like rest shapes, is connected by edge springs, "
                "is pinned at the skin attachment rim or stalk base, and is relaxed under the local gravity vector "
                "with damping, shape-memory, and skin-plane contact. The sequence gives flat, sessile, globular, "
                "pedunculated, gravity-plop, and final sessile morphologies."
            ),
            "radial_segments": args.radial_segments,
            "angular_segments": args.angular_segments,
            "solver": {
                "method": "explicit position-based mass-spring relaxation",
                "spring_constraint_passes": 3,
                "relaxation_iterations_per_frame": 120,
                "skin_contact_plane": "local outward-normal displacement y >= 0",
            },
            "literature_basis": [
                {
                    "topic": "cutaneous neurofibroma staged growth and extracellular-matrix remodeling",
                    "citation": "Rogiers et al., Unraveling the development of cutaneous neurofibromas in neurofibromatosis type 1, Acta Neuropathologica Communications, 2025",
                    "url": "https://link.springer.com/article/10.1186/s40478-025-02075-z",
                },
                {
                    "topic": "pedunculated cutaneous neurofibroma morphology",
                    "citation": "Hoang et al., Pedunculated Cutaneous Neurofibroma: a Case Report and Literature Review, SN Comprehensive Clinical Medicine, 2023",
                    "url": "https://doi.org/10.1007/s42399-023-01494-0",
                },
                {
                    "topic": "soft tissue mass-spring-damper dynamics",
                    "citation": "Murai et al., Dynamic Skin Deformation Simulation Using Musculoskeletal Model and Soft Tissue Dynamics, Pacific Graphics, 2016",
                    "url": "https://la.disneyresearch.com/publication/dynamic-skin-deformation-simulation/",
                },
                {
                    "topic": "tumor growth pressure as a mechanical driver",
                    "citation": "Abdolkarimzadeh et al., A position- and time-dependent pressure profile to model viscoelastic mechanical behavior of brain tissue due to tumor growth, Computer Methods in Biomechanics and Biomedical Engineering, 2022",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/35638726/",
                },
            ],
        },
        "frames": [record["frame"] for record in frame_records],
    }
    metadata_path = data_root / "metadata" / f"{scan_id}_physics_aug_growth_metadata.json"
    metadata_path.write_text(json.dumps(metadata_record, indent=2))

    gif_path = visualization_root / "gifs" / f"{scan_id}_physics_aug_lesion_growth.gif"
    notebook_path = visualization_root / "plotly" / "physics_aug_growth_plotly_viewer.ipynb"
    manifest = {
        "dataset": "physics_aug_growth",
        "scan_id": scan_id,
        "metadata": str(metadata_path.relative_to(DATASET_ROOT)),
        "visualization_root": root_relative(visualization_root),
        "gif": root_relative(gif_path),
        "notebook": root_relative(notebook_path),
        "frames": frame_records,
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (visualization_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    render_growth_gif(
        frame_records,
        gif_path,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width=args.view_half_width,
        half_height=args.view_half_height,
        depth_after=max(0.095, max_height + 0.035),
    )
    write_notebook(
        notebook_path,
        frame_records,
        metadata_record,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width=args.view_half_width,
        half_height=args.view_half_height,
        depth_after=max(0.095, max_height + 0.035),
    )
    print(gif_path)
    print(notebook_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", default="HSR0018-Body-070")
    parser.add_argument("--target-x", type=float, default=-0.09, help="Approximate HSR x-coordinate for target area.")
    parser.add_argument("--target-y", type=float, default=None, help="Optional HSR y-coordinate; omitted picks front surface.")
    parser.add_argument("--target-z", type=float, default=1.09, help="Approximate HSR z-coordinate for target area.")
    parser.add_argument("--target-window", type=float, default=0.040)
    parser.add_argument("--frames", type=int, default=26)
    parser.add_argument("--radial-segments", type=int, default=34)
    parser.add_argument("--angular-segments", type=int, default=112)
    parser.add_argument("--view-half-width", type=float, default=0.155)
    parser.add_argument("--view-half-height", type=float, default=0.180)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()

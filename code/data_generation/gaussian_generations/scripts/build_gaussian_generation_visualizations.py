#!/usr/bin/env python3
"""Build closed-body Plotly visualizations for gaussian generation folders."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from PIL import Image
from plyfile import PlyData, PlyElement

Image.MAX_IMAGE_PIXELS = None

DATA_ROOT = Path(__file__).resolve().parents[4] / "data"
SYNTHETIC_BODY_PARTS_ROOT = DATA_ROOT / "synthetic" / "single_lesion" / "body_parts"
SYNTHETIC_VISUALIZATION_ROOT = DATA_ROOT / "synthetic" / "single_lesion" / "visualization"
SCAN_IDS = ("HSR0018-Body-070", "HSR0152-Body-090")
_CENTER_CACHE: dict[str, np.ndarray] = {}
_SAMPLED_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_BASE_PLY_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


@dataclass
class GenerationFolder:
    name: str
    root: Path
    visualization_root: Path
    metadata_dir: Path
    obj_dir: Path | None
    texture_dir: Path | None
    suffix: str
    mode: str


def parse_face_token(token: str) -> tuple[int, int | None]:
    parts = token.split("/")
    vertex_idx = int(parts[0]) - 1
    texture_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
    return vertex_idx, texture_idx


def rgb_strings(rgb: np.ndarray) -> list[str]:
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]


def read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    face = ply["face"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    faces = np.vstack(face["vertex_indices"]).astype(np.int32)
    return xyz, faces, rgb


def sampled_generation_center(scan_id: str, target_faces: int = 45_000) -> np.ndarray:
    """Return the centering offset used by the original gaussian-cap generator."""
    if scan_id in _CENTER_CACHE:
        return _CENTER_CACHE[scan_id]

    obj_path = DATA_ROOT / "hsr" / "scans" / scan_id / "scan" / f"{scan_id}.obj"
    vertices = []
    face_count = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if " vertices, " in line and " faces" in line:
                face_count = int(line.split(" vertices, ")[1].split(" faces")[0])
            elif line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))

    if face_count is None:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            face_count = sum(1 for line in handle if line.startswith("f "))

    vertices_arr = np.asarray(vertices, dtype=np.float32)
    keep_face_numbers = set(np.linspace(0, face_count - 1, min(target_faces, face_count), dtype=np.int64))
    remap = {}
    sampled_vertices = []
    seen_faces = 0
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            if seen_faces in keep_face_numbers:
                for token in line.split()[1:4]:
                    vertex_idx, _ = parse_face_token(token)
                    if vertex_idx not in remap:
                        remap[vertex_idx] = len(sampled_vertices)
                        sampled_vertices.append(vertices_arr[vertex_idx])
            seen_faces += 1

    center = np.asarray(sampled_vertices, dtype=np.float32).mean(axis=0)
    _CENTER_CACHE[scan_id] = center
    return center


def sampled_generation_mesh(scan_id: str, target_faces: int = 45_000) -> tuple[np.ndarray, np.ndarray]:
    if scan_id in _SAMPLED_MESH_CACHE:
        return _SAMPLED_MESH_CACHE[scan_id]

    obj_path = DATA_ROOT / "hsr" / "scans" / scan_id / "scan" / f"{scan_id}.obj"
    vertices = []
    face_count = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if " vertices, " in line and " faces" in line:
                face_count = int(line.split(" vertices, ")[1].split(" faces")[0])
            elif line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
    if face_count is None:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            face_count = sum(1 for line in handle if line.startswith("f "))

    keep_face_numbers = set(np.linspace(0, face_count - 1, min(target_faces, face_count), dtype=np.int64))
    vertices_arr = np.asarray(vertices, dtype=np.float32)
    remap = {}
    sampled_vertices = []
    sampled_faces = []
    seen_faces = 0
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            if seen_faces in keep_face_numbers:
                tri = []
                for token in line.split()[1:4]:
                    vertex_idx, _ = parse_face_token(token)
                    if vertex_idx not in remap:
                        remap[vertex_idx] = len(sampled_vertices)
                        sampled_vertices.append(vertices_arr[vertex_idx])
                    tri.append(remap[vertex_idx])
                sampled_faces.append(tri)
            seen_faces += 1

    sampled_vertices = np.asarray(sampled_vertices, dtype=np.float32)
    result = (sampled_vertices - sampled_vertices.mean(axis=0), np.asarray(sampled_faces, dtype=np.int32))
    _SAMPLED_MESH_CACHE[scan_id] = result
    return result


def read_base_mesh(scan_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scan_id not in _BASE_PLY_CACHE:
        base_mesh_root = DATA_ROOT / "hsr" / "visualizations" / "meshes"
        _BASE_PLY_CACHE[scan_id] = read_colored_ply(base_mesh_root / f"{scan_id}_closed_textured_mesh.ply")
    return _BASE_PLY_CACHE[scan_id]


def sample_skin_patch_points(
    local_points: np.ndarray,
    anchor: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    normal: np.ndarray | None = None,
) -> np.ndarray:
    tangent_u = tangent_u / np.linalg.norm(tangent_u)
    tangent_v = tangent_v / np.linalg.norm(tangent_v)
    skin_triangles = skin_vertices[skin_faces]
    skin_offsets = skin_triangles - anchor
    skin_triangle_plane = np.stack([skin_offsets @ tangent_u, skin_offsets @ tangent_v], axis=2)
    skin_plane_all = skin_triangle_plane.mean(axis=1)
    if normal is not None:
        normal = normal / np.linalg.norm(normal)
        skin_normal_all = (skin_triangles.mean(axis=1) - anchor) @ normal
        max_radius = float(np.linalg.norm(local_points, axis=1).max()) if len(local_points) else 0.0
        plane_radius = np.linalg.norm(skin_plane_all, axis=1)
        candidate_mask = (plane_radius <= max(0.075, 2.25 * max_radius)) & (
            np.abs(skin_normal_all) <= max(0.045, 1.5 * max_radius)
        )
        if candidate_mask.sum() < 16:
            candidate_mask = plane_radius <= max(0.075, 2.25 * max_radius)
        if candidate_mask.sum() >= 6:
            fit_plane = skin_plane_all[candidate_mask]
            fit_normal = skin_normal_all[candidate_mask]
            fit_radius = np.linalg.norm(fit_plane, axis=1)
            weight_scale = max(0.035, 1.35 * max_radius)
            weights = np.exp(-0.5 * (fit_radius / weight_scale) ** 2)
            design = np.column_stack(
                [
                    np.ones(len(fit_plane), dtype=np.float32),
                    fit_plane[:, 0],
                    fit_plane[:, 1],
                    fit_plane[:, 0] * fit_plane[:, 0],
                    fit_plane[:, 0] * fit_plane[:, 1],
                    fit_plane[:, 1] * fit_plane[:, 1],
                ]
            )
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_normal = fit_normal * np.sqrt(weights)
            try:
                coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_normal, rcond=None)
                u = local_points[:, 0]
                v = local_points[:, 1]
                fitted_normal = (
                    coeffs[0]
                    + coeffs[1] * u
                    + coeffs[2] * v
                    + coeffs[3] * u * u
                    + coeffs[4] * u * v
                    + coeffs[5] * v * v
                )
                return (
                    anchor
                    + local_points[:, 0, None] * tangent_u
                    + local_points[:, 1, None] * tangent_v
                    + fitted_normal[:, None] * normal
                ).astype(np.float32)
            except np.linalg.LinAlgError:
                pass

        if candidate_mask.sum() >= 3:
            skin_triangles = skin_triangles[candidate_mask]
            skin_triangle_plane = skin_triangle_plane[candidate_mask]
            skin_plane = skin_plane_all[candidate_mask]
        else:
            skin_plane = skin_plane_all
    else:
        skin_plane = skin_plane_all

    sampled = np.empty((len(local_points), 3), dtype=np.float32)
    for start in range(0, len(local_points), 256):
        stop = start + 256
        delta = local_points[start:stop, None, :] - skin_plane[None, :, :]
        nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        targets = local_points[start:stop]
        tri_plane = skin_triangle_plane[nearest]
        tri_xyz = skin_triangles[nearest]

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
    return sampled


def build_gaussian_shape_template(
    height: float,
    support_radius: float,
    sigma: float,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_points = [np.array([0.0, 0.0], dtype=np.float32)]
    profile_heights = [float(height)]
    faces = []
    edge_value = np.exp(-(support_radius * support_radius) / (2 * sigma * sigma))

    for ring in range(1, radial_segments + 1):
        rho = support_radius * ring / radial_segments
        raw = np.exp(-(rho * rho) / (2 * sigma * sigma))
        z = height * max(0.0, (raw - edge_value) / max(1.0 - edge_value, 1e-8))
        for step in range(angular_segments):
            theta = 2 * np.pi * step / angular_segments
            local_points.append(np.array([rho * np.cos(theta), rho * np.sin(theta)], dtype=np.float32))
            profile_heights.append(z)

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
        np.asarray(faces, dtype=np.int32),
    )


def snap_outer_ring_to_skin(
    points: np.ndarray,
    radial_segments: int,
    angular_segments: int,
    anchor: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
) -> np.ndarray:
    snapped = points.astype(np.float32).copy()
    base_start = 1 + (radial_segments - 1) * angular_segments
    rim = snapped[base_start : base_start + angular_segments]
    offsets = rim - anchor
    rim_local = np.column_stack([offsets @ tangent_u, offsets @ tangent_v]).astype(np.float32)
    snapped[base_start : base_start + angular_segments] = sample_skin_patch_points(
        rim_local, anchor, tangent_u, tangent_v, skin_vertices, skin_faces
    )
    return snapped


def remove_covered_skin_faces(
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    metadata_path: Path,
    center_offset: np.ndarray,
    margin: float = 0.96,
) -> np.ndarray:
    meta = json.loads(metadata_path.read_text())
    anchor = np.asarray(meta["anchor"], dtype=np.float32) + center_offset
    normal = np.asarray(meta["normal"], dtype=np.float32)
    tangent_u = np.asarray(meta["tangent_u"], dtype=np.float32)
    tangent_v = np.asarray(meta["tangent_v"], dtype=np.float32)
    normal /= np.linalg.norm(normal)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    footprint = float(meta["support_radius"]) * margin
    height = float(meta["height"])

    centroids = base_xyz[base_faces].mean(axis=1)
    offsets = centroids - anchor
    local_u = offsets @ tangent_u
    local_v = offsets @ tangent_v
    local_n = offsets @ normal
    radial = np.sqrt(local_u * local_u + local_v * local_v)
    covered = (radial <= footprint) & (np.abs(local_n) <= max(0.06, 4.0 * height))
    if covered.sum() == 0:
        covered = radial <= footprint
    return base_faces[~covered]


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


def save_preview_gif(preview_path: Path, gif_path: Path) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(preview_path) as image:
        image.save(gif_path, format="GIF")


def remove_degenerate_faces(xyz: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(faces) == 0:
        return faces
    triangles = xyz[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    return faces[areas > eps]


def interpolate_bump_vertex_colors(
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    base_rgb: np.ndarray,
    bump_xyz: np.ndarray,
    metadata_path: Path,
    center_offset: np.ndarray,
    fallback_color: tuple[int, int, int] = (178, 124, 104),
    neighbors: int = 16,
    search_scale: float = 2.8,
) -> np.ndarray:
    meta = json.loads(metadata_path.read_text())
    anchor = np.asarray(meta["anchor"], dtype=np.float32) + center_offset
    normal = np.asarray(meta["normal"], dtype=np.float32)
    tangent_u = np.asarray(meta["tangent_u"], dtype=np.float32)
    tangent_v = np.asarray(meta["tangent_v"], dtype=np.float32)
    normal /= np.linalg.norm(normal)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    support_radius = float(meta["support_radius"])
    sigma = float(meta["sigma"])
    height = float(meta["height"])

    face_centroids = base_xyz[base_faces].mean(axis=1)
    face_rgb = base_rgb[base_faces].mean(axis=1).astype(np.float32)
    offsets = face_centroids - anchor
    local = np.column_stack([offsets @ tangent_u, offsets @ tangent_v, offsets @ normal])
    radial = np.linalg.norm(local[:, :2], axis=1)
    candidate_mask = (radial <= search_scale * support_radius) & (np.abs(local[:, 2]) <= max(0.08, 5.0 * height))
    if candidate_mask.sum() < neighbors:
        candidate_mask = radial <= max(search_scale * support_radius, np.quantile(radial, 0.025))
    if candidate_mask.sum() < neighbors:
        return np.tile(np.asarray(fallback_color, dtype=np.uint8), (len(bump_xyz), 1))

    candidate_plane = local[candidate_mask, :2]
    candidate_rgb = face_rgb[candidate_mask]
    bump_offsets = bump_xyz - anchor
    bump_plane = np.column_stack([bump_offsets @ tangent_u, bump_offsets @ tangent_v])

    output = np.empty((len(bump_xyz), 3), dtype=np.float32)
    k = min(neighbors, len(candidate_plane))
    for start in range(0, len(bump_xyz), 512):
        stop = start + 512
        delta = bump_plane[start:stop, None, :] - candidate_plane[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        weights = 1.0 / np.maximum(nearest_distances, 1e-6) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        output[start:stop] = (candidate_rgb[nearest] * weights[..., None]).sum(axis=1)

    bump_radial = np.linalg.norm(bump_plane, axis=1)
    edge_value = np.exp(-(support_radius * support_radius) / (2 * sigma * sigma))
    raw = np.exp(-(bump_radial * bump_radial) / (2 * sigma * sigma))
    profile = np.clip((raw - edge_value) / max(1.0 - edge_value, 1e-8), 0.0, 1.0)
    output += profile[:, None] * np.array([8.0, 2.5, 1.0], dtype=np.float32)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def build_bump_from_metadata(
    metadata_path: Path,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    color: tuple[int, int, int] = (178, 96, 104),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads(metadata_path.read_text())
    height = float(meta["height"])
    support_radius = float(meta["support_radius"])
    sigma = float(meta["sigma"])
    anchor = np.asarray(meta["anchor"], dtype=np.float32)
    normal = np.asarray(meta["normal"], dtype=np.float32)
    tangent_u = np.asarray(meta["tangent_u"], dtype=np.float32)
    tangent_v = np.asarray(meta["tangent_v"], dtype=np.float32)
    normal /= np.linalg.norm(normal)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)

    radial_segments = 30
    angular_segments = 112
    local_points, profile_heights, faces = build_gaussian_shape_template(
        height=height,
        support_radius=support_radius,
        sigma=sigma,
        radial_segments=radial_segments,
        angular_segments=angular_segments,
    )
    skin_base_points = sample_skin_patch_points(local_points, anchor, tangent_u, tangent_v, skin_vertices, skin_faces, normal=normal)
    points = skin_base_points + profile_heights[:, None] * normal

    # Close the cap with a base disk at the body contact plane. The disk sits on the
    # closed body and prevents the bump mesh itself from having an open rim.
    center_idx = len(points)
    points = np.vstack([points, skin_base_points[0]])
    base_start = 1 + (radial_segments - 1) * angular_segments
    faces = faces.tolist()
    for step in range(angular_segments):
        a = base_start + step
        b = base_start + ((step + 1) % angular_segments)
        faces.append([center_idx, b, a])

    xyz = np.asarray(points, dtype=np.float32)
    face_arr = remove_degenerate_faces(xyz, np.asarray(faces, dtype=np.int32))
    rgb = np.tile(np.asarray(color, dtype=np.uint8), (len(xyz), 1))
    return xyz, face_arr, rgb


def read_obj_cap(obj_path: Path, texture_path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    colors: list[list[int] | None] = []
    texcoords: list[tuple[float, float]] = []
    raw_faces: list[list[tuple[int, int | None]]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([int(round(float(parts[4]) * 255)), int(round(float(parts[5]) * 255)), int(round(float(parts[6]) * 255))])
                else:
                    colors.append(None)
            elif line.startswith("vt "):
                parts = line.split()
                texcoords.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                raw_faces.append([parse_face_token(token) for token in line.split()[1:4]])

    vertices_arr = np.asarray(vertices, dtype=np.float32)
    texcoords_arr = np.asarray(texcoords, dtype=np.float32)
    texture = None
    if texture_path is not None and texture_path.exists():
        image = Image.open(texture_path).convert("RGB")
        texture = np.asarray(image, dtype=np.uint8)

    remap: dict[tuple[int, int | None], int] = {}
    out_vertices: list[np.ndarray] = []
    out_rgb: list[list[int]] = []
    out_faces: list[list[int]] = []
    for face in raw_faces:
        tri = []
        for vertex_idx, texture_idx in face:
            key = (vertex_idx, texture_idx)
            if key not in remap:
                remap[key] = len(out_vertices)
                out_vertices.append(vertices_arr[vertex_idx])
                if texture is not None and texture_idx is not None and 0 <= texture_idx < len(texcoords_arr):
                    u, v = texcoords_arr[texture_idx]
                    h, w = texture.shape[:2]
                    px = int(np.clip(round(u * (w - 1)), 0, w - 1))
                    py = int(np.clip(round((1.0 - v) * (h - 1)), 0, h - 1))
                    out_rgb.append(texture[py, px].astype(int).tolist())
                elif colors[vertex_idx] is not None:
                    out_rgb.append(colors[vertex_idx])
                else:
                    out_rgb.append([178, 96, 104])
            tri.append(remap[key])
        out_faces.append(tri)

    xyz = np.asarray(out_vertices, dtype=np.float32)
    faces = np.asarray(out_faces, dtype=np.int32)
    rgb = np.asarray(out_rgb, dtype=np.uint8)
    return xyz, faces, rgb


def close_open_cap_mesh(xyz: np.ndarray, faces: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) < 3:
        return xyz, faces, rgb

    boundary = np.unique(boundary_edges.ravel())
    center = xyz[boundary].mean(axis=0, keepdims=True)
    centered = xyz[boundary] - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u, axis_v = vh[0], vh[1]
    coords = np.column_stack([centered @ axis_u, centered @ axis_v])
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    ring = boundary[np.argsort(angles)]

    center_idx = len(xyz)
    xyz = np.vstack([xyz, center])
    rgb = np.vstack([rgb, np.median(rgb[boundary], axis=0, keepdims=True).astype(np.uint8)])
    disk_faces = [[center_idx, int(ring[(idx + 1) % len(ring)]), int(ring[idx])] for idx in range(len(ring))]
    faces = np.vstack([faces, np.asarray(disk_faces, dtype=np.int32)])
    faces = remove_degenerate_faces(xyz, faces)
    return xyz, faces, rgb


def normalize_for_plot(xyz: np.ndarray) -> np.ndarray:
    out = xyz.astype(np.float32).copy()
    out -= out.mean(axis=0)
    scale = np.max(np.ptp(out, axis=0))
    if scale > 0:
        out /= scale
    return out


def make_plotly_figure(ply_path: Path, title: str) -> go.Figure:
    xyz, faces, rgb = read_colored_ply(ply_path)
    xyz = normalize_for_plot(xyz)
    colors = rgb_strings(rgb)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=colors,
                flatshading=False,
                lighting=dict(ambient=0.95, diffuse=0.55, specular=0.04, roughness=0.9),
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(242,244,247)",
            aspectmode="data",
            camera=dict(eye=dict(x=0.0, y=2.15, z=0.45), center=dict(x=0, y=0, z=0.04)),
        ),
        width=1000,
        height=780,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def folder_configs() -> list[GenerationFolder]:
    return [
        GenerationFolder(
            name="gaussian_generations",
            root=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations",
            visualization_root=SYNTHETIC_VISUALIZATION_ROOT / "gaussian_generations",
            metadata_dir=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations" / "metadata",
            obj_dir=None,
            texture_dir=None,
            suffix="",
            mode="metadata",
        ),
        GenerationFolder(
            name="gaussian_generations_textured_diffusion",
            root=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_diffusion",
            visualization_root=SYNTHETIC_VISUALIZATION_ROOT / "gaussian_generations_textured_diffusion",
            metadata_dir=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_diffusion" / "data" / "metadata",
            obj_dir=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_diffusion" / "data" / "objs",
            texture_dir=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_diffusion" / "data" / "textures",
            suffix="_textured_diffusion",
            mode="obj_texture",
        ),
        GenerationFolder(
            name="gaussian_generations_textured_interpolation",
            root=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_interpolation",
            visualization_root=SYNTHETIC_VISUALIZATION_ROOT / "gaussian_generations_textured_interpolation",
            metadata_dir=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_interpolation",
            obj_dir=SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations_textured_interpolation" / "objs",
            texture_dir=None,
            suffix="_textured_interpolation",
            mode="obj_vertex_color",
        ),
    ]


def metadata_to_stem(metadata_path: Path, suffix: str) -> str:
    stem = metadata_path.stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def generation_items(folder: GenerationFolder) -> list[dict[str, object]]:
    if folder.name == "gaussian_generations_textured_interpolation":
        manifest_path = folder.root / "manifest.csv"
        rows = []
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                obj_path = DATA_ROOT / row["obj"]
                rows.append(
                    {
                        "stem_base": obj_path.stem.removesuffix(folder.suffix),
                        "out_stem": obj_path.stem,
                        "scan_id": row["scan_id"],
                        "metadata_path": DATA_ROOT / row["source_metadata"],
                    }
                )
        return rows

    items = []
    for metadata_path in sorted(folder.metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        stem_base = metadata_to_stem(metadata_path, folder.suffix)
        out_stem = metadata_path.stem if folder.suffix else stem_base
        items.append(
            {
                "stem_base": stem_base,
                "out_stem": out_stem,
                "scan_id": metadata["scan_id"],
                "metadata_path": metadata_path,
            }
        )
    return items


def clear_old_visualizations(folder: GenerationFolder) -> tuple[Path, Path, Path, Path]:
    vis_root = folder.visualization_root
    mesh_root = vis_root / "meshes"
    preview_root = vis_root / "previews"
    notebook_root = vis_root / "plotly"
    gif_root = vis_root / "gifs"
    for path in (mesh_root, preview_root, notebook_root):
        if path.exists():
            shutil.rmtree(path)
    if gif_root.exists():
        for stale_gif in gif_root.glob("*_closed_plotly_preview.gif"):
            stale_gif.unlink()
    manifest_path = vis_root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    mesh_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)
    notebook_root.mkdir(parents=True, exist_ok=True)
    gif_root.mkdir(parents=True, exist_ok=True)
    return mesh_root, preview_root, notebook_root, gif_root


def build_folder(folder: GenerationFolder) -> None:
    mesh_root, preview_root, notebook_root, gif_root = clear_old_visualizations(folder)
    records = []
    for item in generation_items(folder):
        stem_base = str(item["stem_base"])
        out_stem = str(item["out_stem"])
        scan_id = str(item["scan_id"])
        metadata_path = Path(item["metadata_path"])
        base_xyz, base_faces, base_rgb = read_base_mesh(scan_id)
        skin_vertices, skin_faces = sampled_generation_mesh(scan_id)

        if folder.mode == "metadata":
            bump_xyz, bump_faces, bump_rgb = build_bump_from_metadata(metadata_path, skin_vertices, skin_faces)
        else:
            obj_path = folder.obj_dir / f"{out_stem}.obj"
            texture_path = folder.texture_dir / f"{out_stem}.png" if folder.texture_dir is not None else None
            bump_xyz, bump_faces, bump_rgb = read_obj_cap(obj_path, texture_path)
            bump_xyz, bump_faces, bump_rgb = close_open_cap_mesh(bump_xyz, bump_faces, bump_rgb)

        center_offset = sampled_generation_center(scan_id)
        bump_xyz = bump_xyz + center_offset
        if folder.mode == "metadata":
            bump_rgb = interpolate_bump_vertex_colors(base_xyz, base_faces, base_rgb, bump_xyz, metadata_path, center_offset)
        visible_base_faces = remove_covered_skin_faces(base_xyz, base_faces, metadata_path, center_offset)
        bump_faces_offset = bump_faces + len(base_xyz)
        combined_xyz = np.vstack([base_xyz, bump_xyz])
        combined_faces = np.vstack([visible_base_faces, bump_faces_offset])
        combined_rgb = np.vstack([base_rgb, bump_rgb])
        out_ply = mesh_root / f"{out_stem}_closed_textured_visualization.ply"
        write_colored_ply(out_ply, combined_xyz, combined_faces, combined_rgb)

        preview_path = preview_root / f"{out_stem}_closed_plotly_preview.png"
        make_plotly_figure(out_ply, out_stem).write_image(preview_path, scale=1)
        gif_path = gif_root / f"{out_stem}_closed_plotly_preview.gif"
        save_preview_gif(preview_path, gif_path)
        records.append(
            {
                "stem": out_stem,
                "scan_id": scan_id,
                "mesh": str(out_ply.relative_to(folder.visualization_root)),
                "preview": str(preview_path.relative_to(folder.visualization_root)),
                "gif": str(gif_path.relative_to(folder.visualization_root)),
            }
        )
        print(folder.name, out_stem, out_ply.name)

    manifest_path = folder.visualization_root / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2))
    write_notebook(folder, notebook_root, records)


def write_notebook(folder: GenerationFolder, notebook_root: Path, records: list[dict[str, str]]) -> None:
    records_json = json.dumps(records, indent=2)
    selected_records = list(records)
    selected_json = json.dumps(selected_records, indent=2)
    setup_code = f"""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData

DATASET_NAME = '{folder.name}'
ROOT_CANDIDATES = []
for parent in (Path.cwd(), *Path.cwd().parents):
    ROOT_CANDIDATES.append(parent / 'data' / 'synthetic' / 'single_lesion' / 'visualization' / DATASET_NAME)
    ROOT_CANDIDATES.append(parent / 'data' / 'synthetic' / DATASET_NAME / 'visualizations')
ROOT_CANDIDATES.append(Path.cwd())
ROOT = next((path for path in ROOT_CANDIDATES if (path / 'manifest.json').exists()), ROOT_CANDIDATES[0])
RECORDS = {records_json}
SELECTED_RECORDS = {selected_json}

def _load_colored_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    f = ply['face'].data
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    xyz -= xyz.mean(axis=0)
    scale = np.max(np.ptp(xyz, axis=0))
    if scale > 0:
        xyz /= scale
    faces = np.vstack(f['vertex_indices']).astype(np.int32)
    colors = [f"rgb({{int(r)}},{{int(g)}},{{int(b)}})" for r, g, b in zip(v['red'], v['green'], v['blue'])]
    return xyz, faces, colors

def make_figure(record):
    xyz, faces, colors = _load_colored_ply(ROOT / record['mesh'])
    fig = go.Figure(data=[go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=colors,
        flatshading=False,
        lighting=dict(ambient=0.95, diffuse=0.55, specular=0.04, roughness=0.9),
        hoverinfo='skip',
    )])
    fig.update_layout(
        title=record['stem'],
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgb(242,244,247)', aspectmode='data',
            camera=dict(eye=dict(x=0.0, y=2.15, z=0.45), center=dict(x=0, y=0, z=0.04)),
        ),
        width=1000, height=780,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor='white',
        showlegend=False,
    )
    return fig
"""
    cells = [
        nbf.v4.new_markdown_cell(f"# {folder.name} closed textured Plotly viewer"),
        nbf.v4.new_markdown_cell(
            f"This notebook shows all {len(selected_records)} closed-body Plotly visualizations in `{folder.name}`."
        ),
        nbf.v4.new_code_cell(setup_code),
    ]
    for idx, record in enumerate(selected_records):
        cells.append(nbf.v4.new_markdown_cell(f"## {idx + 1}. {record['stem']}"))
        fig = make_plotly_figure(folder.visualization_root / record["mesh"], record["stem"])
        payload = json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))
        cell = nbf.v4.new_code_cell(f"make_figure(SELECTED_RECORDS[{idx}])")
        cell["execution_count"] = idx + 1
        cell["outputs"] = [
            nbf.v4.new_output(
                output_type="display_data",
                data={
                    "application/vnd.plotly.v1+json": payload,
                    "text/plain": f"<Plotly Figure: {record['stem']}>",
                },
                metadata={},
            )
        ]
        cells.append(cell)
    nb = nbf.v4.new_notebook(cells=cells)
    notebook_path = notebook_root / f"{folder.name}_closed_plotly_viewer.ipynb"
    nbf.write(nb, notebook_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", choices=[cfg.name for cfg in folder_configs()], action="append")
    args = parser.parse_args()
    selected = set(args.folder or [cfg.name for cfg in folder_configs()])
    for folder in folder_configs():
        if folder.name in selected:
            build_folder(folder)


if __name__ == "__main__":
    main()

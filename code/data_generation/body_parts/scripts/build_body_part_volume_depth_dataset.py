#!/usr/bin/env python3
"""Generate body-part-specific synthetic lesion volumes and RGB/depth pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import pyrender
import trimesh

ROOT = Path(__file__).resolve().parents[4]
HSR_MESH_ROOT = ROOT / "data" / "hsr" / "visualizations" / "meshes"
HSR_SEGMENTATION_MANUAL_ROOT = ROOT / "data" / "hsr" / "body_part_segmentation" / "manual" / "data"
HSR_SEGMENTATION_ROOT = (
    HSR_SEGMENTATION_MANUAL_ROOT
    if HSR_SEGMENTATION_MANUAL_ROOT.exists()
    else ROOT / "data" / "hsr" / "body_part_segmentation" / "data"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "synthetic" / "multiple_lesion" / "body_parts" / "physics_aug_growth" / "body_parts_dataset"
DEFAULT_VISUALIZATION_ROOT = (
    ROOT / "data" / "synthetic" / "multiple_lesion" / "visualization" / "physics_aug_growth" / "body_parts_dataset"
)

BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]
SCAN_IDS = ["HSR0018-Body-070", "HSR0152-Body-090"]


@dataclass
class LesionVolume:
    sample_id: str
    body_part: str
    scan_id: str
    patient_volume_index: int
    seed: int
    face_index: int
    radius_m: float
    height_m: float
    support_radius_m: float
    projection_max_distance_m: float
    contact_label_fraction: float
    spherical_cap_volume_m3: float
    spherical_cap_volume_ml: float
    anchor_xyz: list[float]
    normal_xyz: list[float]
    tangent_u_xyz: list[float]
    tangent_v_xyz: list[float]
    base_rgb: list[int]
    lesion_rgb: list[int]
    camera: dict[str, Any]


class ScanSurface:
    def __init__(self, scan_id: str) -> None:
        self.scan_id = scan_id
        mesh_path = HSR_MESH_ROOT / f"{scan_id}_closed_textured_mesh.ply"
        segmentation_path = HSR_SEGMENTATION_ROOT / f"{scan_id}_body_part_segmentation.npz"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing HSR mesh: {mesh_path}")
        if not segmentation_path.exists():
            raise FileNotFoundError(
                f"Missing HSR body-part segmentation: {segmentation_path}. "
                "Run build_hsr_body_part_segmentation.py first."
            )

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.faces = np.asarray(mesh.triangles, dtype=np.int32)
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if mesh.has_vertex_colors():
            self.vertex_rgb = np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255.0), 0, 255).astype(np.uint8)
        else:
            self.vertex_rgb = np.full((len(self.vertices), 3), 185, dtype=np.uint8)

        segmentation = np.load(segmentation_path)
        self.label_names = [str(value) for value in segmentation["label_names"].tolist()]
        self.vertex_labels = segmentation["vertex_labels"].astype(np.uint8)
        self.face_labels = segmentation["face_labels"].astype(np.uint8)
        if len(self.face_labels) != len(self.faces):
            raise ValueError(f"Segmentation face count does not match mesh for {scan_id}")
        if len(self.vertex_labels) != len(self.vertices):
            raise ValueError(f"Segmentation vertex count does not match mesh for {scan_id}")
        self.face_vertex_labels = self.vertex_labels[self.faces]

        self.center = self.vertices.mean(axis=0).astype(np.float32)
        self.face_triangles = self.vertices[self.faces]
        self.face_centroids = self.face_triangles.mean(axis=1).astype(np.float32)
        edge_a = self.face_triangles[:, 1] - self.face_triangles[:, 0]
        edge_b = self.face_triangles[:, 2] - self.face_triangles[:, 0]
        face_normals = np.cross(edge_a, edge_b)
        areas = np.linalg.norm(face_normals, axis=1) * 0.5
        valid = areas > 1e-10
        face_normals[valid] /= np.linalg.norm(face_normals[valid], axis=1, keepdims=True)
        normal_from_vertices = self.vertex_normals[self.faces].mean(axis=1)
        normal_lengths = np.linalg.norm(normal_from_vertices, axis=1)
        use_vertex_normals = normal_lengths > 1e-8
        normal_from_vertices[use_vertex_normals] /= normal_lengths[use_vertex_normals, None]
        face_normals[use_vertex_normals] = normal_from_vertices[use_vertex_normals]
        inward = np.sum(face_normals * (self.face_centroids - self.center), axis=1) < 0.0
        face_normals[inward] *= -1.0
        self.face_normals = face_normals.astype(np.float32)
        self.face_areas = areas.astype(np.float64)
        self.face_rgb = self.vertex_rgb[self.faces].mean(axis=1).astype(np.uint8)

        base_colors = np.column_stack([self.vertex_rgb, np.full(len(self.vertex_rgb), 255, dtype=np.uint8)])
        self.base_trimesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=base_colors,
            process=False,
        )
        self.proximity = trimesh.proximity.ProximityQuery(self.base_trimesh)
        self.base_render_mesh = pyrender.Mesh.from_trimesh(self.base_trimesh, smooth=True)

    def label_id(self, body_part: str) -> int:
        return self.label_names.index(body_part)

    def candidate_faces(self, body_part: str) -> np.ndarray:
        label_id = self.label_id(body_part)
        mask = (
            (self.face_labels == label_id)
            & np.all(self.face_vertex_labels == label_id, axis=1)
            & (self.face_areas > 1e-10)
        )
        if body_part == "arms" and "hands" in self.label_names:
            mask &= ~np.any(self.face_vertex_labels == self.label_id("hands"), axis=1)
        elif body_part == "legs" and "feet" in self.label_names:
            mask &= ~np.any(self.face_vertex_labels == self.label_id("feet"), axis=1)
        candidates = np.flatnonzero(mask)
        if len(candidates) < 32:
            raise ValueError(f"Too few candidate faces for {self.scan_id} {body_part}: {len(candidates)}")
        return candidates

    def project_points_to_surface(
        self,
        points: np.ndarray,
        normal_hint: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        closest, distances, triangle_ids = self.proximity.on_surface(np.asarray(points, dtype=np.float64))
        normals = self.face_normals[triangle_ids].astype(np.float32)
        if normal_hint is not None:
            normal_hint = normalized(normal_hint)
            flip = normals @ normal_hint < 0.0
            normals[flip] *= -1.0
        rgb = self.face_rgb[triangle_ids].astype(np.uint8)
        return (
            closest.astype(np.float32),
            normals,
            rgb,
            triangle_ids.astype(np.int32),
            distances.astype(np.float32),
        )


def resolve_root(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def resolve_output_root(path_value: str | None) -> Path:
    if path_value is None:
        return DEFAULT_OUTPUT_ROOT
    return resolve_root(path_value)


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return (vector / length).astype(np.float32)


def tangent_basis(normal: np.ndarray, preferred: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    normal = normalized(normal)
    if preferred is None:
        preferred = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tangent_v = preferred - float(np.dot(preferred, normal)) * normal
    if float(np.linalg.norm(tangent_v)) <= 1e-7:
        tangent_v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tangent_v = tangent_v - float(np.dot(tangent_v, normal)) * normal
    tangent_v = normalized(tangent_v)
    tangent_u = normalized(np.cross(tangent_v, normal))
    tangent_v = normalized(np.cross(normal, tangent_u))
    return tangent_u, tangent_v


def look_at_camera_to_world(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalized(target - eye)
    right = normalized(np.cross(forward, up))
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def light_pose_from_camera(camera_to_world: np.ndarray, yaw_offset: float, pitch_offset: float) -> np.ndarray:
    pose = np.asarray(camera_to_world, dtype=np.float32)
    camera_forward = -pose[:3, 2]
    camera_right = pose[:3, 0]
    camera_up = pose[:3, 1]
    direction = normalized(camera_forward + math.sin(yaw_offset) * camera_right + math.sin(pitch_offset) * camera_up)
    return look_at_camera_to_world(pose[:3, 3] - direction, pose[:3, 3], camera_up)


def sample_geometry(body_part: str, rng: np.random.Generator) -> tuple[float, float]:
    ranges = {
        "face": (0.0055, 0.016),
        "arms": (0.006, 0.024),
        "hands": (0.004, 0.014),
        "legs": (0.007, 0.030),
        "feet": (0.005, 0.018),
        "front": (0.008, 0.034),
        "back": (0.008, 0.034),
    }
    min_radius, max_radius = ranges[body_part]
    radius = float(np.exp(rng.uniform(np.log(min_radius), np.log(max_radius))))
    height_fraction = float(rng.uniform(0.36, 0.82))
    height = float(np.clip(radius * height_fraction, 0.002, radius * 0.95))
    return radius, height


def spherical_cap_support_radius(radius: float, height: float) -> float:
    return float(np.sqrt(max(0.0, 2.0 * radius * height - height * height)))


def spherical_cap_volume(radius: float, height: float) -> float:
    return float(math.pi * height * height * (3.0 * radius - height) / 3.0)


def spherical_cap_profile(
    radius: float,
    height: float,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support_radius = spherical_cap_support_radius(radius, height)
    local_points = [np.array([0.0, 0.0], dtype=np.float32)]
    profile_heights = [height]
    radial_fraction = [0.0]

    for ring in range(1, radial_segments + 1):
        rho = support_radius * ring / radial_segments
        z_height = math.sqrt(max(0.0, radius * radius - rho * rho)) + height - radius
        for step in range(angular_segments):
            theta = 2.0 * math.pi * step / angular_segments
            local_points.append(np.array([rho * math.cos(theta), rho * math.sin(theta)], dtype=np.float32))
            profile_heights.append(float(z_height))
            radial_fraction.append(float(ring / radial_segments))

    return (
        np.asarray(local_points, dtype=np.float32),
        np.asarray(profile_heights, dtype=np.float32),
        np.asarray(radial_fraction, dtype=np.float32),
    )


def spherical_cap_faces(radial_segments: int, angular_segments: int) -> np.ndarray:
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
    return np.asarray(faces, dtype=np.int32)


def surface_attachment_is_usable(diagnostics: dict[str, float], support_radius: float) -> bool:
    max_allowed_projection = max(0.006, 0.55 * support_radius)
    return (
        diagnostics["contact_label_fraction"] >= 0.82
        and diagnostics["contact_normal_alignment_min"] >= 0.10
        and diagnostics["contact_projection_max_distance_m"] <= max_allowed_projection
    )


def smooth_cap_apex_color(lesion_rgb: np.ndarray, angular_segments: int) -> np.ndarray:
    """Blend the apex color with the first ring to avoid a one-vertex color blip."""
    if len(lesion_rgb) <= angular_segments:
        return lesion_rgb
    smoothed = lesion_rgb.copy()
    first_ring = smoothed[1 : 1 + angular_segments].astype(np.float32)
    smoothed[0] = np.rint(first_ring.mean(axis=0)).astype(np.uint8)
    return smoothed


def build_surface_attached_cap_mesh(
    scan: ScanSurface | None,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    radius: float,
    height: float,
    base_rgb: np.ndarray,
    rng: np.random.Generator,
    radial_segments: int = 12,
    angular_segments: int = 40,
    surface_body_part: str | None = None,
    tint_mean: tuple[float, float, float] = (1.06, 0.94, 0.92),
    tint_noise_sigma: float = 0.018,
    center_highlight_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mottling_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    local_points_arr, profile_arr, radial_arr = spherical_cap_profile(radius, height, radial_segments, angular_segments)
    tangent_plane_points = (
        anchor + local_points_arr[:, 0, None] * tangent_u + local_points_arr[:, 1, None] * tangent_v
    ).astype(np.float32)

    if scan is None:
        skin_points = tangent_plane_points
        surface_normals = np.repeat(normal[None, :], len(local_points_arr), axis=0).astype(np.float32)
        skin_rgb = np.repeat(base_rgb[None, :], len(local_points_arr), axis=0).astype(np.uint8)
        triangle_ids = np.full(len(local_points_arr), -1, dtype=np.int32)
        distances = np.zeros(len(local_points_arr), dtype=np.float32)
    else:
        skin_points, surface_normals, skin_rgb, triangle_ids, distances = scan.project_points_to_surface(
            tangent_plane_points,
            normal_hint=normal,
        )

    vertices = (skin_points + profile_arr[:, None] * surface_normals).astype(np.float32)
    faces = spherical_cap_faces(radial_segments, angular_segments)

    base = skin_rgb.astype(np.float32)
    tint = np.asarray(tint_mean, dtype=np.float32) + rng.normal(0.0, tint_noise_sigma, size=3)
    edge_blend = np.clip((1.0 - radial_arr) * 1.25, 0.0, 1.0)[:, None]
    center_highlight = (1.0 - radial_arr)[:, None] * np.asarray(center_highlight_rgb, dtype=np.float32)
    mottling = rng.normal(0.0, mottling_sigma, size=(len(radial_arr), 3)).astype(np.float32)
    tinted = np.clip(base * tint[None, :] + center_highlight + mottling, 0, 255)
    lesion_rgb = np.clip(base * (1.0 - edge_blend) + tinted * edge_blend, 0, 255).astype(np.uint8)
    lesion_rgb = smooth_cap_apex_color(lesion_rgb, angular_segments)

    contact = radial_arr >= 0.999
    normal_alignment = surface_normals @ normalized(normal)
    if scan is not None and surface_body_part is not None:
        label_id = scan.label_id(surface_body_part)
        contact_label_fraction = float(np.mean(scan.face_labels[triangle_ids[contact]] == label_id))
    else:
        contact_label_fraction = 1.0
    diagnostics = {
        "projection_max_distance_m": float(np.max(distances)),
        "contact_projection_max_distance_m": float(np.max(distances[contact])),
        "normal_alignment_min": float(np.min(normal_alignment)),
        "contact_normal_alignment_min": float(np.min(normal_alignment[contact])),
        "contact_label_fraction": contact_label_fraction,
    }
    return vertices, faces, lesion_rgb, diagnostics


def build_lesion_mesh(
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    radius: float,
    height: float,
    base_rgb: np.ndarray,
    rng: np.random.Generator,
    radial_segments: int = 12,
    angular_segments: int = 40,
    scan: ScanSurface | None = None,
    surface_body_part: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    return build_surface_attached_cap_mesh(
        scan,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        radius,
        height,
        base_rgb,
        rng,
        radial_segments=radial_segments,
        angular_segments=angular_segments,
        surface_body_part=surface_body_part,
    )


def write_lesion_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.export(path)


def depth_visual(depth: np.ndarray) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 0.0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(mask):
        return vis
    near = float(np.percentile(depth[mask], 1))
    far = float(np.percentile(depth[mask], 99))
    if far <= near:
        far = near + 1e-6
    normalized_depth = np.clip((far - depth) / (far - near), 0.0, 1.0)
    vis[mask] = np.rint(normalized_depth[mask] * 255.0).astype(np.uint8)
    return vis


def save_depth_png(depth: np.ndarray, output_path: Path) -> None:
    mask = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[mask] = np.clip(np.rint(depth[mask] * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    imageio.imwrite(output_path, depth_mm)


def camera_for_lesion(
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    radius: float,
    height: float,
    body_part: str,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, float]]:
    fov_deg = float(rng.uniform(38.0, 56.0))
    frame_scale = float(rng.uniform(2.4, 4.2))
    if body_part == "face":
        frame_scale = float(rng.uniform(3.0, 5.2))
    frame_half_height = float(np.clip(radius * frame_scale, 0.045, 0.145))
    distance = max(frame_half_height / math.tan(math.radians(fov_deg) / 2.0), height + 0.050)

    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    off_axis = math.radians(float(rng.uniform(2.0, 18.0)))
    tangent_direction = normalized(math.cos(angle) * tangent_u + math.sin(angle) * tangent_v)
    view_direction = normalized(math.cos(off_axis) * normal + math.sin(off_axis) * tangent_direction)

    target = (
        anchor
        + float(rng.uniform(0.25, 0.65)) * height * normal
        + float(rng.uniform(-0.12, 0.12)) * frame_half_height * tangent_u
        + float(rng.uniform(-0.12, 0.12)) * frame_half_height * tangent_v
    )
    eye = target + distance * view_direction
    roll = math.radians(float(rng.uniform(-18.0, 18.0)))
    up = math.cos(roll) * tangent_v + math.sin(roll) * tangent_u
    up = up - float(np.dot(up, view_direction)) * view_direction
    if float(np.linalg.norm(up)) <= 1e-8:
        up = tangent_u - float(np.dot(tangent_u, view_direction)) * view_direction
    up = normalized(up)
    camera_to_world = look_at_camera_to_world(eye, target, up)

    settings = {
        "fov_deg": fov_deg,
        "frame_scale": frame_scale,
        "frame_half_height_m": frame_half_height,
        "camera_distance_m": float(distance),
        "angle_rad": angle,
        "off_axis_deg": math.degrees(off_axis),
        "roll_deg": math.degrees(roll),
        "ambient": float(rng.uniform(0.34, 0.78)),
        "directional_intensity": float(rng.uniform(0.80, 2.40)),
        "light_yaw_offset": math.radians(float(rng.uniform(-55.0, 55.0))),
        "light_pitch_offset": math.radians(float(rng.uniform(-35.0, 35.0))),
    }
    camera = {
        "eye_xyz": [float(value) for value in eye],
        "target_xyz": [float(value) for value in target],
        "camera_to_world": camera_to_world.tolist(),
        **settings,
    }
    return camera, settings


def render_pair(
    renderer: pyrender.OffscreenRenderer,
    scan: ScanSurface,
    lesion_vertices: np.ndarray,
    lesion_faces: np.ndarray,
    lesion_rgb: np.ndarray,
    camera: dict[str, Any],
    settings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    lesion_colors = np.column_stack([lesion_rgb, np.full(len(lesion_rgb), 255, dtype=np.uint8)])
    lesion_trimesh = trimesh.Trimesh(
        vertices=lesion_vertices,
        faces=lesion_faces,
        vertex_colors=lesion_colors,
        process=False,
    )
    lesion_mesh = pyrender.Mesh.from_trimesh(lesion_trimesh, smooth=True)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[settings["ambient"]] * 3)
    scene.add(scan.base_render_mesh)
    scene.add(lesion_mesh)
    camera_to_world = np.asarray(camera["camera_to_world"], dtype=np.float32)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=settings["directional_intensity"]),
        pose=light_pose_from_camera(camera_to_world, settings["light_yaw_offset"], settings["light_pitch_offset"]),
    )
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(settings["fov_deg"]), znear=0.005, zfar=5.0),
        pose=camera_to_world,
    )
    color, depth = renderer.render(scene)
    return color[:, :, :3].astype(np.uint8), depth.astype(np.float32)


def choose_face(scan: ScanSurface, candidates: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = scan.face_areas[candidates].astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


def build_volume(
    scan: ScanSurface,
    body_part: str,
    patient_volume_index: int,
    seed: int,
    renderer: pyrender.OffscreenRenderer,
    part_root: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    candidates = scan.candidate_faces(body_part)
    radius, height = sample_geometry(body_part, rng)
    support_radius = spherical_cap_support_radius(radius, height)

    best_placement: tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, float],
    ] | None = None
    best_score = -np.inf
    for _ in range(80):
        face_index = choose_face(scan, candidates, rng)
        selected = scan.face_triangles[face_index]
        weights = rng.dirichlet(np.ones(3))
        anchor = (weights @ selected).astype(np.float32)
        normal = normalized(scan.face_normals[face_index])
        tangent_u, tangent_v = tangent_basis(normal)
        base_rgb = scan.face_rgb[face_index].astype(np.uint8)
        lesion_vertices, lesion_faces, lesion_rgb, attachment = build_lesion_mesh(
            anchor,
            normal,
            tangent_u,
            tangent_v,
            radius,
            height,
            base_rgb,
            rng,
            scan=scan,
            surface_body_part=body_part,
        )
        score = (
            attachment["contact_label_fraction"]
            + attachment["contact_normal_alignment_min"]
            - attachment["contact_projection_max_distance_m"] / max(support_radius, 1e-6)
        )
        if score > best_score:
            best_score = score
            best_placement = (
                face_index,
                anchor,
                normal,
                tangent_u,
                tangent_v,
                lesion_vertices,
                lesion_faces,
                lesion_rgb,
                attachment,
            )
        if surface_attachment_is_usable(attachment, support_radius):
            break

    if best_placement is None:
        raise RuntimeError(f"Could not place lesion on {scan.scan_id} {body_part}")
    (
        face_index,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        lesion_vertices,
        lesion_faces,
        lesion_rgb,
        attachment,
    ) = best_placement
    base_rgb = scan.face_rgb[face_index].astype(np.uint8)
    camera, settings = camera_for_lesion(anchor, normal, tangent_u, tangent_v, radius, height, body_part, rng)
    rgb, depth = render_pair(renderer, scan, lesion_vertices, lesion_faces, lesion_rgb, camera, settings)

    sample_id = f"{body_part}_{scan.scan_id}_v{patient_volume_index:03d}"
    data_root = part_root / "data"
    volume_mesh_path = data_root / "volumes" / f"{sample_id}_lesion_volume.ply"
    volume_metadata_path = data_root / "metadata" / f"{sample_id}.json"
    image_path = data_root / "images" / f"{sample_id}_rgb.png"
    depth_npy_path = data_root / "depth" / f"{sample_id}_depth.npy"
    depth_png_path = data_root / "depth" / f"{sample_id}_depth_mm.png"
    depth_vis_path = data_root / "depth_vis" / f"{sample_id}_depth_vis.png"
    for path in [volume_mesh_path.parent, volume_metadata_path.parent, image_path.parent, depth_npy_path.parent, depth_vis_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    write_lesion_ply(volume_mesh_path, lesion_vertices, lesion_faces, lesion_rgb)
    imageio.imwrite(image_path, rgb)
    np.save(depth_npy_path, depth)
    save_depth_png(depth, depth_png_path)
    depth_vis = depth_visual(depth)
    imageio.imwrite(depth_vis_path, depth_vis)

    volume = LesionVolume(
        sample_id=sample_id,
        body_part=body_part,
        scan_id=scan.scan_id,
        patient_volume_index=patient_volume_index,
        seed=seed,
        face_index=face_index,
        radius_m=radius,
        height_m=height,
        support_radius_m=support_radius,
        projection_max_distance_m=attachment["projection_max_distance_m"],
        contact_label_fraction=attachment["contact_label_fraction"],
        spherical_cap_volume_m3=spherical_cap_volume(radius, height),
        spherical_cap_volume_ml=spherical_cap_volume(radius, height) * 1_000_000.0,
        anchor_xyz=[float(value) for value in anchor],
        normal_xyz=[float(value) for value in normal],
        tangent_u_xyz=[float(value) for value in tangent_u],
        tangent_v_xyz=[float(value) for value in tangent_v],
        base_rgb=[int(value) for value in base_rgb],
        lesion_rgb=[int(value) for value in lesion_rgb[0]],
        camera=camera,
    )
    volume_metadata_path.write_text(json.dumps(asdict(volume), indent=2) + "\n", encoding="utf-8")

    valid_depth = int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))
    row = {
        "sample_id": sample_id,
        "body_part": body_part,
        "scan_id": scan.scan_id,
        "patient_volume_index": patient_volume_index,
        "seed": seed,
        "mesh_path": str(volume_mesh_path.relative_to(data_root)),
        "metadata_path": str(volume_metadata_path.relative_to(data_root)),
        "image_path": str(image_path.relative_to(data_root)),
        "depth_npy_path": str(depth_npy_path.relative_to(data_root)),
        "depth_png_path": str(depth_png_path.relative_to(data_root)),
        "depth_vis_path": str(depth_vis_path.relative_to(data_root)),
        "camera_mode": "lesion_closeup",
        "depth_type": "camera_z_distance",
        "depth_visualization": "near_bright_far_dark_background_black_infinite",
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "fov_deg": settings["fov_deg"],
        "radius_m": radius,
        "lesion_height_m": height,
        "support_radius_m": support_radius,
        "projection_max_distance_m": attachment["projection_max_distance_m"],
        "contact_label_fraction": attachment["contact_label_fraction"],
        "spherical_cap_volume_m3": volume.spherical_cap_volume_m3,
        "spherical_cap_volume_ml": volume.spherical_cap_volume_ml,
        "face_index": face_index,
        "valid_depth_pixels": valid_depth,
        "eye_xyz": json.dumps(camera["eye_xyz"]),
        "target_xyz": json.dumps(camera["target_xyz"]),
    }
    return row


def write_manifest(part_root: Path, visualization_part_root: Path, rows: list[dict[str, Any]]) -> None:
    data_root = part_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "body_part",
        "scan_id",
        "patient_volume_index",
        "seed",
        "mesh_path",
        "metadata_path",
        "image_path",
        "depth_npy_path",
        "depth_png_path",
        "depth_vis_path",
        "camera_mode",
        "depth_type",
        "depth_visualization",
        "width",
        "height",
        "fov_deg",
        "radius_m",
        "lesion_height_m",
        "support_radius_m",
        "projection_max_distance_m",
        "contact_label_fraction",
        "spherical_cap_volume_m3",
        "spherical_cap_volume_ml",
        "face_index",
        "valid_depth_pixels",
        "eye_xyz",
        "target_xyz",
    ]
    with (data_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    by_scan = {scan_id: sum(1 for row in rows if row["scan_id"] == scan_id) for scan_id in SCAN_IDS}
    summary = {
        "body_part": part_root.name,
        "sample_count": len(rows),
        "volume_count": len(rows),
        "rgb_depth_pair_count": len(rows),
        "samples_by_scan": by_scan,
        "data_root": root_relative(data_root),
        "visualization_root": root_relative(visualization_part_root),
        "manifest": root_relative(data_root / "manifest.csv"),
        "camera_mode": "lesion_closeup",
        "depth_type": "camera_z_distance",
        "volume_shape": "spherical_cap_nf_like",
    }
    (data_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (part_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_visualizations(part_roots: list[Path], visualization_part_roots: list[Path], frame_count: int, tile_size: int) -> None:
    script = ROOT / "code" / "depth_maps" / "scripts" / "build_depth_dataset_visualizations.py"
    command = [
        sys.executable,
        str(script),
        *[root_relative(path) for path in part_roots],
        "--visualization-root",
        *[root_relative(path) for path in visualization_part_roots],
        "--frame_count",
        str(frame_count),
        "--tile_size",
        str(tile_size),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def resolve_sample_counts(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    body_parts = [body_part for body_part in args.body_part if body_part != "clothes"]
    groups = [(body_part, scan_id) for body_part in body_parts for scan_id in args.scan_id]
    if args.target_total_rgb_depth_pairs is not None and args.target_rgb_depth_pairs_per_body_part is not None:
        raise ValueError("Use either --target-total-rgb-depth-pairs or --target-rgb-depth-pairs-per-body-part, not both")

    if args.target_rgb_depth_pairs_per_body_part is not None:
        if args.target_rgb_depth_pairs_per_body_part < len(args.scan_id):
            raise ValueError("--target-rgb-depth-pairs-per-body-part must be at least the number of scans")
        sample_counts: dict[tuple[str, str], int] = {}
        base_count, remainder = divmod(args.target_rgb_depth_pairs_per_body_part, len(args.scan_id))
        for body_part in body_parts:
            for scan_idx, scan_id in enumerate(args.scan_id):
                sample_counts[(body_part, scan_id)] = base_count + (1 if scan_idx < remainder else 0)
        return sample_counts

    if args.target_total_rgb_depth_pairs is None:
        return {
            group: args.volumes_per_patient_per_body_part
            for group in groups
        }

    min_total = 10 * len(groups)
    max_total = 100 * len(groups)
    if args.target_total_rgb_depth_pairs < min_total or args.target_total_rgb_depth_pairs > max_total:
        raise ValueError(
            "--target-total-rgb-depth-pairs must allow 10-100 samples per patient/body-part "
            f"({min_total}-{max_total} for this selection)"
        )

    base_count, remainder = divmod(args.target_total_rgb_depth_pairs, len(groups))
    return {
        group: base_count + (1 if group_idx < remainder else 0)
        for group_idx, group in enumerate(groups)
    }


def build_dataset(args: argparse.Namespace) -> None:
    output_root = resolve_output_root(args.output_root)
    visualization_root = resolve_root(args.visualization_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if visualization_root.exists() and args.overwrite:
        shutil.rmtree(visualization_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visualization_root.mkdir(parents=True, exist_ok=True)

    scans = {scan_id: ScanSurface(scan_id) for scan_id in args.scan_id}
    sample_counts = resolve_sample_counts(args)
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    overall_summary: dict[str, Any] = {
        "dataset": "body_parts",
        "output_root": root_relative(output_root),
        "visualization_root": root_relative(visualization_root),
        "body_parts": args.body_part,
        "scan_ids": args.scan_id,
        "volumes_per_patient_per_body_part": args.volumes_per_patient_per_body_part,
        "target_total_rgb_depth_pairs": args.target_total_rgb_depth_pairs,
        "target_rgb_depth_pairs_per_body_part": args.target_rgb_depth_pairs_per_body_part,
        "actual_volumes_per_patient_per_body_part": {
            f"{body_part}/{scan_id}": count
            for (body_part, scan_id), count in sample_counts.items()
        },
        "image_size": args.image_size,
        "seed": args.seed,
        "total_rgb_depth_pairs": 0,
        "parts": {},
    }
    part_roots = []
    visualization_part_roots = []
    try:
        for body_part_index, body_part in enumerate(args.body_part):
            if body_part == "clothes":
                continue
            part_root = output_root / body_part
            visualization_part_root = visualization_root / body_part
            part_roots.append(part_root)
            visualization_part_roots.append(visualization_part_root)
            if part_root.exists() and args.overwrite:
                shutil.rmtree(part_root)
            if visualization_part_root.exists() and args.overwrite:
                shutil.rmtree(visualization_part_root)
            rows: list[dict[str, Any]] = []
            for scan_index, scan_id in enumerate(args.scan_id):
                scan = scans[scan_id]
                volume_count = sample_counts[(body_part, scan_id)]
                for patient_volume_index in range(volume_count):
                    seed = args.seed + body_part_index * 1_000_000 + scan_index * 100_000 + patient_volume_index
                    row = build_volume(
                        scan,
                        body_part,
                        patient_volume_index,
                        seed,
                        renderer,
                        part_root,
                    )
                    rows.append(row)
                    print(
                        f"[{body_part}] {scan_id} {patient_volume_index + 1:03d}/"
                        f"{volume_count:03d} -> {row['sample_id']}",
                        flush=True,
                    )

            write_manifest(part_root, visualization_part_root, rows)
            overall_summary["total_rgb_depth_pairs"] += len(rows)
            overall_summary["parts"][body_part] = {
                "folder": root_relative(part_root),
                "visualization_folder": root_relative(visualization_part_root),
                "sample_count": len(rows),
                "manifest": root_relative(part_root / "data" / "manifest.csv"),
                "summary": root_relative(part_root / "summary.json"),
            }
    finally:
        renderer.delete()

    (output_root / "summary.json").write_text(json.dumps(overall_summary, indent=2) + "\n", encoding="utf-8")
    if args.build_visualizations:
        build_visualizations(part_roots, visualization_part_roots, args.visualization_frames, args.visualization_tile_size)
    print(json.dumps(overall_summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT.relative_to(ROOT)))
    parser.add_argument("--visualization-root", default=str(DEFAULT_VISUALIZATION_ROOT.relative_to(ROOT)))
    parser.add_argument("--body-part", action="append", choices=BODY_PARTS, default=None)
    parser.add_argument("--scan-id", action="append", choices=SCAN_IDS, default=None)
    parser.add_argument("--volumes-per-patient-per-body-part", type=int, default=100)
    parser.add_argument("--target-total-rgb-depth-pairs", type=int, default=None)
    parser.add_argument("--target-rgb-depth-pairs-per-body-part", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-frames", type=int, default=32)
    parser.add_argument("--visualization-tile-size", type=int, default=192)
    args = parser.parse_args()
    if args.body_part is None:
        args.body_part = BODY_PARTS
    if args.scan_id is None:
        args.scan_id = SCAN_IDS
    if args.volumes_per_patient_per_body_part < 10 or args.volumes_per_patient_per_body_part > 100:
        raise ValueError("--volumes-per-patient-per-body-part must be between 10 and 100")
    return args


if __name__ == "__main__":
    build_dataset(build_parser())

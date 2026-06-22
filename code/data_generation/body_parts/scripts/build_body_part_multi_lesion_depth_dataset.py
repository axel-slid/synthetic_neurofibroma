#!/usr/bin/env python3
"""Generate body-part-framed RGB/depth pairs with 10-100 synthetic lesions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh

from build_body_part_volume_depth_dataset import (
    BODY_PARTS,
    SCAN_IDS,
    ROOT,
    ScanSurface,
    build_surface_attached_cap_mesh,
    depth_visual,
    light_pose_from_camera,
    look_at_camera_to_world,
    normalized,
    root_relative,
    save_depth_png,
    spherical_cap_support_radius,
    spherical_cap_volume,
    surface_attachment_is_usable,
    tangent_basis,
    write_lesion_ply,
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "synthetic"
    / "body_parts_multi_lesion"
)
DEFAULT_VISUALIZATION_ROOT = (
    ROOT
    / "data"
    / "synthetic"
    / "multiple_lesion"
    / "visualization"
    / "physics_aug_growth"
    / "body_parts_multi_lesion"
)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
SPLIT_REGION_PARTS = {"arms", "hands", "legs", "feet"}
FRONT_DIRECTION_FLIP_SCAN_IDS = {"HSR0152-Body-090"}
SEGMENTATION_LABEL_OVERRIDES: dict[tuple[str, str], str] = {}


@dataclass
class LesionRecord:
    lesion_index: int
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


@dataclass
class MultiLesionSample:
    sample_id: str
    body_part: str
    source_segmentation_body_part: str
    scan_id: str
    patient_volume_index: int
    seed: int
    lesion_count: int
    lesion_count_range: list[int]
    body_region: str
    camera_mode: str
    depth_type: str
    lesion_pattern_source: str
    camera: dict[str, Any]
    lesions: list[LesionRecord]


def resolve_output_root(path_value: str | None) -> Path:
    if path_value is None:
        return DEFAULT_OUTPUT_ROOT
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def resolve_root(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalized(axis)
    vector = np.asarray(vector, dtype=np.float32)
    return normalized(
        vector * math.cos(angle_rad)
        + np.cross(axis, vector) * math.sin(angle_rad)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle_rad))
    )


def horizontal_front_direction(scan: ScanSurface) -> np.ndarray:
    front_id = scan.label_id("front")
    back_id = scan.label_id("back")
    front_center = scan.face_centroids[scan.face_labels == front_id].mean(axis=0)
    back_center = scan.face_centroids[scan.face_labels == back_id].mean(axis=0)
    direction = front_center - back_center
    direction[2] = 0.0
    if float(np.linalg.norm(direction)) <= 1e-8:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    direction = normalized(direction)
    if scan.scan_id in FRONT_DIRECTION_FLIP_SCAN_IDS:
        direction = -direction
    return direction


def segmentation_body_part(scan: ScanSurface, body_part: str) -> str:
    return SEGMENTATION_LABEL_OVERRIDES.get((scan.scan_id, body_part), body_part)


def view_direction_for_body_part(scan: ScanSurface, body_part: str, rng: np.random.Generator) -> np.ndarray:
    front = horizontal_front_direction(scan)
    if body_part == "back":
        direction = -front
    elif body_part == "feet":
        direction = normalized(0.82 * front + 0.42 * WORLD_UP)
    elif body_part in {"legs", "hands"}:
        direction = normalized(0.97 * front + 0.12 * WORLD_UP)
    else:
        direction = normalized(0.995 * front + 0.04 * WORLD_UP)

    yaw_jitter = math.radians(float(rng.uniform(-5.0, 5.0)))
    direction = rotate_about_axis(direction, WORLD_UP, yaw_jitter)
    return direction


def select_body_part_region(scan: ScanSurface, body_part: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    candidates = scan.candidate_faces(segmentation_body_part(scan, body_part))
    if body_part not in SPLIT_REGION_PARTS:
        return candidates, "full"

    centroids = scan.face_centroids[candidates]
    split_x = float(scan.center[0])
    use_positive_x = bool(rng.integers(0, 2))
    if use_positive_x:
        region_candidates = candidates[centroids[:, 0] >= split_x]
        region = "x_positive"
    else:
        region_candidates = candidates[centroids[:, 0] < split_x]
        region = "x_negative"
    if len(region_candidates) < 64:
        return candidates, "full"
    return region_candidates, region


def part_vertices(scan: ScanSurface, candidates: np.ndarray) -> np.ndarray:
    vertex_indices = np.unique(scan.faces[candidates].reshape(-1))
    return scan.vertices[vertex_indices]


def camera_for_body_part(
    scan: ScanSurface,
    body_part: str,
    region_candidates: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    view_direction = view_direction_for_body_part(scan, body_part, rng)
    vertices = part_vertices(scan, region_candidates)
    target = ((vertices.min(axis=0) + vertices.max(axis=0)) * 0.5).astype(np.float32)

    up = WORLD_UP - float(np.dot(WORLD_UP, view_direction)) * view_direction
    if float(np.linalg.norm(up)) <= 1e-8:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = normalized(up)

    right = normalized(np.cross(-view_direction, up))
    up = normalized(np.cross(right, -view_direction))

    projected = vertices - target
    half_width = float(np.max(np.abs(projected @ right)))
    half_height = float(np.max(np.abs(projected @ up)))
    fov_deg = float(rng.uniform(34.0, 46.0))
    if body_part in {"hands", "feet", "face"}:
        fov_deg = float(rng.uniform(36.0, 50.0))

    margin = float(rng.uniform(1.10, 1.28))
    if body_part in {"arms", "hands", "legs", "feet"}:
        margin = float(rng.uniform(1.14, 1.34))
    frame_half_extent = max(half_width, half_height) * margin
    distance = max(frame_half_extent / math.tan(math.radians(fov_deg) / 2.0), 0.12)

    eye = target + distance * view_direction
    roll = math.radians(float(rng.uniform(-4.0, 4.0)))
    rolled_up = normalized(math.cos(roll) * up + math.sin(roll) * right)
    camera_to_world = look_at_camera_to_world(eye, target, rolled_up)

    settings = {
        "fov_deg": fov_deg,
        "frame_margin": margin,
        "frame_half_width_m": half_width * margin,
        "frame_half_height_m": half_height * margin,
        "camera_distance_m": float(distance),
        "roll_deg": math.degrees(roll),
        "ambient": float(rng.uniform(0.42, 0.78)),
        "directional_intensity": float(rng.uniform(0.75, 1.80)),
        "light_yaw_offset": math.radians(float(rng.uniform(-24.0, 24.0))),
        "light_pitch_offset": math.radians(float(rng.uniform(-18.0, 18.0))),
    }
    camera = {
        "eye_xyz": [float(value) for value in eye],
        "target_xyz": [float(value) for value in target],
        "camera_to_world": camera_to_world.tolist(),
        "view_direction_xyz": [float(value) for value in view_direction],
        **settings,
    }
    return camera, settings, view_direction


def choose_closeup_lesion(lesions: list[LesionRecord], rng: np.random.Generator) -> LesionRecord:
    weights = np.asarray([max(lesion.support_radius_m, 1e-5) ** 2 for lesion in lesions], dtype=np.float64)
    weights /= weights.sum()
    return lesions[int(rng.choice(len(lesions), p=weights))]


def camera_for_lesion_closeup(
    lesions: list[LesionRecord],
    body_part: str,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, float], LesionRecord]:
    target_lesion = choose_closeup_lesion(lesions, rng)
    anchor = np.asarray(target_lesion.anchor_xyz, dtype=np.float32)
    normal = normalized(np.asarray(target_lesion.normal_xyz, dtype=np.float32))
    tangent_u = normalized(np.asarray(target_lesion.tangent_u_xyz, dtype=np.float32))
    tangent_v = normalized(np.asarray(target_lesion.tangent_v_xyz, dtype=np.float32))

    azimuth = math.radians(float(rng.uniform(0.0, 360.0)))
    tangent_direction = normalized(math.cos(azimuth) * tangent_u + math.sin(azimuth) * tangent_v)
    off_axis = math.radians(float(rng.uniform(0.0, 32.0)))
    view_direction = normalized(math.cos(off_axis) * normal + math.sin(off_axis) * tangent_direction)

    fov_deg = float(rng.uniform(30.0, 56.0))
    if body_part in {"face", "hands", "feet"}:
        fov_deg = float(rng.uniform(34.0, 60.0))

    min_half_heights = {
        "face": 0.028,
        "hands": 0.030,
        "feet": 0.034,
        "arms": 0.044,
        "legs": 0.052,
        "front": 0.060,
        "back": 0.060,
    }
    max_half_heights = {
        "face": 0.095,
        "hands": 0.105,
        "feet": 0.115,
        "arms": 0.145,
        "legs": 0.175,
        "front": 0.210,
        "back": 0.210,
    }
    frame_scale = float(rng.uniform(4.0, 9.0))
    if body_part in {"front", "back", "legs"}:
        frame_scale = float(rng.uniform(5.0, 11.5))
    frame_half_height = float(
        np.clip(
            max(target_lesion.support_radius_m, target_lesion.radius_m) * frame_scale,
            min_half_heights[body_part],
            max_half_heights[body_part],
        )
    )
    frame_half_width = float(frame_half_height * rng.uniform(0.92, 1.16))
    target_u_offset = float(rng.uniform(-0.32, 0.32) * frame_half_width)
    target_v_offset = float(rng.uniform(-0.32, 0.32) * frame_half_height)
    target_normal_offset = float(rng.uniform(0.12, 0.68) * target_lesion.height_m)
    target = (
        anchor
        + target_normal_offset * normal
        + target_u_offset * tangent_u
        + target_v_offset * tangent_v
    )

    distance = max(
        max(frame_half_height, frame_half_width) / math.tan(math.radians(fov_deg) / 2.0),
        target_lesion.height_m + 0.035,
    )
    eye = target + distance * view_direction

    roll = math.radians(float(rng.uniform(-24.0, 24.0)))
    up = math.cos(roll) * tangent_v + math.sin(roll) * tangent_u
    up = up - float(np.dot(up, view_direction)) * view_direction
    if float(np.linalg.norm(up)) <= 1e-8:
        up = tangent_u - float(np.dot(tangent_u, view_direction)) * view_direction
    up = normalized(up)
    camera_to_world = look_at_camera_to_world(eye, target, up)

    settings = {
        "fov_deg": fov_deg,
        "frame_margin": frame_scale,
        "frame_half_width_m": frame_half_width,
        "frame_half_height_m": frame_half_height,
        "camera_distance_m": float(distance),
        "roll_deg": math.degrees(roll),
        "off_axis_deg": math.degrees(off_axis),
        "target_u_offset_m": target_u_offset,
        "target_v_offset_m": target_v_offset,
        "target_normal_offset_m": target_normal_offset,
        "ambient": float(rng.uniform(0.36, 0.82)),
        "directional_intensity": float(rng.uniform(0.70, 2.35)),
        "light_yaw_offset": math.radians(float(rng.uniform(-62.0, 62.0))),
        "light_pitch_offset": math.radians(float(rng.uniform(-42.0, 42.0))),
    }
    camera = {
        "eye_xyz": [float(value) for value in eye],
        "target_xyz": [float(value) for value in target],
        "camera_to_world": camera_to_world.tolist(),
        "view_direction_xyz": [float(value) for value in view_direction],
        "target_lesion_index": int(target_lesion.lesion_index),
        "target_lesion_face_index": int(target_lesion.face_index),
        "target_lesion_anchor_xyz": [float(value) for value in anchor],
        "target_lesion_normal_xyz": [float(value) for value in normal],
        **settings,
    }
    return camera, settings, target_lesion


def visible_candidates(scan: ScanSurface, body_part: str, view_direction: np.ndarray, region_candidates: np.ndarray) -> np.ndarray:
    candidates = region_candidates
    facing = scan.face_normals[candidates] @ view_direction
    threshold = 0.12
    if body_part in {"hands", "feet", "legs"}:
        threshold = 0.02
    visible = candidates[facing > threshold]
    if len(visible) < 64:
        visible = candidates[facing > -0.08]
    if len(visible) < 64:
        visible = candidates
    return visible


def sample_multi_lesion_geometry(body_part: str, lesion_count: int, rng: np.random.Generator) -> tuple[float, float]:
    ranges = {
        "face": (0.0024, 0.0085),
        "arms": (0.0030, 0.0140),
        "hands": (0.0022, 0.0085),
        "legs": (0.0034, 0.0155),
        "feet": (0.0026, 0.0100),
        "front": (0.0040, 0.0180),
        "back": (0.0040, 0.0180),
    }
    min_radius, max_radius = ranges[body_part]
    crowd_scale = float(np.interp(lesion_count, [10, 100], [1.0, 0.62]))
    radius = float(np.exp(rng.uniform(np.log(min_radius), np.log(max_radius)))) * crowd_scale
    if rng.random() < 0.08:
        radius *= float(rng.uniform(1.25, 1.80))
    radius = float(np.clip(radius, min_radius * 0.72, max_radius))
    height_fraction = float(rng.uniform(0.28, 0.74))
    height = float(np.clip(radius * height_fraction, 0.0012, radius * 0.90))
    return radius, height


def choose_face(scan: ScanSurface, candidates: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = scan.face_areas[candidates].astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


def build_multi_lesion_mesh(
    scan: ScanSurface,
    body_part: str,
    lesion_count: int,
    view_direction: np.ndarray,
    region_candidates: np.ndarray,
    rng: np.random.Generator,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[LesionRecord], int]:
    candidates = visible_candidates(scan, body_part, view_direction, region_candidates)
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    lesion_records: list[LesionRecord] = []
    vertex_offset = 0

    for lesion_index in range(lesion_count):
        radius, height = sample_multi_lesion_geometry(body_part, lesion_count, rng)
        support_radius = spherical_cap_support_radius(radius, height)
        surface_part = segmentation_body_part(scan, body_part)
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
            tangent_u, tangent_v = tangent_basis(normal, WORLD_UP)
            base_rgb = scan.face_rgb[face_index].astype(np.uint8)
            lesion_vertices, lesion_faces, lesion_rgb, attachment = build_textured_cap(
                scan,
                anchor,
                normal,
                tangent_u,
                tangent_v,
                radius,
                height,
                base_rgb,
                rng,
                radial_segments,
                angular_segments,
                surface_part,
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
        vertices_parts.append(lesion_vertices)
        faces_parts.append(lesion_faces + vertex_offset)
        rgb_parts.append(lesion_rgb)
        vertex_offset += len(lesion_vertices)

        volume_m3 = spherical_cap_volume(radius, height)
        lesion_records.append(
            LesionRecord(
                lesion_index=lesion_index,
                face_index=face_index,
                radius_m=radius,
                height_m=height,
                support_radius_m=support_radius,
                projection_max_distance_m=attachment["projection_max_distance_m"],
                contact_label_fraction=attachment["contact_label_fraction"],
                spherical_cap_volume_m3=volume_m3,
                spherical_cap_volume_ml=volume_m3 * 1_000_000.0,
                anchor_xyz=[float(value) for value in anchor],
                normal_xyz=[float(value) for value in normal],
                tangent_u_xyz=[float(value) for value in tangent_u],
                tangent_v_xyz=[float(value) for value in tangent_v],
                base_rgb=[int(value) for value in base_rgb],
                lesion_rgb=[int(value) for value in lesion_rgb[0]],
            )
        )

    return (
        np.vstack(vertices_parts).astype(np.float32),
        np.vstack(faces_parts).astype(np.int32),
        np.vstack(rgb_parts).astype(np.uint8),
        lesion_records,
        len(candidates),
    )


def build_textured_cap(
    scan: ScanSurface,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    radius: float,
    height: float,
    base_rgb: np.ndarray,
    rng: np.random.Generator,
    radial_segments: int,
    angular_segments: int,
    surface_body_part: str,
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
        tint_mean=(1.07, 0.93, 0.91),
        tint_noise_sigma=0.028,
        center_highlight_rgb=(0.0, 0.0, 0.0),
        mottling_sigma=2.2,
    )


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


def row_paths(output_root: Path, body_part: str, sample_id: str) -> dict[str, Path]:
    part_data = output_root / "data" / body_part
    return {
        "volume_mesh": part_data / "volumes" / f"{sample_id}_multi_lesion_volume.ply",
        "metadata": part_data / "metadata" / f"{sample_id}.json",
        "image": part_data / "images" / f"{sample_id}_rgb.png",
        "depth_npy": part_data / "depth" / f"{sample_id}_depth.npy",
        "depth_png": part_data / "depth" / f"{sample_id}_depth_mm.png",
        "depth_vis": part_data / "depth_vis" / f"{sample_id}_depth_vis.png",
    }


def build_sample(
    scan: ScanSurface,
    body_part: str,
    patient_volume_index: int,
    seed: int,
    renderer: pyrender.OffscreenRenderer,
    output_root: Path,
    lesion_min: int,
    lesion_max: int,
    radial_segments: int,
    angular_segments: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lesion_count = int(rng.integers(lesion_min, lesion_max + 1))
    region_candidates, body_region = select_body_part_region(scan, body_part, rng)
    _, _, view_direction = camera_for_body_part(scan, body_part, region_candidates, rng)
    lesion_vertices, lesion_faces, lesion_rgb, lesions, visible_face_count = build_multi_lesion_mesh(
        scan,
        body_part,
        lesion_count,
        view_direction,
        region_candidates,
        rng,
        radial_segments,
        angular_segments,
    )
    camera, settings, target_lesion = camera_for_lesion_closeup(lesions, body_part, rng)
    rgb, depth = render_pair(renderer, scan, lesion_vertices, lesion_faces, lesion_rgb, camera, settings)

    sample_id = f"{body_part}_{scan.scan_id}_multi_v{patient_volume_index:03d}"
    paths = row_paths(output_root, body_part, sample_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_lesion_ply(paths["volume_mesh"], lesion_vertices, lesion_faces, lesion_rgb)
    imageio.imwrite(paths["image"], rgb)
    np.save(paths["depth_npy"], depth)
    save_depth_png(depth, paths["depth_png"])
    imageio.imwrite(paths["depth_vis"], depth_visual(depth))

    sample = MultiLesionSample(
        sample_id=sample_id,
        body_part=body_part,
        source_segmentation_body_part=segmentation_body_part(scan, body_part),
        scan_id=scan.scan_id,
        patient_volume_index=patient_volume_index,
        seed=seed,
        lesion_count=lesion_count,
        lesion_count_range=[lesion_min, lesion_max],
        body_region=body_region,
        camera_mode="lesion_closeup_random",
        depth_type="camera_z_distance",
        lesion_pattern_source="10-100 random spherical-cap NF-like lesions with interpolated skin-color texture",
        camera=camera,
        lesions=lesions,
    )
    paths["metadata"].write_text(json.dumps(asdict(sample), indent=2) + "\n", encoding="utf-8")

    radii = np.asarray([lesion.radius_m for lesion in lesions], dtype=np.float64)
    heights = np.asarray([lesion.height_m for lesion in lesions], dtype=np.float64)
    supports = np.asarray([lesion.support_radius_m for lesion in lesions], dtype=np.float64)
    projection_distances = np.asarray([lesion.projection_max_distance_m for lesion in lesions], dtype=np.float64)
    contact_fractions = np.asarray([lesion.contact_label_fraction for lesion in lesions], dtype=np.float64)
    volumes = np.asarray([lesion.spherical_cap_volume_ml for lesion in lesions], dtype=np.float64)
    valid_depth = int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))
    row = {
        "sample_id": sample_id,
        "body_part": body_part,
        "source_segmentation_body_part": segmentation_body_part(scan, body_part),
        "scan_id": scan.scan_id,
        "patient_volume_index": patient_volume_index,
        "seed": seed,
        "image_path": root_relative(paths["image"]),
        "depth_npy_path": root_relative(paths["depth_npy"]),
        "depth_png_path": root_relative(paths["depth_png"]),
        "depth_vis_path": root_relative(paths["depth_vis"]),
        "volume_mesh_path": root_relative(paths["volume_mesh"]),
        "metadata_path": root_relative(paths["metadata"]),
        "camera_mode": sample.camera_mode,
        "depth_type": sample.depth_type,
        "lesion_pattern_source": sample.lesion_pattern_source,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "valid_depth_pixels": valid_depth,
        "lesion_count": lesion_count,
        "lesion_count_min": lesion_min,
        "lesion_count_max": lesion_max,
        "body_region": body_region,
        "radius_m": float(radii.mean()),
        "radius_min_m": float(radii.min()),
        "radius_mean_m": float(radii.mean()),
        "radius_max_m": float(radii.max()),
        "lesion_height_m": float(heights.mean()),
        "lesion_height_min_m": float(heights.min()),
        "lesion_height_mean_m": float(heights.mean()),
        "lesion_height_max_m": float(heights.max()),
        "support_radius_m": float(supports.mean()),
        "support_radius_min_m": float(supports.min()),
        "support_radius_mean_m": float(supports.mean()),
        "support_radius_max_m": float(supports.max()),
        "projection_max_distance_m": float(projection_distances.max()),
        "projection_mean_distance_m": float(projection_distances.mean()),
        "contact_label_fraction_min": float(contact_fractions.min()),
        "contact_label_fraction_mean": float(contact_fractions.mean()),
        "spherical_cap_volume_ml": float(volumes.sum()),
        "total_spherical_cap_volume_ml": float(volumes.sum()),
        "visible_candidate_faces": visible_face_count,
        "fov_deg": settings["fov_deg"],
        "roll_deg": settings["roll_deg"],
        "off_axis_deg": settings["off_axis_deg"],
        "frame_margin": settings["frame_margin"],
        "frame_half_width_m": settings["frame_half_width_m"],
        "frame_half_height_m": settings["frame_half_height_m"],
        "camera_distance_m": settings["camera_distance_m"],
        "target_lesion_index": int(target_lesion.lesion_index),
        "target_lesion_face_index": int(target_lesion.face_index),
        "target_lesion_radius_m": float(target_lesion.radius_m),
        "target_lesion_height_m": float(target_lesion.height_m),
        "target_u_offset_m": settings["target_u_offset_m"],
        "target_v_offset_m": settings["target_v_offset_m"],
        "target_normal_offset_m": settings["target_normal_offset_m"],
        "ambient": settings["ambient"],
        "directional_intensity": settings["directional_intensity"],
        "light_yaw_offset": settings["light_yaw_offset"],
        "light_pitch_offset": settings["light_pitch_offset"],
        "eye_xyz": json.dumps(camera["eye_xyz"]),
        "target_xyz": json.dumps(camera["target_xyz"]),
        "view_direction_xyz": json.dumps(camera["view_direction_xyz"]),
        "camera_to_world": json.dumps(camera["camera_to_world"]),
    }
    return row


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_manifests(
    output_root: Path,
    visualization_root: Path,
    body_parts: list[str],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    data_root = output_root / "data"
    manifest_csv = data_root / "camera_depth_manifest.csv"
    manifest_jsonl = data_root / "camera_depth_manifest.jsonl"
    write_rows_csv(manifest_csv, rows)
    write_jsonl(manifest_jsonl, rows)

    by_part = {body_part: sum(row["body_part"] == body_part for row in rows) for body_part in body_parts}
    by_scan: dict[str, int] = {}
    lesion_counts = [int(row["lesion_count"]) for row in rows]
    part_manifests: dict[str, Any] = {}
    for row in rows:
        by_scan[row["scan_id"]] = by_scan.get(row["scan_id"], 0) + 1

    for body_part in body_parts:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        part_dir = data_root / body_part
        write_rows_csv(part_dir / "manifest.csv", part_rows)
        write_jsonl(part_dir / "manifest.jsonl", part_rows)
        part_summary = {
            "body_part": body_part,
            "sample_count": len(part_rows),
            "rgb_depth_pair_count": len(part_rows),
            "samples_by_scan": {
                scan_id: sum(row["scan_id"] == scan_id for row in part_rows)
                for scan_id in args.scan_id
            },
            "data_root": root_relative(part_dir),
            "manifest": root_relative(part_dir / "manifest.csv"),
            "camera_mode": "lesion_closeup_random",
            "depth_type": "camera_z_distance",
            "volume_shape": "multi_spherical_cap_nf_like",
            "lesion_count_min": min(int(row["lesion_count"]) for row in part_rows) if part_rows else None,
            "lesion_count_max": max(int(row["lesion_count"]) for row in part_rows) if part_rows else None,
        }
        (part_dir / "summary.json").write_text(json.dumps(part_summary, indent=2) + "\n", encoding="utf-8")
        part_manifests[body_part] = {
            "manifest_csv": root_relative(part_dir / "manifest.csv"),
            "manifest_jsonl": root_relative(part_dir / "manifest.jsonl"),
            "sample_count": len(part_rows),
        }

    summary = {
        "dataset": "body_parts_multi_lesion",
        "output_root": root_relative(output_root),
        "visualization_root": root_relative(visualization_root),
        "body_parts": body_parts,
        "scan_ids": args.scan_id,
        "camera_depth_row_count": len(rows),
        "unique_camera_setting_count": len({row["camera_to_world"] for row in rows}),
        "depth_map_count": sum(1 for row in rows if (ROOT / row["depth_npy_path"]).exists()),
        "depth_png_count": sum(1 for row in rows if (ROOT / row["depth_png_path"]).exists()),
        "image_count": sum(1 for row in rows if (ROOT / row["image_path"]).exists()),
        "volume_mesh_count": sum(1 for row in rows if (ROOT / row["volume_mesh_path"]).exists()),
        "by_part": by_part,
        "by_scan": by_scan,
        "lesion_count_min": min(lesion_counts) if lesion_counts else None,
        "lesion_count_max": max(lesion_counts) if lesion_counts else None,
        "lesion_count_mean": float(np.mean(lesion_counts)) if lesion_counts else None,
        "camera_mode": "lesion_closeup_random",
        "framing": "random close-up camera centered near a sampled visible lesion",
        "lesion_pattern_source": "10-100 random spherical-cap NF-like lesions with interpolated skin-color texture per image",
        "volume_shape": "multi_spherical_cap_nf_like",
        "camera_depth_manifest_csv": root_relative(manifest_csv),
        "camera_depth_manifest_jsonl": root_relative(manifest_jsonl),
        "part_manifests": part_manifests,
    }
    (data_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    visualization_root.mkdir(parents=True, exist_ok=True)
    return summary


def resolve_sample_counts(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    groups = [(body_part, scan_id) for body_part in args.body_part for scan_id in args.scan_id]
    if args.target_rgb_depth_pairs_per_body_part is not None:
        if args.target_rgb_depth_pairs_per_body_part < len(args.scan_id):
            raise ValueError("--target-rgb-depth-pairs-per-body-part must be at least the number of scans")
        sample_counts: dict[tuple[str, str], int] = {}
        base_count, remainder = divmod(args.target_rgb_depth_pairs_per_body_part, len(args.scan_id))
        for body_part in args.body_part:
            for scan_idx, scan_id in enumerate(args.scan_id):
                sample_counts[(body_part, scan_id)] = base_count + (1 if scan_idx < remainder else 0)
        return sample_counts
    return {group: args.samples_per_scan_per_body_part for group in groups}


def build_dataset(args: argparse.Namespace) -> None:
    output_root = resolve_output_root(args.output_root)
    visualization_root = resolve_root(args.visualization_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if visualization_root.exists() and args.overwrite:
        shutil.rmtree(visualization_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    visualization_root.mkdir(parents=True, exist_ok=True)

    scans = {scan_id: ScanSurface(scan_id) for scan_id in args.scan_id}
    sample_counts = resolve_sample_counts(args)
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    rows: list[dict[str, Any]] = []
    try:
        for body_part_index, body_part in enumerate(args.body_part):
            for scan_index, scan_id in enumerate(args.scan_id):
                volume_count = sample_counts[(body_part, scan_id)]
                scan = scans[scan_id]
                for patient_volume_index in range(volume_count):
                    seed = args.seed + body_part_index * 1_000_000 + scan_index * 100_000 + patient_volume_index
                    row = build_sample(
                        scan,
                        body_part,
                        patient_volume_index,
                        seed,
                        renderer,
                        output_root,
                        args.lesion_count_min,
                        args.lesion_count_max,
                        args.radial_segments,
                        args.angular_segments,
                    )
                    rows.append(row)
                    print(
                        f"[{body_part}] {scan_id} {patient_volume_index + 1:03d}/"
                        f"{volume_count:03d} lesions={row['lesion_count']:03d} -> {row['sample_id']}",
                        flush=True,
                    )
    finally:
        renderer.delete()

    summary = write_manifests(output_root, visualization_root, args.body_part, rows, args)
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=root_relative(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--visualization-root", default=root_relative(DEFAULT_VISUALIZATION_ROOT))
    parser.add_argument("--body-part", action="append", choices=BODY_PARTS, default=None)
    parser.add_argument("--scan-id", action="append", choices=SCAN_IDS, default=None)
    parser.add_argument("--samples-per-scan-per-body-part", type=int, default=100)
    parser.add_argument("--target-rgb-depth-pairs-per-body-part", type=int, default=None)
    parser.add_argument("--lesion-count-min", type=int, default=10)
    parser.add_argument("--lesion-count-max", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-segments", type=int, default=7)
    parser.add_argument("--angular-segments", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.body_part is None:
        args.body_part = BODY_PARTS
    if args.scan_id is None:
        args.scan_id = SCAN_IDS
    if args.lesion_count_min < 1 or args.lesion_count_max < args.lesion_count_min:
        raise ValueError("Invalid lesion count range")
    if args.lesion_count_min < 10 or args.lesion_count_max > 100:
        raise ValueError("This generator is intended for 10-100 lesions per image")
    if args.samples_per_scan_per_body_part < 1:
        raise ValueError("--samples-per-scan-per-body-part must be positive")
    return args


if __name__ == "__main__":
    build_dataset(build_parser())

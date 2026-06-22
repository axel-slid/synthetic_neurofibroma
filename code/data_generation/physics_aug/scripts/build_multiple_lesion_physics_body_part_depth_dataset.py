#!/usr/bin/env python3
"""Generate body-part-first multi-lesion physics RGB/depth pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import pyrender
from plotly.utils import PlotlyJSONEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
BODY_PART_SCRIPT_DIR = SCRIPT_DIR.parents[1] / "body_parts" / "scripts"
for path in (SCRIPT_DIR, BODY_PART_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_body_part_multi_lesion_depth_dataset import (  # noqa: E402
    BODY_PARTS,
    SCAN_IDS,
    WORLD_UP,
    LesionRecord,
    camera_for_body_part,
    camera_for_lesion_closeup,
    choose_face,
    depth_visual,
    normalized,
    render_pair,
    root_relative,
    save_depth_png,
    sample_multi_lesion_geometry,
    segmentation_body_part,
    select_body_part_region,
    surface_attachment_is_usable,
    tangent_basis,
    visible_candidates,
    write_lesion_ply,
)
from build_body_part_volume_depth_dataset import (  # noqa: E402
    HSR_SEGMENTATION_ROOT,
    ROOT,
    ScanSurface,
    spherical_cap_support_radius,
    spherical_cap_volume,
)
from build_multiple_lesion_physics_dataset import (  # noqa: E402
    LesionSpec,
    build_local_shape,
    mesh_faces_for_rings,
)

DATASET_ROOT = ROOT / "data" / "synthetic" / "multiple_lesion_physics"
METHOD = "physics"
MODEL_NAME = "continuous_gravity_multi_lesion_flop"

MANIFEST_FIELDS = [
    "sample_id",
    "body_part",
    "source_segmentation_body_part",
    "scan_id",
    "patient_volume_index",
    "seed",
    "image_path",
    "depth_npy_path",
    "depth_png_path",
    "depth_vis_path",
    "volume_mesh_path",
    "metadata_path",
    "camera_mode",
    "depth_type",
    "lesion_pattern_source",
    "width",
    "height",
    "valid_depth_pixels",
    "lesion_count",
    "lesion_count_min",
    "lesion_count_max",
    "body_region",
    "radius_m",
    "radius_min_m",
    "radius_mean_m",
    "radius_max_m",
    "lesion_height_m",
    "lesion_height_min_m",
    "lesion_height_mean_m",
    "lesion_height_max_m",
    "support_radius_m",
    "support_radius_min_m",
    "support_radius_mean_m",
    "support_radius_max_m",
    "projection_max_distance_m",
    "projection_mean_distance_m",
    "contact_label_fraction_min",
    "contact_label_fraction_mean",
    "spherical_cap_volume_ml",
    "total_spherical_cap_volume_ml",
    "visible_candidate_faces",
    "fov_deg",
    "roll_deg",
    "frame_margin",
    "frame_half_width_m",
    "frame_half_height_m",
    "camera_distance_m",
    "ambient",
    "directional_intensity",
    "light_yaw_offset",
    "light_pitch_offset",
    "eye_xyz",
    "target_xyz",
    "view_direction_xyz",
    "camera_to_world",
    "mesh_path",
    "pair_index",
    "split",
    "method",
    "shape_family",
    "texture_variant",
    "source_pair_manifest",
    "source_sample_id",
    "off_axis_deg",
    "target_lesion_index",
    "target_lesion_face_index",
    "target_lesion_radius_m",
    "target_lesion_height_m",
    "target_u_offset_m",
    "target_v_offset_m",
    "target_normal_offset_m",
    "physics_model",
    "simulation_frame_index",
    "simulation_frame_count",
    "growth_rate_min",
    "growth_rate_mean",
    "growth_rate_max",
    "growth_delay_min",
    "growth_delay_mean",
    "growth_delay_max",
    "gravity_scale_min",
    "gravity_scale_mean",
    "gravity_scale_max",
    "pedunculation_mean",
    "flop_mean",
]

SETTINGS_FIELDS = [
    "setting_id",
    "setting_index",
    "split",
    "body_part",
    "method",
    "shape_family",
    "texture_variant",
    "source_manifest",
    "source_sample_id",
    "scan_id",
    "patient_volume_index",
    "seed",
    "face_index",
    "lesion_count",
    "radius_m",
    "lesion_height_m",
    "support_radius_m",
    "spherical_cap_volume_ml",
    "target_xyz",
    "eye_xyz",
    "camera_to_world",
    "source_image_path",
    "source_depth_npy_path",
    "source_metadata_path",
    "sample_id",
    "image_path",
    "depth_npy_path",
    "depth_png_path",
    "depth_vis_path",
    "metadata_path",
    "volume_mesh_path",
    "camera_depth_manifest",
    "fov_deg",
    "roll_deg",
    "off_axis_deg",
    "camera_distance_m",
    "frame_half_width_m",
    "frame_half_height_m",
    "target_lesion_index",
    "target_lesion_radius_m",
    "target_lesion_height_m",
    "camera_mode",
    "physics_model",
    "simulation_frame_index",
    "growth_rate_mean",
    "gravity_scale_mean",
]


@dataclass
class PhysicsLesionRecord:
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
    final_height_m: float
    neck_radius_m: float
    bulb_radius_m: float
    stalk_fraction: float
    gravity_scale: float
    flop_distance_m: float
    growth_delay: float
    growth_duration: float
    growth_power: float
    growth_rate: float
    adjusted_growth_t: float
    pedunculation: float
    gravity_term: float
    flop: float


@dataclass
class PhysicsMultiLesionSample:
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
    physics_model: str
    simulation_frame_index: int
    simulation_frame_count: int
    camera: dict[str, Any]
    lesions: list[PhysicsLesionRecord]


def method_root(body_part: str) -> Path:
    return DATASET_ROOT / "body_parts" / body_part / METHOD


def method_data_path(method_root_path: Path, path: Path) -> str:
    return str(path.relative_to(method_root_path / "data"))


def json_vector(values: Any) -> str:
    return json.dumps(values)


def physics_rgb(skin_rgb: np.ndarray, heights: np.ndarray, radial_weight: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = skin_rgb.astype(np.float32)
    warm = np.array([205.0, 126.0, 98.0], dtype=np.float32)
    tint = np.asarray([1.05, 0.92, 0.90], dtype=np.float32) + rng.normal(0.0, 0.018, size=3)
    height_amount = np.clip(heights / max(float(np.max(heights)), 1e-6), 0.0, 1.0)
    center_amount = np.clip(1.0 - radial_weight, 0.0, 1.0)
    color = 0.80 * base * tint[None, :] + 0.20 * warm[None, :]
    color += (14.0 * height_amount * center_amount)[:, None]
    color += rng.normal(0.0, 2.0, size=color.shape)
    return np.clip(color, 0, 255).astype(np.uint8)


def sample_physics_spec(
    lesion_index: int,
    face_index: int,
    anchor: np.ndarray,
    base_rgb: np.ndarray,
    body_part: str,
    lesion_count: int,
    rng: np.random.Generator,
) -> tuple[LesionSpec, float, float]:
    radius, cap_height = sample_multi_lesion_geometry(body_part, lesion_count, rng)
    support_radius = spherical_cap_support_radius(radius, cap_height)
    final_height = float(np.clip(cap_height * rng.uniform(1.75, 3.35), 0.0022, radius * 1.28))
    support_radius = float(np.clip(support_radius * rng.uniform(0.86, 1.18), 0.0018, radius * 1.25))
    neck_radius = float(np.clip(support_radius * rng.uniform(0.23, 0.44), 0.0008, support_radius * 0.70))
    bulb_radius = float(np.clip(radius * rng.uniform(0.58, 1.10), neck_radius * 1.40, support_radius * 1.22))
    stalk_fraction = float(rng.uniform(0.34, 0.58))
    gravity_scale = float(rng.uniform(5.8, 12.8))
    flop_distance = float(rng.uniform(0.012, 0.046) + final_height * rng.uniform(0.18, 0.46))
    arch_height = float(rng.uniform(0.004, 0.014) + final_height * rng.uniform(0.10, 0.34))
    distal_center_height = float(rng.uniform(0.002, 0.010) + final_height * rng.uniform(0.08, 0.28))
    sag = float(rng.uniform(0.002, 0.010) + final_height * rng.uniform(0.03, 0.14))
    growth_delay = float(rng.uniform(0.0, 0.30))
    growth_duration = float(rng.uniform(0.52, 1.05))
    growth_power = float(rng.uniform(0.62, 1.55))
    growth_rate = float(1.0 / growth_duration)
    color = np.clip(0.74 * base_rgb.astype(np.float32) + 0.26 * np.array([205, 126, 98]), 0, 255).astype(np.uint8)
    return (
        LesionSpec(
            lesion_id=f"lesion_{lesion_index:03d}",
            target_x=float(anchor[0]),
            target_y=float(anchor[1]),
            target_z=float(anchor[2]),
            target_vertex_index=face_index,
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
            lateral=float(rng.uniform(-0.65, 0.65)),
            twist=float(rng.uniform(-0.85, 1.05)),
            lobe_amp=float(rng.uniform(0.004, 0.048)),
            pear_bias=float(rng.uniform(0.16, 0.56)),
            growth_delay=growth_delay,
            growth_duration=growth_duration,
            growth_power=growth_power,
            growth_rate=growth_rate,
            color_rgb=(int(color[0]), int(color[1]), int(color[2])),
        ),
        radius,
        cap_height,
    )


def physics_lesion_mesh(
    scan: ScanSurface,
    body_part: str,
    lesion_index: int,
    face_index: int,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    base_rgb: np.ndarray,
    lesion_count: int,
    frame_index: int,
    frame_count: int,
    rng: np.random.Generator,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PhysicsLesionRecord, dict[str, float]]:
    spec, radius, cap_height = sample_physics_spec(lesion_index, face_index, anchor, base_rgb, body_part, lesion_count, rng)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_direction_2d = np.array([float(gravity_world @ tangent_u), float(gravity_world @ tangent_v)], dtype=np.float32)
    local_xyz, faces, radial_weight, state = build_local_shape(
        spec,
        frame_index,
        frame_count,
        gravity_direction_2d,
        radial_segments,
        angular_segments,
    )
    local_points = local_xyz[:, [0, 2]].astype(np.float32)
    tangent_points = anchor + local_points[:, 0, None] * tangent_u + local_points[:, 1, None] * tangent_v
    skin_points, surface_normals, skin_rgb, triangle_ids, distances = scan.project_points_to_surface(
        tangent_points,
        normal_hint=normal,
    )
    heights = np.maximum(local_xyz[:, 1], 0.0).astype(np.float32)
    vertices = (skin_points + heights[:, None] * normal).astype(np.float32)
    lesion_rgb = physics_rgb(skin_rgb, heights, np.clip(radial_weight, 0.0, 1.0), rng)

    contact_count = 1 + angular_segments
    contact_indices = np.arange(min(contact_count, len(vertices)))
    surface_part = segmentation_body_part(scan, body_part)
    label_id = scan.label_id(surface_part)
    contact_label_fraction = float(np.mean(scan.face_labels[triangle_ids[contact_indices]] == label_id))
    normal_alignment = surface_normals[contact_indices] @ normalized(normal)
    diagnostics = {
        "projection_max_distance_m": float(np.max(distances)),
        "contact_projection_max_distance_m": float(np.max(distances[contact_indices])),
        "contact_normal_alignment_min": float(np.min(normal_alignment)),
        "contact_label_fraction": contact_label_fraction,
    }
    volume_m3 = spherical_cap_volume(radius, cap_height)
    record = PhysicsLesionRecord(
        lesion_index=lesion_index,
        face_index=face_index,
        radius_m=radius,
        height_m=float(np.max(heights)),
        support_radius_m=spec.support_radius,
        projection_max_distance_m=diagnostics["projection_max_distance_m"],
        contact_label_fraction=contact_label_fraction,
        spherical_cap_volume_m3=volume_m3,
        spherical_cap_volume_ml=volume_m3 * 1_000_000.0,
        anchor_xyz=[float(value) for value in anchor],
        normal_xyz=[float(value) for value in normal],
        tangent_u_xyz=[float(value) for value in tangent_u],
        tangent_v_xyz=[float(value) for value in tangent_v],
        base_rgb=[int(value) for value in base_rgb],
        lesion_rgb=[int(value) for value in lesion_rgb[0]],
        final_height_m=spec.final_height,
        neck_radius_m=spec.neck_radius,
        bulb_radius_m=spec.bulb_radius,
        stalk_fraction=spec.stalk_fraction,
        gravity_scale=spec.gravity_scale,
        flop_distance_m=spec.flop_distance,
        growth_delay=spec.growth_delay,
        growth_duration=spec.growth_duration,
        growth_power=spec.growth_power,
        growth_rate=spec.growth_rate,
        adjusted_growth_t=state["adjusted_growth_t"],
        pedunculation=state["pedunculation"],
        gravity_term=state["gravity_term"],
        flop=state["flop"],
    )
    return vertices, faces, lesion_rgb, record, diagnostics


def build_multi_lesion_physics_mesh(
    scan: ScanSurface,
    body_part: str,
    lesion_count: int,
    view_direction: np.ndarray,
    region_candidates: np.ndarray,
    frame_index: int,
    frame_count: int,
    rng: np.random.Generator,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PhysicsLesionRecord], int]:
    candidates = visible_candidates(scan, body_part, view_direction, region_candidates)
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    lesion_records: list[PhysicsLesionRecord] = []
    vertex_offset = 0

    for lesion_index in range(lesion_count):
        best: tuple[np.ndarray, np.ndarray, np.ndarray, PhysicsLesionRecord, dict[str, float]] | None = None
        best_score = -np.inf
        for _ in range(36):
            face_index = choose_face(scan, candidates, rng)
            selected = scan.face_triangles[face_index]
            weights = rng.dirichlet(np.ones(3))
            anchor = (weights @ selected).astype(np.float32)
            normal = normalized(scan.face_normals[face_index])
            tangent_u, tangent_v = tangent_basis(normal, WORLD_UP)
            base_rgb = scan.face_rgb[face_index].astype(np.uint8)
            placement = physics_lesion_mesh(
                scan,
                body_part,
                lesion_index,
                face_index,
                anchor,
                normal,
                tangent_u,
                tangent_v,
                base_rgb,
                lesion_count,
                frame_index,
                frame_count,
                rng,
                radial_segments,
                angular_segments,
            )
            diagnostics = placement[4]
            score = (
                diagnostics["contact_label_fraction"]
                + diagnostics["contact_normal_alignment_min"]
                - diagnostics["contact_projection_max_distance_m"] / max(placement[3].support_radius_m, 1e-6)
            )
            if score > best_score:
                best_score = score
                best = placement
            if surface_attachment_is_usable(
                {
                    "contact_label_fraction": diagnostics["contact_label_fraction"],
                    "contact_normal_alignment_min": diagnostics["contact_normal_alignment_min"],
                    "contact_projection_max_distance_m": diagnostics["contact_projection_max_distance_m"],
                },
                placement[3].support_radius_m,
            ):
                break
        if best is None:
            raise RuntimeError(f"Could not place physics lesion on {scan.scan_id} {body_part}")

        lesion_vertices, lesion_faces, lesion_rgb, lesion_record, _diagnostics = best
        vertices_parts.append(lesion_vertices)
        faces_parts.append(lesion_faces + vertex_offset)
        rgb_parts.append(lesion_rgb)
        vertex_offset += len(lesion_vertices)
        lesion_records.append(lesion_record)

    return (
        np.vstack(vertices_parts).astype(np.float32),
        np.vstack(faces_parts).astype(np.int32),
        np.vstack(rgb_parts).astype(np.uint8),
        lesion_records,
        len(candidates),
    )


def row_paths(root: Path, body_part: str, sample_id: str) -> dict[str, Path]:
    data_root = root / "body_parts" / body_part / METHOD / "data"
    return {
        "volume_mesh": data_root / "volumes" / f"{sample_id}_multi_lesion_volume.ply",
        "metadata": data_root / "metadata" / f"{sample_id}.json",
        "image": data_root / "images" / f"{sample_id}_rgb.png",
        "depth_npy": data_root / "depth" / f"{sample_id}_depth.npy",
        "depth_png": data_root / "depth" / f"{sample_id}_depth_mm.png",
        "depth_vis": data_root / "depth_vis" / f"{sample_id}_depth_vis.png",
    }


def build_sample(
    scan: ScanSurface,
    body_part: str,
    patient_volume_index: int,
    seed: int,
    renderer: pyrender.OffscreenRenderer,
    lesion_min: int,
    lesion_max: int,
    radial_segments: int,
    angular_segments: int,
    frame_count: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lesion_count = int(rng.integers(lesion_min, lesion_max + 1))
    frame_index = int(rng.integers(max(1, int(frame_count * 0.62)), frame_count))
    region_candidates, body_region = select_body_part_region(scan, body_part, rng)
    _part_camera, _part_settings, view_direction = camera_for_body_part(scan, body_part, region_candidates, rng)
    lesion_vertices, lesion_faces, lesion_rgb, lesions, visible_face_count = build_multi_lesion_physics_mesh(
        scan,
        body_part,
        lesion_count,
        view_direction,
        region_candidates,
        frame_index,
        frame_count,
        rng,
        radial_segments,
        angular_segments,
    )
    closeup_records = [
        LesionRecord(
            lesion_index=lesion.lesion_index,
            face_index=lesion.face_index,
            radius_m=lesion.radius_m,
            height_m=lesion.height_m,
            support_radius_m=lesion.support_radius_m,
            projection_max_distance_m=lesion.projection_max_distance_m,
            contact_label_fraction=lesion.contact_label_fraction,
            spherical_cap_volume_m3=lesion.spherical_cap_volume_m3,
            spherical_cap_volume_ml=lesion.spherical_cap_volume_ml,
            anchor_xyz=lesion.anchor_xyz,
            normal_xyz=lesion.normal_xyz,
            tangent_u_xyz=lesion.tangent_u_xyz,
            tangent_v_xyz=lesion.tangent_v_xyz,
            base_rgb=lesion.base_rgb,
            lesion_rgb=lesion.lesion_rgb,
        )
        for lesion in lesions
    ]
    camera, settings, target_lesion = camera_for_lesion_closeup(closeup_records, body_part, rng)
    rgb, depth = render_pair(renderer, scan, lesion_vertices, lesion_faces, lesion_rgb, camera, settings)

    sample_id = f"{body_part}_{scan.scan_id}_multi_v{patient_volume_index:03d}"
    paths = row_paths(DATASET_ROOT, body_part, sample_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_lesion_ply(paths["volume_mesh"], lesion_vertices, lesion_faces, lesion_rgb)
    imageio.imwrite(paths["image"], rgb)
    np.save(paths["depth_npy"], depth)
    save_depth_png(depth, paths["depth_png"])
    imageio.imwrite(paths["depth_vis"], depth_visual(depth))

    sample = PhysicsMultiLesionSample(
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
        lesion_pattern_source="10-100 continuous-gravity physics lesions with varied growth rates",
        physics_model=MODEL_NAME,
        simulation_frame_index=frame_index,
        simulation_frame_count=frame_count,
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
    growth_rates = np.asarray([lesion.growth_rate for lesion in lesions], dtype=np.float64)
    growth_delays = np.asarray([lesion.growth_delay for lesion in lesions], dtype=np.float64)
    gravity_scales = np.asarray([lesion.gravity_scale for lesion in lesions], dtype=np.float64)
    pedunculations = np.asarray([lesion.pedunculation for lesion in lesions], dtype=np.float64)
    flops = np.asarray([lesion.flop for lesion in lesions], dtype=np.float64)
    target_physics = lesions[int(target_lesion.lesion_index)]
    valid_depth = int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))
    method_root_path = method_root(body_part)
    row = {
        "sample_id": sample_id,
        "body_part": body_part,
        "source_segmentation_body_part": segmentation_body_part(scan, body_part),
        "scan_id": scan.scan_id,
        "patient_volume_index": patient_volume_index,
        "seed": seed,
        "image_path": method_data_path(method_root_path, paths["image"]),
        "depth_npy_path": method_data_path(method_root_path, paths["depth_npy"]),
        "depth_png_path": method_data_path(method_root_path, paths["depth_png"]),
        "depth_vis_path": method_data_path(method_root_path, paths["depth_vis"]),
        "volume_mesh_path": method_data_path(method_root_path, paths["volume_mesh"]),
        "metadata_path": method_data_path(method_root_path, paths["metadata"]),
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
        "eye_xyz": json_vector(camera["eye_xyz"]),
        "target_xyz": json_vector(camera["target_xyz"]),
        "view_direction_xyz": json_vector(camera["view_direction_xyz"]),
        "camera_to_world": json_vector(camera["camera_to_world"]),
        "mesh_path": root_relative(paths["volume_mesh"]),
        "pair_index": patient_volume_index,
        "split": "multiple_lesion_physics",
        "method": METHOD,
        "shape_family": "continuous_gravity_pedunculated",
        "texture_variant": "skin_interpolated_physics",
        "source_pair_manifest": "",
        "source_sample_id": sample_id,
        "physics_model": MODEL_NAME,
        "simulation_frame_index": frame_index,
        "simulation_frame_count": frame_count,
        "growth_rate_min": float(growth_rates.min()),
        "growth_rate_mean": float(growth_rates.mean()),
        "growth_rate_max": float(growth_rates.max()),
        "growth_delay_min": float(growth_delays.min()),
        "growth_delay_mean": float(growth_delays.mean()),
        "growth_delay_max": float(growth_delays.max()),
        "gravity_scale_min": float(gravity_scales.min()),
        "gravity_scale_mean": float(gravity_scales.mean()),
        "gravity_scale_max": float(gravity_scales.max()),
        "pedunculation_mean": float(pedunculations.mean()),
        "flop_mean": float(flops.mean()),
    }
    row["_target_physics"] = target_physics
    return row


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key, "") for key in MANIFEST_FIELDS}


def settings_row(row: dict[str, Any], setting_index: int) -> dict[str, Any]:
    target = row["_target_physics"]
    values = {
        "setting_id": f"multiple_lesion_physics_{row['body_part']}_{METHOD}_{setting_index:04d}",
        "setting_index": setting_index,
        "split": "multiple_lesion_physics",
        "body_part": row["body_part"],
        "method": METHOD,
        "shape_family": "continuous_gravity_pedunculated",
        "texture_variant": "skin_interpolated_physics",
        "source_manifest": "",
        "source_sample_id": row["sample_id"],
        "scan_id": row["scan_id"],
        "patient_volume_index": row["patient_volume_index"],
        "seed": row["seed"],
        "face_index": target.face_index,
        "lesion_count": row["lesion_count"],
        "radius_m": row["radius_m"],
        "lesion_height_m": row["lesion_height_m"],
        "support_radius_m": row["support_radius_m"],
        "spherical_cap_volume_ml": row["spherical_cap_volume_ml"],
        "target_xyz": row["target_xyz"],
        "eye_xyz": row["eye_xyz"],
        "camera_to_world": row["camera_to_world"],
        "source_image_path": "",
        "source_depth_npy_path": "",
        "source_metadata_path": "",
        "sample_id": row["sample_id"],
        "image_path": row["image_path"],
        "depth_npy_path": row["depth_npy_path"],
        "depth_png_path": row["depth_png_path"],
        "depth_vis_path": row["depth_vis_path"],
        "metadata_path": row["metadata_path"],
        "volume_mesh_path": row["volume_mesh_path"],
        "camera_depth_manifest": "camera_depth_manifest.csv",
        "fov_deg": row["fov_deg"],
        "roll_deg": row["roll_deg"],
        "off_axis_deg": row["off_axis_deg"],
        "camera_distance_m": row["camera_distance_m"],
        "frame_half_width_m": row["frame_half_width_m"],
        "frame_half_height_m": row["frame_half_height_m"],
        "target_lesion_index": row["target_lesion_index"],
        "target_lesion_radius_m": row["target_lesion_radius_m"],
        "target_lesion_height_m": row["target_lesion_height_m"],
        "camera_mode": row["camera_mode"],
        "physics_model": MODEL_NAME,
        "simulation_frame_index": row["simulation_frame_index"],
        "growth_rate_mean": row["growth_rate_mean"],
        "gravity_scale_mean": row["gravity_scale_mean"],
    }
    return {key: values.get(key, "") for key in SETTINGS_FIELDS}


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def resolve_method_data_path(method_root_path: Path, rel_path: str) -> Path:
    return method_root_path / "data" / rel_path


def image_to_plotly_source(path: Path) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def rgb_strings(rgb: np.ndarray) -> list[str]:
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return [f"rgb({int(red)},{int(green)},{int(blue)})" for red, green, blue in rgb]


def rgb_string(color: list[int] | tuple[int, int, int] | np.ndarray) -> str:
    red, green, blue = [int(value) for value in color]
    return f"rgb({red},{green},{blue})"


def simplify_textured_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    target_faces: int = 7200,
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


def metadata_row_for_progression(method_root_path: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_row = max(rows, key=lambda row: int(row.get("lesion_count", 0)))
    metadata_path = resolve_method_data_path(method_root_path, best_row["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return best_row, metadata


def spec_from_metadata(lesion: dict[str, Any]) -> LesionSpec:
    final_height = float(lesion["final_height_m"])
    support_radius = float(lesion["support_radius_m"])
    lesion_index = int(lesion["lesion_index"])
    # Older saved metadata did not keep these secondary shape parameters, so
    # the viewer reconstructs a deterministic progression from the persisted
    # growth, neck, bulb, gravity, flop, anchor, normal, and tangent fields.
    lateral = 0.42 * math.sin(1.61803398875 * (lesion_index + 1))
    twist = 0.58 * math.cos(0.754877666 * (lesion_index + 3))
    lobe_amp = 0.018 + 0.012 * (0.5 + 0.5 * math.sin(0.37 * (lesion_index + 5)))
    pear_bias = 0.28 + 0.14 * (0.5 + 0.5 * math.cos(0.29 * (lesion_index + 7)))
    anchor = [float(value) for value in lesion["anchor_xyz"]]
    return LesionSpec(
        lesion_id=f"lesion_{lesion_index:03d}",
        target_x=anchor[0],
        target_y=anchor[1],
        target_z=anchor[2],
        target_vertex_index=int(lesion["face_index"]),
        final_height=final_height,
        support_radius=support_radius,
        neck_radius=float(lesion["neck_radius_m"]),
        bulb_radius=float(lesion["bulb_radius_m"]),
        stalk_fraction=float(lesion["stalk_fraction"]),
        gravity_scale=float(lesion["gravity_scale"]),
        flop_distance=float(lesion["flop_distance_m"]),
        arch_height=max(0.004, 0.24 * final_height),
        distal_center_height=max(0.002, 0.12 * final_height),
        sag=max(0.002, 0.08 * final_height),
        lateral=float(lateral),
        twist=float(twist),
        lobe_amp=float(lobe_amp),
        pear_bias=float(pear_bias),
        growth_delay=float(lesion["growth_delay"]),
        growth_duration=float(lesion["growth_duration"]),
        growth_power=float(lesion["growth_power"]),
        growth_rate=float(lesion["growth_rate"]),
        color_rgb=tuple(int(value) for value in lesion["lesion_rgb"]),
    )


def lesion_vertices_for_frame(
    lesion: dict[str, Any],
    spec: LesionSpec,
    frame_index: int,
    frame_count: int,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    anchor = np.asarray(lesion["anchor_xyz"], dtype=np.float32)
    normal = normalized(np.asarray(lesion["normal_xyz"], dtype=np.float32))
    tangent_u = normalized(np.asarray(lesion["tangent_u_xyz"], dtype=np.float32))
    tangent_v = normalized(np.asarray(lesion["tangent_v_xyz"], dtype=np.float32))
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_direction_2d = np.array([float(gravity_world @ tangent_u), float(gravity_world @ tangent_v)], dtype=np.float32)
    local_xyz, faces, _radial_weight, state = build_local_shape(
        spec,
        frame_index,
        frame_count,
        gravity_direction_2d,
        radial_segments,
        angular_segments,
    )
    local_points = local_xyz[:, [0, 2]].astype(np.float32)
    heights = np.maximum(local_xyz[:, 1], 0.0).astype(np.float32)
    vertices = anchor + local_points[:, 0, None] * tangent_u + local_points[:, 1, None] * tangent_v
    vertices = vertices + heights[:, None] * normal
    return vertices.astype(np.float32), faces.astype(np.int32), state


def camera_for_progression(body_part: str) -> dict[str, dict[str, float]]:
    if body_part == "back":
        return {"eye": {"x": 0.0, "y": -2.65, "z": 0.55}, "center": {"x": 0.0, "y": 0.0, "z": 0.0}, "up": {"x": 0.0, "y": 0.0, "z": 1.0}}
    if body_part == "face":
        return {"eye": {"x": 0.0, "y": 2.35, "z": 0.82}, "center": {"x": 0.0, "y": 0.0, "z": 0.10}, "up": {"x": 0.0, "y": 0.0, "z": 1.0}}
    if body_part in {"hands", "arms"}:
        return {"eye": {"x": 1.45, "y": 2.10, "z": 0.42}, "center": {"x": 0.0, "y": 0.0, "z": 0.02}, "up": {"x": 0.0, "y": 0.0, "z": 1.0}}
    if body_part in {"feet", "legs"}:
        return {"eye": {"x": 0.70, "y": 2.30, "z": 0.24}, "center": {"x": 0.0, "y": 0.0, "z": -0.16}, "up": {"x": 0.0, "y": 0.0, "z": 1.0}}
    return {"eye": {"x": 0.0, "y": 2.65, "z": 0.55}, "center": {"x": 0.0, "y": 0.0, "z": 0.0}, "up": {"x": 0.0, "y": 0.0, "z": 1.0}}


def build_progression_figure(
    method_root_path: Path,
    body_part: str,
    rows: list[dict[str, Any]],
    radial_segments: int = 7,
    angular_segments: int = 24,
) -> tuple[go.Figure, dict[str, Any]]:
    row, metadata = metadata_row_for_progression(method_root_path, rows)
    frame_count = int(metadata.get("simulation_frame_count", 100))
    lesions = list(metadata["lesions"])
    scan = ScanSurface(str(metadata["scan_id"]))
    body_xyz, body_faces, body_rgb = simplify_textured_mesh(scan.vertices, scan.faces, scan.vertex_rgb)

    specs = [spec_from_metadata(lesion) for lesion in lesions]
    frame_zero = [
        lesion_vertices_for_frame(lesion, spec, 0, frame_count, radial_segments, angular_segments)
        for lesion, spec in zip(lesions, specs)
    ]

    data: list[go.BaseTraceType] = [
        go.Mesh3d(
            x=np.round(body_xyz[:, 0], 5),
            y=np.round(body_xyz[:, 1], 5),
            z=np.round(body_xyz[:, 2], 5),
            i=body_faces[:, 0],
            j=body_faces[:, 1],
            k=body_faces[:, 2],
            vertexcolor=rgb_strings(body_rgb),
            opacity=0.86,
            flatshading=False,
            lighting=dict(ambient=0.90, diffuse=0.60, specular=0.018, roughness=0.94),
            hoverinfo="skip",
            name="textured body",
            showlegend=False,
        )
    ]
    lesion_trace_indices = []
    for lesion, spec, (vertices, faces, _state) in zip(lesions, specs, frame_zero):
        lesion_trace_indices.append(len(data))
        data.append(
            go.Mesh3d(
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
                    f"{metadata['sample_id']} {spec.lesion_id}<br>"
                    f"growth rate {spec.growth_rate:.2f}<br>"
                    f"delay {spec.growth_delay:.2f}<br>"
                    f"gravity {spec.gravity_scale:.1f}<extra></extra>"
                ),
                name=spec.lesion_id,
                showlegend=False,
            )
        )

    frames = []
    for frame_index in range(frame_count):
        frame_data = []
        growth_values = []
        gravity_values = []
        for lesion, spec in zip(lesions, specs):
            vertices, _faces, state = lesion_vertices_for_frame(
                lesion,
                spec,
                frame_index,
                frame_count,
                radial_segments,
                angular_segments,
            )
            frame_data.append(
                go.Mesh3d(
                    x=np.round(vertices[:, 0], 4),
                    y=np.round(vertices[:, 1], 4),
                    z=np.round(vertices[:, 2], 4),
                )
            )
            growth_values.append(float(state["adjusted_growth_t"]))
            gravity_values.append(float(state["gravity_term"]))
        frames.append(
            go.Frame(
                name=f"{frame_index + 1:03d}",
                data=frame_data,
                traces=lesion_trace_indices,
                layout=go.Layout(
                    title_text=(
                        f"{body_part} physics: {len(lesions)} lesions - frame {frame_index + 1:03d}/{frame_count} - "
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
        title=dict(text=f"{body_part} physics: {len(lesions)} lesions - frame 001/{frame_count}", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(visible=False, range=[float(xyz_min[0]), float(xyz_max[0])]),
            yaxis=dict(visible=False, range=[float(xyz_min[1]), float(xyz_max[1])]),
            zaxis=dict(visible=False, range=[float(xyz_min[2]), float(xyz_max[2])]),
            bgcolor="rgb(244,246,249)",
            aspectmode="data",
            camera=camera_for_progression(body_part),
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
    record = {
        "sample_id": metadata["sample_id"],
        "scan_id": metadata["scan_id"],
        "lesion_count": len(lesions),
        "frame_count": frame_count,
        "metadata": root_relative(resolve_method_data_path(method_root_path, row["metadata_path"])),
    }
    return fig, record


def compact_payload(value: object) -> object:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [compact_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_payload(item) for key, item in value.items()}
    return value


def write_code_free_plotly_notebook(notebook_path: Path, figure: go.Figure, body_part: str) -> None:
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    payload = compact_payload(payload)
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
                            "text/plain": f"<Plotly Figure: {body_part} physics 3D progression>",
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


def render_progression_gif(figure: go.Figure, gif_path: Path, frame_count: int, gif_frames: int = 10, fps: int = 5) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = np.unique(np.linspace(0, frame_count - 1, gif_frames, dtype=np.int32))
    working = go.Figure(data=figure.data, layout=figure.layout)
    images = []
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"{gif_path.stem}_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for output_index, frame_index in enumerate(sample_indices):
            frame = figure.frames[int(frame_index)]
            if frame.data:
                for trace_index, trace_update in zip(frame.traces, frame.data):
                    working.data[int(trace_index)].update(trace_update)
            working.update_layout(title_text=frame.layout.title.text if frame.layout and frame.layout.title else None)
            png_path = tmp_dir / f"frame_{output_index:03d}.png"
            working.write_image(png_path, width=900, height=650, scale=1)
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=1 / fps, loop=0)


def write_preview_visualizations(method_root_path: Path, body_part: str, rows: list[dict[str, Any]], sample_count: int) -> tuple[Path, Path, Path]:
    gif_dir = method_root_path / "visualization" / "gifs"
    plotly_dir = method_root_path / "visualization" / "plotly"
    gif_dir.mkdir(parents=True, exist_ok=True)
    plotly_dir.mkdir(parents=True, exist_ok=True)
    preview_rows = rows[: min(sample_count, len(rows))]
    tile_size = 160
    label_height = 22
    frames = []
    for row in preview_rows[:18]:
        rgb = imageio.imread(resolve_method_data_path(method_root_path, row["image_path"]))
        depth = imageio.imread(resolve_method_data_path(method_root_path, row["depth_vis_path"]))
        from PIL import Image, ImageDraw

        rgb_img = Image.fromarray(rgb).convert("RGB").resize((tile_size, tile_size))
        depth_img = Image.fromarray(depth).convert("L").resize((tile_size, tile_size)).convert("RGB")
        frame = Image.new("RGB", (tile_size * 2, tile_size + label_height), "white")
        frame.paste(rgb_img, (0, label_height))
        frame.paste(depth_img, (tile_size, label_height))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 6), "RGB", fill=(32, 38, 48))
        draw.text((tile_size + 8, 6), "Depth", fill=(32, 38, 48))
        frames.append(np.asarray(frame))
    gif_path = gif_dir / f"{METHOD}_rgb_depth_preview.gif"
    if frames:
        imageio.mimsave(gif_path, frames, duration=0.55, loop=0)

    fig, progression_record = build_progression_figure(method_root_path, body_part, preview_rows)
    notebook_path = plotly_dir / f"{METHOD}_closed_body_lesion_viewer.ipynb"
    write_code_free_plotly_notebook(notebook_path, fig, body_part)
    progression_gif_path = gif_dir / f"{METHOD}_3d_progression.gif"
    render_progression_gif(fig, progression_gif_path, int(progression_record["frame_count"]))

    manifest_path = plotly_dir / f"{METHOD}_closed_body_lesion_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "body_part": body_part,
                "method": METHOD,
                "visualization_type": "3d_physics_progression",
                "preview_sample_count": len(preview_rows),
                "sample_id": progression_record["sample_id"],
                "scan_id": progression_record["scan_id"],
                "lesion_count": progression_record["lesion_count"],
                "frame_count": progression_record["frame_count"],
                "metadata": progression_record["metadata"],
                "notebook": root_relative(notebook_path),
                "gif": root_relative(progression_gif_path),
                "rgb_depth_preview_gif": root_relative(gif_path),
                "source_manifest": root_relative(method_root_path / "data" / "camera_depth_manifest.csv"),
                "segmentation_source": root_relative(HSR_SEGMENTATION_ROOT),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return progression_gif_path, notebook_path, manifest_path


def write_method_outputs(body_part: str, raw_rows: list[dict[str, Any]], preview_count: int) -> dict[str, Any]:
    method_root_path = method_root(body_part)
    data_root = method_root_path / "data"
    rows = [clean_row(row) for row in raw_rows]
    settings = [settings_row(row, idx) for idx, row in enumerate(raw_rows)]
    write_rows_csv(data_root / "camera_depth_manifest.csv", rows, MANIFEST_FIELDS)
    write_jsonl(data_root / "camera_depth_manifest.jsonl", rows)
    write_rows_csv(data_root / "settings.csv", settings, SETTINGS_FIELDS)
    gif_path, notebook_path, plotly_manifest_path = write_preview_visualizations(method_root_path, body_part, rows, preview_count)
    summary = {
        "split": "multiple_lesion_physics",
        "body_part": body_part,
        "method": METHOD,
        "physics_model": MODEL_NAME,
        "setting_count": len(settings),
        "rgb_depth_pair_count": len(rows),
        "image_count": len(list((data_root / "images").glob("*.png"))),
        "depth_npy_count": len(list((data_root / "depth").glob("*_depth.npy"))),
        "depth_png_count": len(list((data_root / "depth").glob("*_depth_mm.png"))),
        "depth_vis_count": len(list((data_root / "depth_vis").glob("*.png"))),
        "volume_mesh_count": len(list((data_root / "volumes").glob("*.ply"))),
        "settings": root_relative(data_root / "settings.csv"),
        "camera_depth_manifest": root_relative(data_root / "camera_depth_manifest.csv"),
        "camera_depth_manifest_jsonl": root_relative(data_root / "camera_depth_manifest.jsonl"),
        "visualization_plotly": root_relative(notebook_path),
        "visualization_plotly_manifest": root_relative(plotly_manifest_path),
        "visualization_gif": root_relative(gif_path),
        "visualization_type": "rgb_depth_preview_with_physics_volume_files",
        "pair_storage": "local_method_folder",
        "camera_mode": "lesion_closeup_random",
        "framing": "random close-up camera centered near a sampled visible physics lesion",
    }
    (method_root_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def resolve_sample_counts(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    groups = [(body_part, scan_id) for body_part in args.body_part for scan_id in args.scan_id]
    if args.target_rgb_depth_pairs_per_body_part is not None:
        if args.target_rgb_depth_pairs_per_body_part < len(args.scan_id):
            raise ValueError("--target-rgb-depth-pairs-per-body-part must be at least the number of scans")
        counts = {}
        base_count, remainder = divmod(args.target_rgb_depth_pairs_per_body_part, len(args.scan_id))
        for body_part in args.body_part:
            for scan_idx, scan_id in enumerate(args.scan_id):
                counts[(body_part, scan_id)] = base_count + (1 if scan_idx < remainder else 0)
        return counts
    return {group: args.samples_per_scan_per_body_part for group in groups}


def build_dataset(args: argparse.Namespace) -> None:
    body_root = DATASET_ROOT / "body_parts"
    if args.overwrite:
        for body_part in args.body_part:
            target = body_root / body_part / METHOD
            if target.exists():
                shutil.rmtree(target)
    body_root.mkdir(parents=True, exist_ok=True)
    scans = {scan_id: ScanSurface(scan_id) for scan_id in args.scan_id}
    sample_counts = resolve_sample_counts(args)
    total_expected = sum(sample_counts[(body_part, scan_id)] for body_part in args.body_part for scan_id in args.scan_id)
    completed = 0
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    rows_by_part: dict[str, list[dict[str, Any]]] = {body_part: [] for body_part in args.body_part}
    try:
        for body_part_index, body_part in enumerate(args.body_part):
            for scan_index, scan_id in enumerate(args.scan_id):
                scan = scans[scan_id]
                volume_count = sample_counts[(body_part, scan_id)]
                for patient_volume_index in range(volume_count):
                    seed = args.seed + body_part_index * 1_000_000 + scan_index * 100_000 + patient_volume_index
                    row = build_sample(
                        scan=scan,
                        body_part=body_part,
                        patient_volume_index=patient_volume_index,
                        seed=seed,
                        renderer=renderer,
                        lesion_min=args.lesion_count_min,
                        lesion_max=args.lesion_count_max,
                        radial_segments=args.radial_segments,
                        angular_segments=args.angular_segments,
                        frame_count=args.frame_count,
                    )
                    rows_by_part[body_part].append(row)
                    completed += 1
                    if completed == 1 or completed == total_expected or completed % args.progress_interval == 0:
                        print(
                            f"[{completed:05d}/{total_expected:05d}] [{body_part}] {scan_id} "
                            f"{patient_volume_index + 1:04d}/{volume_count:04d} "
                            f"lesions={row['lesion_count']:03d} frame={row['simulation_frame_index']:03d} "
                            f"-> {row['sample_id']}",
                            flush=True,
                        )
    finally:
        renderer.delete()

    summaries = [write_method_outputs(body_part, rows_by_part[body_part], args.preview_count) for body_part in args.body_part]
    all_rows = [clean_row(row) for body_part in args.body_part for row in rows_by_part[body_part]]
    data_root = DATASET_ROOT / "data"
    write_rows_csv(data_root / "camera_depth_manifest.csv", all_rows, MANIFEST_FIELDS)
    write_jsonl(data_root / "camera_depth_manifest.jsonl", all_rows)
    root_summary = {
        "dataset": "multiple_lesion_physics",
        "schema": "body_part_first_synthetic_assets_v2",
        "physics_model": MODEL_NAME,
        "body_parts": args.body_part,
        "scan_ids": args.scan_id,
        "method": METHOD,
        "expected_rgb_depth_pairs_per_body_part": args.target_rgb_depth_pairs_per_body_part,
        "total_rgb_depth_pairs": len(all_rows),
        "camera_depth_manifest_csv": root_relative(data_root / "camera_depth_manifest.csv"),
        "camera_depth_manifest_jsonl": root_relative(data_root / "camera_depth_manifest.jsonl"),
        "summaries": summaries,
    }
    (DATASET_ROOT / "summary.json").write_text(json.dumps(root_summary, indent=2) + "\n", encoding="utf-8")
    (data_root / "summary.json").write_text(json.dumps(root_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(root_summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-part", action="append", choices=BODY_PARTS, default=None)
    parser.add_argument("--scan-id", action="append", choices=SCAN_IDS, default=None)
    parser.add_argument("--samples-per-scan-per-body-part", type=int, default=500)
    parser.add_argument("--target-rgb-depth-pairs-per-body-part", type=int, default=1000)
    parser.add_argument("--lesion-count-min", type=int, default=10)
    parser.add_argument("--lesion-count-max", type=int, default=100)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-segments", type=int, default=7)
    parser.add_argument("--angular-segments", type=int, default=24)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=30260621)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.body_part is None:
        args.body_part = BODY_PARTS
    if args.scan_id is None:
        args.scan_id = SCAN_IDS
    if args.lesion_count_min < 10 or args.lesion_count_max > 100 or args.lesion_count_max < args.lesion_count_min:
        raise ValueError("Lesion count range must stay within 10-100")
    return args


def main() -> None:
    build_dataset(build_parser())


if __name__ == "__main__":
    main()

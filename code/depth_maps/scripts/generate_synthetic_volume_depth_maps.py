from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyrender
import trimesh
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "depth_maps" / "synthetic"
SYNTHETIC_BODY_PARTS_ROOT = ROOT / "data" / "synthetic" / "single_lesion" / "body_parts"
SYNTHETIC_VISUALIZATION_ROOT = ROOT / "data" / "synthetic" / "single_lesion" / "visualization"
_CENTER_CACHE: dict[str, np.ndarray] = {}


@dataclass
class Volume:
    volume_id: str
    source_folder: str
    mesh_path: Path
    metadata_path: Path | None = None


def discover_source_meshes(source_folder: str) -> list[Volume]:
    source_data_root = SYNTHETIC_BODY_PARTS_ROOT / source_folder
    source_visualization_root = SYNTHETIC_VISUALIZATION_ROOT / source_folder
    mesh_root = source_visualization_root if source_visualization_root.exists() else source_data_root
    paths = sorted(mesh_root.rglob("*.ply"))
    if not paths:
        paths = sorted(source_data_root.rglob("*.obj"))

    # The Gaussian diffusion folder may exist before its diffusion textures are generated.
    # Depth is geometry-only, so use the matching Gaussian geometry as its source fallback.
    if not paths and source_folder == "gaussian_generations_textured_diffusion":
        fallback_visualization = SYNTHETIC_VISUALIZATION_ROOT / "gaussian_generations"
        fallback_data = SYNTHETIC_BODY_PARTS_ROOT / "gaussian_generations"
        paths = sorted(fallback_visualization.rglob("*.ply"))
        source_data_root = fallback_data

    if not paths:
        raise FileNotFoundError(f"No renderable OBJ/PLY meshes found for {source_folder}: {source_data_root}")

    return [
        Volume(
            volume_id=f"{source_folder}_mesh_{idx:02d}",
            source_folder=source_folder,
            mesh_path=path,
            metadata_path=metadata_path_for_mesh(source_data_root, path),
        )
        for idx, path in enumerate(paths)
    ]


def metadata_path_for_mesh(source_root: Path, mesh_path: Path) -> Path | None:
    stem = mesh_path.stem
    for suffix in [
        "_closed_textured_visualization",
        "_closed_visualization",
        "_textured_visualization",
        "_visualization",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates = [
        source_root / "metadata" / f"{stem}.json",
        source_root / "data" / "metadata" / f"{stem}.json",
        source_root / f"{stem}.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_volume_metadata(volume: Volume) -> dict[str, Any]:
    if volume.metadata_path is None:
        raise FileNotFoundError(f"No metadata file found for close-up rendering: {volume.mesh_path}")
    return json.loads(volume.metadata_path.read_text(encoding="utf-8"))


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_face_vertex_index(token: str) -> int:
    return int(token.split("/")[0]) - 1


def sampled_generation_center(scan_id: str, target_faces: int = 45_000) -> np.ndarray:
    """Return the original HSR centering offset used by the synthetic mesh builders."""
    if scan_id in _CENTER_CACHE:
        return _CENTER_CACHE[scan_id]

    obj_path = ROOT / "data" / "hsr" / "scans" / scan_id / "scan" / f"{scan_id}.obj"
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
    remap = set()
    sampled_vertices = []
    seen_faces = 0
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            if seen_faces in keep_face_numbers:
                for token in line.split()[1:4]:
                    vertex_idx = parse_face_vertex_index(token)
                    if vertex_idx not in remap:
                        remap.add(vertex_idx)
                        sampled_vertices.append(vertices_arr[vertex_idx])
            seen_faces += 1

    center = np.asarray(sampled_vertices, dtype=np.float32).mean(axis=0).astype(float)
    _CENTER_CACHE[scan_id] = center
    return center


def metadata_in_mesh_coordinates(metadata: dict[str, Any]) -> dict[str, Any]:
    converted = dict(metadata)
    center_offset = sampled_generation_center(str(metadata["scan_id"]))
    converted["anchor"] = (np.asarray(metadata["anchor"], dtype=float) + center_offset).tolist()
    converted["mesh_coordinate_offset"] = center_offset.tolist()
    return converted


def discover_synthetic_sources(args: argparse.Namespace) -> dict[str, list[Volume]]:
    source_root = SYNTHETIC_BODY_PARTS_ROOT
    source_names = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    if args.source_folder:
        requested = set(args.source_folder)
        missing = sorted(requested - set(source_names))
        if missing:
            raise FileNotFoundError(f"Requested source folders do not exist below {source_root}: {missing}")
        source_names = [name for name in source_names if name in requested]
    if args.source_prefix:
        source_names = [name for name in source_names if any(name.startswith(prefix) for prefix in args.source_prefix)]
    if not source_names:
        raise FileNotFoundError(f"No synthetic source folders matched below {source_root}")
    return {name: discover_source_meshes(name) for name in source_names}


def resolve_output_root(output_root: str | None) -> Path:
    if output_root is None:
        return DEFAULT_OUTPUT_ROOT
    path = Path(output_root)
    return path if path.is_absolute() else ROOT / path


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_root = resolve_output_root(args.output_root)
    if args.dataset_layout:
        data_root = output_root / "data"
        visualization_root = output_root / "visualizations"
        path_root = output_root
    else:
        data_root = output_root
        visualization_root = output_root
        path_root = output_root
    return {
        "output_root": output_root,
        "data_root": data_root,
        "visualization_root": visualization_root,
        "path_root": path_root,
    }


def look_at_camera_to_world(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=float)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def per_volume_view_counts(total_samples: int, volume_count: int) -> list[int]:
    base = total_samples // volume_count
    counts = [base] * volume_count
    for idx in range(total_samples % volume_count):
        counts[idx] += 1
    return counts


def per_source_sample_counts(args: argparse.Namespace, source_names: list[str]) -> dict[str, int]:
    if args.total_samples is None:
        return {source_name: args.samples_per_folder for source_name in source_names}
    if args.total_samples <= 0:
        raise ValueError("--total_samples must be positive")
    return dict(zip(source_names, per_volume_view_counts(args.total_samples, len(source_names)), strict=True))


def view_settings(volume_index: int, view_index: int, views_for_volume: int, rng: np.random.Generator) -> dict[str, float]:
    base_angle = 360.0 * view_index / views_for_volume
    return {
        "angle_deg": float((base_angle + rng.uniform(-0.45, 0.45) * 360.0 / views_for_volume) % 360.0),
        "fov_deg": float(rng.uniform(22.0, 58.0)),
        "distance_scale": float(rng.uniform(1.05, 1.85)),
        "elevation_fraction": float(rng.uniform(-0.22, 0.24)),
        "target_up_fraction": float(rng.uniform(-0.12, 0.14)),
        "target_h0_fraction": float(rng.uniform(-0.08, 0.08)),
        "target_h1_fraction": float(rng.uniform(-0.08, 0.08)),
        "ambient": float(rng.uniform(0.24, 0.82)),
        "directional_intensity": float(rng.uniform(0.55, 2.85)),
        "light_yaw_offset": float(math.radians(rng.uniform(-90.0, 90.0))),
        "light_pitch_offset": float(math.radians(rng.uniform(-40.0, 40.0))),
        "volume_index": float(volume_index),
        "view_index": float(view_index),
    }


def clinical_closeup_view_settings(
    volume_index: int,
    view_index: int,
    views_for_volume: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    base_angle = 360.0 * view_index / views_for_volume
    return {
        "angle_deg": float((base_angle + rng.uniform(-0.60, 0.60) * 360.0 / views_for_volume) % 360.0),
        "fov_deg": float(rng.uniform(30.0, 62.0)),
        "frame_scale": float(rng.uniform(1.45, 3.25)),
        "off_axis_deg": float(rng.uniform(0.0, 34.0)),
        "roll_deg": float(rng.uniform(-28.0, 28.0)),
        "target_u_fraction": float(rng.uniform(-0.24, 0.24)),
        "target_v_fraction": float(rng.uniform(-0.24, 0.24)),
        "target_normal_fraction": float(rng.uniform(0.08, 0.82)),
        "ambient": float(rng.uniform(0.28, 0.82)),
        "directional_intensity": float(rng.uniform(0.65, 2.65)),
        "light_yaw_offset": float(math.radians(rng.uniform(-70.0, 70.0))),
        "light_pitch_offset": float(math.radians(rng.uniform(-45.0, 45.0))),
        "volume_index": float(volume_index),
        "view_index": float(view_index),
    }


def camera_for_view(vertices: np.ndarray, settings: dict[str, float]) -> dict[str, Any]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extents = vmax - vmin
    up_axis = int(np.argmax(extents))
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    up = np.zeros(3, dtype=float)
    up[up_axis] = 1.0

    theta = math.radians(settings["angle_deg"])
    distance = float(np.linalg.norm(extents)) * settings["distance_scale"]
    eye = center.copy()
    eye[horizontal_axes[0]] += distance * math.cos(theta)
    eye[horizontal_axes[1]] += distance * math.sin(theta)
    eye[up_axis] += settings["elevation_fraction"] * float(extents[up_axis])

    target = center.copy()
    target[up_axis] += settings["target_up_fraction"] * float(extents[up_axis])
    target[horizontal_axes[0]] += settings["target_h0_fraction"] * float(extents[horizontal_axes[0]])
    target[horizontal_axes[1]] += settings["target_h1_fraction"] * float(extents[horizontal_axes[1]])

    return {
        "angle_deg": settings["angle_deg"],
        "eye_xyz": [float(v) for v in eye],
        "target_xyz": [float(v) for v in target],
        "up_axis": up_axis,
        "horizontal_axes": horizontal_axes,
        "camera_to_world": look_at_camera_to_world(eye, target, up).tolist(),
        "bounds_min": [float(v) for v in vmin],
        "bounds_max": [float(v) for v in vmax],
        "extents": [float(v) for v in extents],
        "distance_scale": settings["distance_scale"],
        "elevation_fraction": settings["elevation_fraction"],
        "target_up_fraction": settings["target_up_fraction"],
        "target_h0_fraction": settings["target_h0_fraction"],
        "target_h1_fraction": settings["target_h1_fraction"],
    }


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def lesion_scale_from_metadata(metadata: dict[str, Any]) -> tuple[float, float]:
    lesion_radius = float(metadata.get("support_radius") or metadata.get("radius") or 0.025)
    lesion_height = float(metadata.get("height") or max(0.004, lesion_radius * 0.45))
    return lesion_radius, lesion_height


def camera_for_clinical_closeup(metadata: dict[str, Any], settings: dict[str, float]) -> dict[str, Any]:
    anchor = np.asarray(metadata["anchor"], dtype=float)
    normal = normalized(np.asarray(metadata["normal"], dtype=float))
    tangent_u = normalized(np.asarray(metadata["tangent_u"], dtype=float))
    tangent_v = normalized(np.asarray(metadata["tangent_v"], dtype=float))
    lesion_radius, lesion_height = lesion_scale_from_metadata(metadata)

    theta = math.radians(settings["angle_deg"])
    tangent_direction = normalized(math.cos(theta) * tangent_u + math.sin(theta) * tangent_v)
    off_axis = math.radians(settings["off_axis_deg"])
    view_direction = normalized(math.cos(off_axis) * normal + math.sin(off_axis) * tangent_direction)

    frame_half_height = float(np.clip(lesion_radius * settings["frame_scale"], 0.026, 0.115))
    distance = frame_half_height / math.tan(math.radians(settings["fov_deg"]) / 2.0)
    distance = max(distance, lesion_height + 0.028)

    target = (
        anchor
        + settings["target_normal_fraction"] * lesion_height * normal
        + settings["target_u_fraction"] * frame_half_height * tangent_u
        + settings["target_v_fraction"] * frame_half_height * tangent_v
    )
    eye = target + distance * view_direction

    roll = math.radians(settings["roll_deg"])
    up = math.cos(roll) * tangent_v + math.sin(roll) * tangent_u
    up = up - np.dot(up, view_direction) * view_direction
    if np.linalg.norm(up) <= 1e-8:
        up = tangent_u - np.dot(tangent_u, view_direction) * view_direction
    up = normalized(up)

    return {
        "angle_deg": settings["angle_deg"],
        "eye_xyz": [float(v) for v in eye],
        "target_xyz": [float(v) for v in target],
        "camera_to_world": look_at_camera_to_world(eye, target, up).tolist(),
        "mesh_coordinate_offset_xyz": [float(v) for v in metadata.get("mesh_coordinate_offset", [0.0, 0.0, 0.0])],
        "lesion_anchor_xyz": [float(v) for v in anchor],
        "lesion_normal_xyz": [float(v) for v in normal],
        "lesion_tangent_u_xyz": [float(v) for v in tangent_u],
        "lesion_tangent_v_xyz": [float(v) for v in tangent_v],
        "lesion_radius_m": lesion_radius,
        "lesion_height_m": lesion_height,
        "clinical_frame_half_height_m": frame_half_height,
        "clinical_camera_distance_m": float(distance),
        "clinical_off_axis_deg": settings["off_axis_deg"],
        "clinical_roll_deg": settings["roll_deg"],
        "target_u_fraction": settings["target_u_fraction"],
        "target_v_fraction": settings["target_v_fraction"],
        "target_normal_fraction": settings["target_normal_fraction"],
    }


def light_pose_from_camera(camera_to_world: np.ndarray, yaw_offset: float, pitch_offset: float) -> np.ndarray:
    pose = np.asarray(camera_to_world, dtype=float)
    camera_forward = -pose[:3, 2]
    camera_right = pose[:3, 0]
    camera_up = pose[:3, 1]
    direction = camera_forward + math.sin(yaw_offset) * camera_right + math.sin(pitch_offset) * camera_up
    direction = direction / np.linalg.norm(direction)
    return look_at_camera_to_world(pose[:3, 3] - direction, pose[:3, 3], camera_up)


def save_depth_png(depth: np.ndarray, output_path: Path) -> None:
    mask = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[mask] = np.clip(np.rint(depth[mask] * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    imageio.imwrite(output_path, depth_mm)


def near_bright_depth_visual(depth: np.ndarray) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 0.0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(mask):
        return vis
    near = float(np.percentile(depth[mask], 1))
    far = float(np.percentile(depth[mask], 99))
    if far <= near:
        far = near + 1e-6
    normalized = np.clip((far - depth) / (far - near), 0.0, 1.0)
    vis[mask] = np.rint(normalized[mask] * 255.0).astype(np.uint8)
    return vis


def save_pair_plot(rgb: np.ndarray, depth_vis: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[1].imshow(depth_vis, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_montage(
    rows: list[dict[str, Any]],
    output_path: Path,
    path_root: Path,
    tile_size: int = 92,
    columns: int = 10,
) -> None:
    tiles = []
    for row in rows:
        rgb = Image.open(path_root / row["image_path"]).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        depth = (
            Image.open(path_root / row["depth_vis_path"])
            .convert("L")
            .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            .convert("RGB")
        )
        tile = Image.new("RGB", (tile_size * 2, tile_size), "white")
        tile.paste(rgb, (0, 0))
        tile.paste(depth, (tile_size, 0))
        tiles.append(tile)

    rows_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * tile_size * 2, rows_count * tile_size), "white")
    for idx, tile in enumerate(tiles):
        montage.paste(tile, ((idx % columns) * tile.width, (idx // columns) * tile.height))
    montage.save(output_path)


def render_dataset(args: argparse.Namespace) -> None:
    paths = output_paths(args)
    output_root = paths["output_root"]
    data_root = paths["data_root"]
    visualization_root = paths["visualization_root"]
    path_root = paths["path_root"]

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    source_map = discover_synthetic_sources(args)
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    rows: list[dict[str, Any]] = []
    summary_sources = {}

    source_sample_counts = per_source_sample_counts(args, list(source_map))

    for source_index, (source_folder, volumes) in enumerate(source_map.items()):
        source_root = data_root / source_folder
        image_root = source_root / "images"
        depth_root = source_root / "depth"
        depth_vis_root = source_root / "depth_vis"
        metadata_root = source_root / "metadata"
        if args.dataset_layout:
            source_plots_root = visualization_root / source_folder / "plots"
        else:
            source_plots_root = source_root / "plots"
        for path in [image_root, depth_root, depth_vis_root, metadata_root, source_plots_root]:
            path.mkdir(parents=True, exist_ok=True)

        counts = per_volume_view_counts(source_sample_counts[source_folder], len(volumes))
        source_rows: list[dict[str, Any]] = []

        for volume_index, (volume, view_count) in enumerate(zip(volumes, counts, strict=True)):
            print(f"[{source_folder}] loading {volume.mesh_path.name}", flush=True)
            mesh = trimesh.load(volume.mesh_path, process=False)
            if hasattr(mesh, "geometry"):
                mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values() if geom is not None])
            vertices = np.asarray(mesh.vertices)
            render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=args.camera_mode == "clinical_closeup")
            volume_metadata = (
                metadata_in_mesh_coordinates(load_volume_metadata(volume))
                if args.camera_mode == "clinical_closeup"
                else None
            )
            rng = np.random.default_rng(args.seed + source_index * 100_003 + volume_index * 1009)

            for view_index in range(view_count):
                if args.camera_mode == "clinical_closeup":
                    settings = clinical_closeup_view_settings(volume_index, view_index, view_count, rng)
                    camera = camera_for_clinical_closeup(volume_metadata or {}, settings)
                else:
                    settings = view_settings(volume_index, view_index, view_count, rng)
                    camera = camera_for_view(vertices, settings)
                sample_id = f"{source_folder}_m{volume_index:02d}_v{view_index:03d}"

                scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[settings["ambient"]] * 3)
                scene.add(render_mesh)
                scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=settings["directional_intensity"]),
                    pose=light_pose_from_camera(
                        np.asarray(camera["camera_to_world"], dtype=float),
                        settings["light_yaw_offset"],
                        settings["light_pitch_offset"],
                    ),
                )
                scene.add(
                    pyrender.PerspectiveCamera(yfov=np.deg2rad(settings["fov_deg"]), znear=0.005, zfar=5.0),
                    pose=np.asarray(camera["camera_to_world"], dtype=float),
                )

                color, depth = renderer.render(scene)
                rgb = color[:, :, :3].astype(np.uint8)
                depth = depth.astype(np.float32)
                depth_vis = near_bright_depth_visual(depth)

                image_path = image_root / f"{sample_id}_rgb.png"
                depth_npy_path = depth_root / f"{sample_id}_depth.npy"
                depth_png_path = depth_root / f"{sample_id}_depth_mm.png"
                depth_vis_path = depth_vis_root / f"{sample_id}_depth_vis.png"
                metadata_path = metadata_root / f"{sample_id}.json"
                plot_path = source_plots_root / f"{sample_id}_rgb_depth.png"

                imageio.imwrite(image_path, rgb)
                np.save(depth_npy_path, depth)
                save_depth_png(depth, depth_png_path)
                imageio.imwrite(depth_vis_path, depth_vis)
                save_pair_plot(rgb, depth_vis, plot_path)

                row = {
                    "sample_id": sample_id,
                    "source_folder": source_folder,
                    "volume_id": volume.volume_id,
                    "volume_index": volume_index,
                    "view_index": view_index,
                    "mesh_path": str(volume.mesh_path.relative_to(ROOT)),
                    "metadata_path": root_relative(volume.metadata_path) if volume.metadata_path is not None else "",
                    "image_path": str(image_path.relative_to(path_root)),
                    "depth_npy_path": str(depth_npy_path.relative_to(path_root)),
                    "depth_png_path": str(depth_png_path.relative_to(path_root)),
                    "depth_vis_path": str(depth_vis_path.relative_to(path_root)),
                    "plot_path": str(plot_path.relative_to(path_root)),
                    "camera_mode": args.camera_mode,
                    "depth_type": "camera_z_distance",
                    "depth_visualization": "near_bright_far_dark_background_black_infinite",
                    "width": args.image_size,
                    "height": args.image_size,
                    "fov_deg": settings["fov_deg"],
                    **camera,
                    "lighting": {
                        "ambient": settings["ambient"],
                        "directional_intensity": settings["directional_intensity"],
                        "light_yaw_offset": settings["light_yaw_offset"],
                        "light_pitch_offset": settings["light_pitch_offset"],
                    },
                }
                metadata_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
                rows.append(row)
                source_rows.append(row)
                print(f"[{source_folder}] rendered {sample_id}", flush=True)

        montage_path = source_plots_root / "montage_all_rgb_depth.png"
        build_montage(source_rows, montage_path, path_root)
        summary_sources[source_folder] = {
            "sample_count": len(source_rows),
            "mesh_count": len(volumes),
            "samples_per_mesh": counts,
            "folder": root_relative(source_root),
            "montage": root_relative(montage_path),
        }

    renderer.delete()

    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "source_folder",
            "volume_id",
            "volume_index",
            "view_index",
            "mesh_path",
            "metadata_path",
            "image_path",
            "depth_npy_path",
            "depth_png_path",
            "depth_vis_path",
            "plot_path",
            "camera_mode",
            "depth_type",
            "depth_visualization",
            "width",
            "height",
            "fov_deg",
            "angle_deg",
            "distance_scale",
            "elevation_fraction",
            "target_up_fraction",
            "target_h0_fraction",
            "target_h1_fraction",
            "lesion_radius_m",
            "lesion_height_m",
            "clinical_frame_half_height_m",
            "clinical_camera_distance_m",
            "clinical_off_axis_deg",
            "clinical_roll_deg",
            "target_u_fraction",
            "target_v_fraction",
            "target_normal_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "sample_count": len(rows),
        "source_folder_count": len(source_map),
        "source_folders": list(source_map),
        "requested_total_samples": args.total_samples,
        "samples_per_source_folder": args.samples_per_folder if args.total_samples is None else None,
        "source_sample_counts": source_sample_counts,
        "camera_mode": args.camera_mode,
        "image_size": args.image_size,
        "seed": args.seed,
        "output_root": root_relative(output_root),
        "data_root": root_relative(data_root),
        "visualization_root": root_relative(visualization_root),
        "paths_relative_to": root_relative(path_root),
        "dataset_layout": bool(args.dataset_layout),
        "sources": summary_sources,
    }
    (data_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render RGB/depth-map pairs for each single-lesion synthetic source folder.")
    parser.add_argument("--output_root", type=str, default=None, help="Output folder. Relative paths are resolved from the repo root.")
    parser.add_argument(
        "--dataset_layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output_root/data and output_root/visualizations subfolders.",
    )
    parser.add_argument(
        "--source_folder",
        action="append",
        default=[],
        help="Exact folder below data/synthetic/single_lesion/body_parts to render. May be repeated.",
    )
    parser.add_argument("--source_prefix", action="append", default=[], help="Render source folders with this prefix. May be repeated.")
    parser.add_argument("--camera_mode", choices=["full_body", "clinical_closeup"], default="clinical_closeup")
    parser.add_argument("--samples_per_folder", type=int, default=100)
    parser.add_argument("--total_samples", type=int, default=None, help="Total RGB/depth pairs to render across all source folders.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    render_dataset(build_parser().parse_args())

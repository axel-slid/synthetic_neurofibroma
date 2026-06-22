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
from trimesh.visual.texture import TextureVisuals

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[3]
HSR_ROOT = ROOT / "data" / "hsr" / "scans"
OUTPUT_ROOT = ROOT / "data" / "depth_maps"
BASE_ROOT = OUTPUT_ROOT / "base"
EXAMPLES_ROOT = BASE_ROOT / "images"
PLOTS_ROOT = BASE_ROOT / "plots"


@dataclass
class Subject:
    subject_id: str
    obj_path: Path
    texture_path: Path
    person_metadata: dict[str, Any]
    pose_metadata: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_subjects(hsr_root: Path) -> list[Subject]:
    subjects = []
    for obj_path in sorted(hsr_root.glob("*/scan/*.obj")):
        subject_id = obj_path.stem
        texture_path = obj_path.parent / f"{subject_id}_u0_v0_diffuse.png"
        if not texture_path.exists():
            raise FileNotFoundError(f"Missing texture for {subject_id}: {texture_path}")
        subjects.append(
            Subject(
                subject_id=subject_id,
                obj_path=obj_path,
                texture_path=texture_path,
                person_metadata=load_json(obj_path.parent / "person_metadata.json"),
                pose_metadata=load_json(obj_path.parent / "pose_metadata.json"),
            )
        )
    if not subjects:
        raise FileNotFoundError(f"No HSR OBJ files found below {hsr_root}")
    return subjects


def load_textured_mesh(subject: Subject, max_texture_px: int) -> trimesh.Trimesh:
    mesh = trimesh.load(subject.obj_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values() if geom is not None])
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        raise ValueError(f"{subject.subject_id} has no UV coordinates")
    texture = Image.open(subject.texture_path).convert("RGB")
    texture.thumbnail((max_texture_px, max_texture_px), Image.Resampling.LANCZOS)
    mesh.visual = TextureVisuals(uv=mesh.visual.uv, image=texture)
    return mesh


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


def view_settings(subject_index: int, view_index: int, views_per_subject: int, rng: np.random.Generator) -> dict[str, float]:
    base_angle = 360.0 * view_index / views_per_subject
    angle_deg = base_angle + float(rng.uniform(-0.42, 0.42) * 360.0 / views_per_subject)
    return {
        "angle_deg": angle_deg % 360.0,
        "fov_deg": float(rng.uniform(24.0, 54.0)),
        "distance_scale": float(rng.uniform(1.02, 1.62)),
        "elevation_fraction": float(rng.uniform(-0.16, 0.18)),
        "target_up_fraction": float(rng.uniform(-0.08, 0.10)),
        "target_h0_fraction": float(rng.uniform(-0.06, 0.06)),
        "target_h1_fraction": float(rng.uniform(-0.06, 0.06)),
        "ambient": float(rng.uniform(0.28, 0.78)),
        "directional_intensity": float(rng.uniform(0.65, 2.65)),
        "light_yaw_offset": float(math.radians(rng.uniform(-85.0, 85.0))),
        "light_pitch_offset": float(math.radians(rng.uniform(-35.0, 35.0))),
        "subject_index": float(subject_index),
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
        "angle_deg": float(settings["angle_deg"]),
        "eye_xyz": [float(v) for v in eye],
        "target_xyz": [float(v) for v in target],
        "up_axis": up_axis,
        "horizontal_axes": horizontal_axes,
        "camera_to_world": look_at_camera_to_world(eye, target, up).tolist(),
        "bounds_min": [float(v) for v in vmin],
        "bounds_max": [float(v) for v in vmax],
        "extents": [float(v) for v in extents],
        "distance_scale": float(settings["distance_scale"]),
        "elevation_fraction": float(settings["elevation_fraction"]),
        "target_up_fraction": float(settings["target_up_fraction"]),
        "target_h0_fraction": float(settings["target_h0_fraction"]),
        "target_h1_fraction": float(settings["target_h1_fraction"]),
    }


def light_pose_from_camera(camera_to_world: np.ndarray, yaw_offset: float, pitch_offset: float) -> np.ndarray:
    pose = np.asarray(camera_to_world, dtype=float).copy()
    camera_forward = -pose[:3, 2]
    camera_right = pose[:3, 0]
    camera_up = pose[:3, 1]
    direction = camera_forward + math.sin(yaw_offset) * camera_right + math.sin(pitch_offset) * camera_up
    direction = direction / np.linalg.norm(direction)
    eye = pose[:3, 3] - direction
    target = pose[:3, 3]
    return look_at_camera_to_world(eye, target, camera_up)


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


def save_comparison_plot(rgb: np.ndarray, depth_vis: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[1].imshow(depth_vis, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_montage(rows: list[dict[str, Any]], output_path: Path, tile_size: int = 92, columns: int = 10) -> None:
    tiles = []
    for row in rows:
        rgb = Image.open(BASE_ROOT / row["image_path"]).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        depth = Image.open(BASE_ROOT / row["depth_vis_path"]).convert("L").resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert("RGB")
        tile = Image.new("RGB", (tile_size * 2, tile_size), "white")
        tile.paste(rgb, (0, 0))
        tile.paste(depth, (tile_size, 0))
        tiles.append(tile)

    rows_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * tile_size * 2, rows_count * tile_size), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile.width
        y = (idx // columns) * tile.height
        montage.paste(tile, (x, y))
    montage.save(output_path)


def render_dataset(args: argparse.Namespace) -> None:
    if BASE_ROOT.exists() and args.overwrite:
        shutil.rmtree(BASE_ROOT)

    image_root = EXAMPLES_ROOT / "images"
    depth_root = EXAMPLES_ROOT / "depth"
    depth_vis_root = EXAMPLES_ROOT / "depth_vis"
    metadata_root = EXAMPLES_ROOT / "metadata"
    for path in [image_root, depth_root, depth_vis_root, metadata_root, PLOTS_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(HSR_ROOT)
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    rows: list[dict[str, Any]] = []

    for subject_index, subject in enumerate(subjects):
        print(f"[{subject.subject_id}] loading mesh", flush=True)
        mesh = load_textured_mesh(subject, args.max_texture_px)
        vertices = np.asarray(mesh.vertices)
        rng = np.random.default_rng(int(args.seed) + subject_index * 1009)

        for view_index in range(args.views_per_subject):
            settings = view_settings(subject_index, view_index, args.views_per_subject, rng)
            sample_id = f"{subject.subject_id}_v{view_index:03d}"
            camera = camera_for_view(vertices, settings)

            scene = pyrender.Scene(
                bg_color=[255, 255, 255, 255],
                ambient_light=[settings["ambient"]] * 3,
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=settings["directional_intensity"]),
                pose=light_pose_from_camera(
                    np.asarray(camera["camera_to_world"], dtype=float),
                    settings["light_yaw_offset"],
                    settings["light_pitch_offset"],
                ),
            )
            scene.add(
                pyrender.PerspectiveCamera(yfov=np.deg2rad(settings["fov_deg"])),
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
            plot_path = PLOTS_ROOT / f"{sample_id}_rgb_depth.png"

            imageio.imwrite(image_path, rgb)
            np.save(depth_npy_path, depth)
            save_depth_png(depth, depth_png_path)
            imageio.imwrite(depth_vis_path, depth_vis)
            save_comparison_plot(rgb, depth_vis, plot_path)

            row = {
                "sample_id": sample_id,
                "subject_id": subject.subject_id,
                "view_index": view_index,
                "image_path": str(image_path.relative_to(BASE_ROOT)),
                "depth_npy_path": str(depth_npy_path.relative_to(BASE_ROOT)),
                "depth_png_path": str(depth_png_path.relative_to(BASE_ROOT)),
                "depth_vis_path": str(depth_vis_path.relative_to(BASE_ROOT)),
                "plot_path": str(plot_path.relative_to(BASE_ROOT)),
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
                "obj_path": str(subject.obj_path),
                "texture_path": str(subject.texture_path),
                "person_metadata": subject.person_metadata,
                "pose_metadata": subject.pose_metadata,
            }
            metadata_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            print(
                f"[{subject.subject_id}] rendered {sample_id} "
                f"angle={settings['angle_deg']:.1f} fov={settings['fov_deg']:.1f} "
                f"dist={settings['distance_scale']:.2f}",
                flush=True,
            )

        del mesh

    renderer.delete()

    manifest_csv = BASE_ROOT / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "subject_id",
            "view_index",
            "image_path",
            "depth_npy_path",
            "depth_png_path",
            "depth_vis_path",
            "plot_path",
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    montage_path = BASE_ROOT / "montage_all_rgb_depth.png"
    build_montage(rows, montage_path)

    summary = {
        "sample_count": len(rows),
        "subject_count": len(subjects),
        "views_per_subject": args.views_per_subject,
        "image_size": args.image_size,
        "seed": args.seed,
        "camera_variation": {
            "angle_deg": "full 360 degrees with jitter",
            "fov_deg": [24.0, 54.0],
            "distance_scale": [1.02, 1.62],
            "elevation_fraction": [-0.16, 0.18],
            "target_offsets_fraction": [-0.08, 0.10],
        },
        "lighting_variation": {
            "ambient": [0.28, 0.78],
            "directional_intensity": [0.65, 2.65],
            "light_yaw_offset_deg": [-85.0, 85.0],
            "light_pitch_offset_deg": [-35.0, 35.0],
        },
        "folders": {
            "base_images": str(image_root.relative_to(ROOT)),
            "base_depth": str(depth_root.relative_to(ROOT)),
            "base_depth_visualizations": str(depth_vis_root.relative_to(ROOT)),
            "base_metadata": str(metadata_root.relative_to(ROOT)),
            "plots": str(PLOTS_ROOT.relative_to(ROOT)),
            "montage": str(montage_path.relative_to(ROOT)),
        },
    }
    (BASE_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render RGB/depth-map pairs from local HSR textured meshes.")
    parser.add_argument("--views_per_subject", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_texture_px", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    render_dataset(build_parser().parse_args())

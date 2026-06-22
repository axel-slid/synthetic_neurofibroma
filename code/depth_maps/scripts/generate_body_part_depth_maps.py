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
import nbformat
import numpy as np
import pyrender
import trimesh
from nbclient import NotebookClient
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "depth_maps" / "body_parts"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
SEGMENTATION_ROOT = ROOT / "data" / "hsr" / "body_part_segmentation" / "data"

BODY_PARTS = {
    "torso_front": "front",
    "torso_back": "back",
    "face": "face",
    "arms": "arms",
    "hands": "hands",
    "legs": "legs",
    "feet": "feet",
}

FRAME_HEIGHT_RANGES_M = {
    "torso_front": (0.34, 0.58),
    "torso_back": (0.34, 0.58),
    "face": (0.18, 0.30),
    "arms": (0.20, 0.36),
    "hands": (0.12, 0.22),
    "legs": (0.24, 0.44),
    "feet": (0.14, 0.26),
}


@dataclass
class ScanParts:
    scan_id: str
    vertices: np.ndarray
    triangles: np.ndarray
    vertex_colors: np.ndarray
    vertex_normals: np.ndarray
    face_labels: np.ndarray
    label_names: list[str]
    front_sign: int


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_output_root(output_root: str | None) -> Path:
    if output_root is None:
        return DEFAULT_OUTPUT_ROOT
    path = Path(output_root)
    return path if path.is_absolute() else ROOT / path


def load_scan_parts(npz_path: Path) -> ScanParts:
    payload = np.load(npz_path, allow_pickle=True)
    return ScanParts(
        scan_id=str(payload["scan_id"]),
        vertices=payload["vertices"].astype(np.float32),
        triangles=payload["triangles"].astype(np.int32),
        vertex_colors=payload["vertex_colors"].astype(np.uint8),
        vertex_normals=payload["vertex_normals"].astype(np.float32),
        face_labels=payload["face_labels"].astype(np.int32),
        label_names=[str(label) for label in payload["label_names"].tolist()],
        front_sign=int(payload["front_sign"]),
    )


def discover_scans() -> list[ScanParts]:
    paths = sorted(SEGMENTATION_ROOT.glob("*_body_part_segmentation.npz"))
    if not paths:
        raise FileNotFoundError(f"No body-part segmentation NPZ files found below {SEGMENTATION_ROOT}")
    return [load_scan_parts(path) for path in paths]


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


def normalized(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-10:
        return vector / norm
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=float)
    return normalized(fallback)


def rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalized(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def compute_face_geometry(scan: ScanParts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles_xyz = scan.vertices[scan.triangles]
    centroids = triangles_xyz.mean(axis=1)
    normals = scan.vertex_normals[scan.triangles].mean(axis=1).astype(float)
    edge_a = triangles_xyz[:, 1] - triangles_xyz[:, 0]
    edge_b = triangles_xyz[:, 2] - triangles_xyz[:, 0]
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) / 2.0

    mesh_center = scan.vertices.mean(axis=0)
    outward = centroids - mesh_center
    dot = np.einsum("ij,ij->i", normals, outward)
    normals[dot < 0.0] *= -1.0
    normal_norm = np.linalg.norm(normals, axis=1)
    fallback = outward / np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-10)
    normals[normal_norm <= 1e-10] = fallback[normal_norm <= 1e-10]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    return centroids, normals, areas


def body_part_face_indices(scan: ScanParts, source_label: str) -> np.ndarray:
    label_idx = scan.label_names.index(source_label)
    indices = np.flatnonzero(scan.face_labels == label_idx)
    if len(indices) == 0:
        raise ValueError(f"{scan.scan_id} has no faces for label {source_label}")
    return indices


def choose_target_face(
    indices: np.ndarray,
    areas: np.ndarray,
    rng: np.random.Generator,
) -> int:
    weights = areas[indices].astype(float)
    if not np.any(weights > 0.0):
        return int(rng.choice(indices))
    weights = weights / weights.sum()
    return int(rng.choice(indices, p=weights))


def camera_for_body_part(
    scan: ScanParts,
    body_part: str,
    source_label: str,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, float]]:
    indices = body_part_face_indices(scan, source_label)
    face_index = choose_target_face(indices, areas, rng)
    target = centroids[face_index].astype(float)
    view_direction = normalized(normals[face_index].astype(float), fallback=target - scan.vertices.mean(axis=0))

    off_axis = math.radians(float(rng.uniform(0.0, 18.0)))
    azimuth = math.radians(float(rng.uniform(0.0, 360.0)))
    tangent_a = normalized(np.cross(view_direction, np.array([0.0, 0.0, 1.0], dtype=float)), fallback=np.array([1.0, 0.0, 0.0]))
    tangent_b = normalized(np.cross(view_direction, tangent_a))
    side_direction = normalized(math.cos(azimuth) * tangent_a + math.sin(azimuth) * tangent_b)
    view_direction = normalized(math.cos(off_axis) * view_direction + math.sin(off_axis) * side_direction)

    frame_min, frame_max = FRAME_HEIGHT_RANGES_M[body_part]
    frame_height = float(rng.uniform(frame_min, frame_max))
    fov_deg = float(rng.uniform(34.0, 50.0))
    distance = max(frame_height / (2.0 * math.tan(math.radians(fov_deg) / 2.0)), 0.08)
    eye = target + distance * view_direction

    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up, view_direction))) > 0.92:
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    up = up - np.dot(up, view_direction) * view_direction
    up = normalized(up)
    roll = math.radians(float(rng.uniform(-15.0, 15.0)))
    up = rotation_around_axis(view_direction, roll) @ up

    settings = {
        "fov_deg": fov_deg,
        "distance_m": float(distance),
        "frame_height_m": frame_height,
        "off_axis_deg": math.degrees(off_axis),
        "roll_deg": math.degrees(roll),
        "ambient": float(rng.uniform(0.34, 0.76)),
        "directional_intensity": float(rng.uniform(0.65, 2.0)),
        "light_yaw_offset": float(math.radians(rng.uniform(-55.0, 55.0))),
        "light_pitch_offset": float(math.radians(rng.uniform(-35.0, 35.0))),
    }
    camera = {
        "face_index": face_index,
        "target_xyz": [float(v) for v in target],
        "eye_xyz": [float(v) for v in eye],
        "view_direction_xyz": [float(v) for v in view_direction],
        "camera_to_world": look_at_camera_to_world(eye, target, up).tolist(),
    }
    return camera, settings


def light_pose_from_camera(camera_to_world: np.ndarray, yaw_offset: float, pitch_offset: float) -> np.ndarray:
    pose = np.asarray(camera_to_world, dtype=float)
    camera_forward = -pose[:3, 2]
    camera_right = pose[:3, 0]
    camera_up = pose[:3, 1]
    direction = camera_forward + math.sin(yaw_offset) * camera_right + math.sin(pitch_offset) * camera_up
    direction = normalized(direction)
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
    normalized_depth = np.clip((far - depth) / (far - near), 0.0, 1.0)
    vis[mask] = np.rint(normalized_depth[mask] * 255.0).astype(np.uint8)
    return vis


def save_pair_figure(rgb: np.ndarray, depth_vis: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[1].imshow(depth_vis, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def output_part_dirs(data_root: Path, body_part: str) -> tuple[Path, Path, Path]:
    part_root = data_root / body_part
    image_root = part_root / "2d_images"
    depth_root = part_root / "2d_gt_depth_maps"
    figure_root = part_root / "2d_rgb_gt_depth_figures"
    for path in (image_root, depth_root, figure_root):
        path.mkdir(parents=True, exist_ok=True)
    return image_root, depth_root, figure_root


def rows_for_montage(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    body_parts = sorted({row["body_part"] for row in rows})
    selected = []
    for body_part in body_parts:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        take = max(1, count // len(body_parts))
        indices = np.linspace(0, len(part_rows) - 1, min(take, len(part_rows)), dtype=int)
        selected.extend(part_rows[int(index)] for index in indices)
    if len(selected) < count:
        selected_ids = {row["sample_id"] for row in selected}
        remaining = [row for row in rows if row["sample_id"] not in selected_ids]
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def build_montage(rows: list[dict[str, Any]], output_path: Path, tile_height: int = 96, columns: int = 10) -> None:
    tiles = []
    for row in rows:
        figure = Image.open(OUTPUT_ROOT / row["figure_path"]).convert("RGB")
        width = tile_height * 2
        tile = figure.resize((width, tile_height), Image.Resampling.LANCZOS)
        tiles.append(tile)

    row_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * tile_height * 2, row_count * tile_height), "white")
    for idx, tile in enumerate(tiles):
        montage.paste(tile, ((idx % columns) * tile.width, (idx // columns) * tile.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def build_preview_gif(rows: list[dict[str, Any]], output_path: Path, frame_count: int = 32, tile_size: int = 192) -> None:
    frames = []
    selected = rows_for_montage(rows, frame_count)
    for row in selected:
        rgb = Image.open(OUTPUT_ROOT / row["image_path"]).convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        depth = (
            Image.open(OUTPUT_ROOT / row["depth_vis_path"])
            .convert("L")
            .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            .convert("RGB")
        )
        frame = Image.new("RGB", (tile_size * 2, tile_size), "white")
        frame.paste(rgb, (0, 0))
        frame.paste(depth, (tile_size, 0))
        frames.append(np.asarray(frame))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=0.18, loop=0)


def build_plotly_notebook(rows: list[dict[str, Any]], output_path: Path) -> None:
    row = rows[len(rows) // 2]
    dataset_root_text = str(OUTPUT_ROOT)
    cells = [
        nbformat.v4.new_markdown_cell("# Body-part RGB / GT depth surface\n\nExecuted Plotly notebook for one representative body-part sample."),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"DATASET_ROOT = Path({dataset_root_text!r})\n"
            f"SAMPLE_ID = {row['sample_id']!r}\n"
            "rows = list(csv.DictReader((DATASET_ROOT / 'data' / 'manifest.csv').open(newline='', encoding='utf-8')))\n"
            "row = next(item for item in rows if item['sample_id'] == SAMPLE_ID)\n"
            "row"
        ),
        nbformat.v4.new_code_cell(
            "depth = np.load(DATASET_ROOT / row['depth_npy_path']).astype(float)\n"
            "rgb = np.array(Image.open(DATASET_ROOT / row['image_path']).convert('RGB'))\n"
            "stride = 4\n"
            "z = depth[::stride, ::stride]\n"
            "z[~np.isfinite(z) | (z <= 0)] = np.nan\n"
            "yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]\n"
            "fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'xy'}]], column_widths=[0.62, 0.38])\n"
            "fig.add_trace(go.Surface(x=xx, y=-yy, z=z, surfacecolor=z, colorscale='Viridis', colorbar={'title': 'meters'}, connectgaps=False), row=1, col=1)\n"
            "fig.add_trace(go.Image(z=rgb), row=1, col=2)\n"
            "fig.update_layout(width=1050, height=560, title=f\"{row['body_part']} / {row['scan_id']} RGB and GT depth\")\n"
            "fig.update_layout(scene={'xaxis_title': 'x pixels / stride', 'yaxis_title': 'y pixels / stride', 'zaxis_title': 'camera z distance (m)', 'aspectmode': 'data'})\n"
            "fig.update_xaxes(showticklabels=False, row=1, col=2)\n"
            "fig.update_yaxes(showticklabels=False, row=1, col=2)\n"
            "fig"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    NotebookClient(notebook, timeout=180, kernel_name="python3").execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


def write_manifest(data_root: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_path = data_root / "manifest.csv"
    fieldnames = [
        "sample_id",
        "scan_id",
        "body_part",
        "source_label",
        "image_path",
        "depth_npy_path",
        "depth_png_path",
        "depth_vis_path",
        "figure_path",
        "depth_type",
        "width",
        "height",
        "face_index",
        "fov_deg",
        "camera_distance_m",
        "frame_height_m",
        "off_axis_deg",
        "roll_deg",
        "target_xyz",
        "eye_xyz",
        "view_direction_xyz",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return manifest_path


def render_dataset(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT
    OUTPUT_ROOT = resolve_output_root(args.output_root)

    if OUTPUT_ROOT.exists() and args.overwrite:
        shutil.rmtree(OUTPUT_ROOT)

    data_root = OUTPUT_ROOT / "data"
    visualization_root = OUTPUT_ROOT / "visualizations"
    data_root.mkdir(parents=True, exist_ok=True)
    visualization_root.mkdir(parents=True, exist_ok=True)

    scans = discover_scans()
    renderer = pyrender.OffscreenRenderer(viewport_width=args.image_size, viewport_height=args.image_size)
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {body_part: {} for body_part in BODY_PARTS}

    for scan_index, scan in enumerate(scans):
        mesh = trimesh.Trimesh(
            vertices=scan.vertices,
            faces=scan.triangles,
            vertex_colors=scan.vertex_colors,
            process=False,
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        centroids, normals, areas = compute_face_geometry(scan)

        for body_part, source_label in BODY_PARTS.items():
            image_root, depth_root, figure_root = output_part_dirs(data_root, body_part)
            source_counts[body_part][scan.scan_id] = args.samples_per_body_part
            rng = np.random.default_rng(args.seed + scan_index * 100_003 + len(rows) * 17)

            for sample_index in range(args.samples_per_body_part):
                camera, settings = camera_for_body_part(scan, body_part, source_label, centroids, normals, areas, rng)
                sample_id = f"{body_part}_{scan.scan_id}_s{sample_index:03d}"
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

                image_path = image_root / f"{sample_id}_2d.png"
                depth_npy_path = depth_root / f"{sample_id}_gt_depth_m.npy"
                depth_png_path = depth_root / f"{sample_id}_gt_depth_mm.png"
                depth_vis_path = depth_root / f"{sample_id}_gt_depth_vis.png"
                figure_path = figure_root / f"{sample_id}_2d_gt_depth.png"

                imageio.imwrite(image_path, rgb)
                np.save(depth_npy_path, depth)
                save_depth_png(depth, depth_png_path)
                imageio.imwrite(depth_vis_path, depth_vis)
                save_pair_figure(rgb, depth_vis, figure_path)

                row = {
                    "sample_id": sample_id,
                    "scan_id": scan.scan_id,
                    "body_part": body_part,
                    "source_label": source_label,
                    "image_path": str(image_path.relative_to(OUTPUT_ROOT)),
                    "depth_npy_path": str(depth_npy_path.relative_to(OUTPUT_ROOT)),
                    "depth_png_path": str(depth_png_path.relative_to(OUTPUT_ROOT)),
                    "depth_vis_path": str(depth_vis_path.relative_to(OUTPUT_ROOT)),
                    "figure_path": str(figure_path.relative_to(OUTPUT_ROOT)),
                    "depth_type": "camera_z_distance",
                    "width": args.image_size,
                    "height": args.image_size,
                    "face_index": camera["face_index"],
                    "fov_deg": settings["fov_deg"],
                    "camera_distance_m": settings["distance_m"],
                    "frame_height_m": settings["frame_height_m"],
                    "off_axis_deg": settings["off_axis_deg"],
                    "roll_deg": settings["roll_deg"],
                    "target_xyz": camera["target_xyz"],
                    "eye_xyz": camera["eye_xyz"],
                    "view_direction_xyz": camera["view_direction_xyz"],
                }
                rows.append(row)
                print(f"[{body_part}] rendered {sample_id}", flush=True)

    renderer.delete()

    manifest_path = write_manifest(data_root, rows)
    montage_rows = rows_for_montage(rows, args.montage_count)
    montage_path = visualization_root / f"montage_{args.montage_count}_rgb_gt_depth.png"
    build_montage(montage_rows, montage_path)
    gif_path = visualization_root / "body_parts_rgb_gt_depth_preview.gif"
    build_preview_gif(rows, gif_path)
    notebook_path = visualization_root / "body_parts_depth_surface.ipynb"
    build_plotly_notebook(rows, notebook_path)

    summary = {
        "dataset": "body_parts",
        "sample_count": len(rows),
        "scan_count": len(scans),
        "samples_per_body_part_per_scan": args.samples_per_body_part,
        "body_parts": BODY_PARTS,
        "source_counts": source_counts,
        "image_size": args.image_size,
        "seed": args.seed,
        "layout": {
            "root": root_relative(OUTPUT_ROOT),
            "data_root": root_relative(data_root),
            "visualizations": root_relative(visualization_root),
            "per_body_part_folders": ["2d_images", "2d_gt_depth_maps", "2d_rgb_gt_depth_figures"],
        },
        "montage": root_relative(montage_path),
        "gif": root_relative(gif_path),
        "plotly_notebook": root_relative(notebook_path),
    }
    (data_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render RGB/GT-depth body-part crops from HSR body-part segmentations.")
    parser.add_argument("--output_root", type=str, default=None, help="Output folder. Relative paths are resolved from the repo root.")
    parser.add_argument("--samples_per_body_part", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--montage_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    render_dataset(build_parser().parse_args())

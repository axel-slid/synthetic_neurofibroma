#!/usr/bin/env python3
"""Bake HSR OBJ texture colors into connected decimated PLY meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[4]


def parse_face_token(token: str) -> tuple[int, int]:
    parts = token.split("/")
    vertex_idx = int(parts[0]) - 1
    texture_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else vertex_idx
    return vertex_idx, texture_idx


def load_obj_with_texture_colors(obj_path: Path, texture_path: Path, texture_size: int | None) -> o3d.geometry.TriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    color_sums: np.ndarray | None = None
    color_counts: np.ndarray | None = None

    image = Image.open(texture_path).convert("RGB")
    if texture_size is not None:
        image.thumbnail((texture_size, texture_size), Image.Resampling.LANCZOS)
    texture = np.asarray(image, dtype=np.uint8)
    tex_h, tex_w = texture.shape[:2]

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith("vt "):
                texcoords.append(tuple(map(float, line.split()[1:3])))

    color_sums = np.zeros((len(vertices), 3), dtype=np.float64)
    color_counts = np.zeros(len(vertices), dtype=np.float64)
    texcoords_arr = np.asarray(texcoords, dtype=np.float32)

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            parsed = [parse_face_token(token) for token in line.split()[1:4]]
            face = []
            for vertex_idx, texture_idx in parsed:
                face.append(vertex_idx)
                if 0 <= texture_idx < len(texcoords_arr):
                    u, v = texcoords_arr[texture_idx]
                else:
                    u, v = 0.5, 0.5
                px = int(np.clip(round(u * (tex_w - 1)), 0, tex_w - 1))
                py = int(np.clip(round((1.0 - v) * (tex_h - 1)), 0, tex_h - 1))
                color_sums[vertex_idx] += texture[py, px].astype(np.float64) / 255.0
                color_counts[vertex_idx] += 1.0
            faces.append(tuple(face))

    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)
    colors = np.zeros_like(color_sums, dtype=np.float64)
    valid = color_counts > 0
    colors[valid] = color_sums[valid] / color_counts[valid, None]
    if np.any(~valid):
        colors[~valid] = colors[valid].mean(axis=0)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_arr),
        o3d.utility.Vector3iVector(faces_arr),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    return mesh


def bake_scan(
    scan_root: Path,
    out_dir: Path,
    scan_id: str,
    target_faces: int,
    texture_size: int,
    method: str,
    voxel_size: float,
) -> Path:
    scan_dir = scan_root / scan_id / "scan"
    obj_path = scan_dir / f"{scan_id}.obj"
    texture_path = scan_dir / f"{scan_id}_u0_v0_diffuse.png"
    out_path = out_dir / f"{scan_id}_closed_textured_mesh.ply"

    mesh = load_obj_with_texture_colors(obj_path, texture_path, texture_size)
    original_faces = len(mesh.triangles)
    original_vertices = len(mesh.vertices)
    if method == "quadric":
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    elif method == "vertex-clustering":
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
    else:
        raise ValueError(f"Unknown simplification method: {method}")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False, compressed=False, write_vertex_colors=True)

    colors = np.asarray(mesh.vertex_colors)
    print(
        f"{scan_id}: {original_vertices:,}v/{original_faces:,}f -> "
        f"{len(mesh.vertices):,}v/{len(mesh.triangles):,}f; "
        f"method={method}; mean_rgb={np.round(colors.mean(axis=0) * 255, 1).tolist()}; {out_path}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsr-root", type=Path, default=ROOT / "data" / "hsr")
    parser.add_argument("--target-faces", type=int, default=120_000)
    parser.add_argument("--texture-size", type=int, default=0, help="Maximum texture side; 0 uses the native full-resolution texture.")
    parser.add_argument("--method", choices=["quadric", "vertex-clustering"], default="vertex-clustering")
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--scan-id", action="append", default=None)
    args = parser.parse_args()

    scan_root = args.hsr_root / "scans"
    out_dir = args.hsr_root / "visualizations" / "meshes"
    scan_ids = args.scan_id or ["HSR0018-Body-070", "HSR0152-Body-090"]
    for scan_id in scan_ids:
        texture_size = None if args.texture_size == 0 else args.texture_size
        bake_scan(scan_root, out_dir, scan_id, args.target_faces, texture_size, args.method, args.voxel_size)


if __name__ == "__main__":
    main()

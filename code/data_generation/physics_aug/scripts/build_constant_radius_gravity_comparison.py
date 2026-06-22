#!/usr/bin/env python3
"""Compare constant-radius lesion physics methods with interpolated coloring."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from plotly.utils import PlotlyJSONEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_physics_aug_growth import (  # noqa: E402
    HSR_MESH_ROOT,
    ROOT,
    GrowthFrame,
    combine_base_and_lesion,
    compute_vertex_normals,
    crop_mesh_to_target,
    localize_points,
    pick_target_vertex,
    read_colored_ply,
    remove_degenerate_faces,
    rgb_strings,
    sample_skin_and_color,
    simulate_soft_body_surface,
    target_basis,
    write_colored_ply,
)

DATASET_ROOT = ROOT / "data" / "synthetic" / "physics_aug_constant_radius"


@dataclass(frozen=True)
class PhysicsVariant:
    method_id: str
    label: str
    solver: str
    gravity_scale: float
    shape_memory: float
    contact_adhesion: float
    sag_gain: float
    compression: float
    roundness: float
    notes: str


def variant_plan() -> list[PhysicsVariant]:
    return [
        PhysicsVariant(
            method_id="analytic_no_gravity",
            label="Analytic control",
            solver="analytic_profile",
            gravity_scale=0.0,
            shape_memory=1.0,
            contact_adhesion=0.0,
            sag_gain=0.0,
            compression=0.0,
            roundness=1.42,
            notes="Constant-radius pressure dome with no gravity term.",
        ),
        PhysicsVariant(
            method_id="analytic_gravity_sag",
            label="Analytic gravity sag",
            solver="analytic_profile_plus_gravity_offset",
            gravity_scale=1.0,
            shape_memory=1.0,
            contact_adhesion=0.0,
            sag_gain=0.70,
            compression=0.20,
            roundness=1.42,
            notes="Adds a closed-form gravity displacement along the local downhill tangent while keeping the rim fixed.",
        ),
        PhysicsVariant(
            method_id="mass_spring_low_gravity",
            label="Mass-spring low gravity",
            solver="pbd_mass_spring",
            gravity_scale=1.25,
            shape_memory=0.075,
            contact_adhesion=0.05,
            sag_gain=0.20,
            compression=0.06,
            roundness=1.42,
            notes="Position-based mass-spring relaxation with a moderate gravity term and weak skin adhesion.",
        ),
        PhysicsVariant(
            method_id="mass_spring_contact_gravity",
            label="Mass-spring contact gravity",
            solver="pbd_mass_spring_contact",
            gravity_scale=3.65,
            shape_memory=0.035,
            contact_adhesion=0.38,
            sag_gain=0.48,
            compression=0.16,
            roundness=1.42,
            notes="Stronger gravity, lower shape memory, and contact adhesion to show settling/plop behavior.",
        ),
    ]


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


def lesion_template(
    support_radius: float,
    height: float,
    roundness: float,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_points = [np.array([0.0, 0.0], dtype=np.float32)]
    profile_heights = [float(height)]
    radial_weight = [0.0]
    ring_index = [0]

    for ring in range(1, radial_segments + 1):
        rho = support_radius * ring / radial_segments
        q = float(np.clip(rho / support_radius, 0.0, 1.0))
        profile = height * np.power(np.clip(1.0 - q * q, 0.0, 1.0), roundness)
        for step in range(angular_segments):
            theta = 2.0 * np.pi * step / angular_segments
            local_points.append(np.array([rho * np.cos(theta), rho * np.sin(theta)], dtype=np.float32))
            profile_heights.append(float(profile))
            radial_weight.append(q)
            ring_index.append(ring)

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
        np.asarray(ring_index, dtype=np.int32),
        np.asarray(faces, dtype=np.int32),
    )


def quadratic_skin_points(
    local_points: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    support_radius: float,
) -> np.ndarray:
    centroids = skin_vertices[skin_faces].mean(axis=1)
    offsets = centroids - anchor
    local_u = offsets @ tangent_u
    local_v = offsets @ tangent_v
    local_n = offsets @ normal
    radial = np.sqrt(local_u * local_u + local_v * local_v)
    candidate = (radial <= max(0.095, 2.25 * support_radius)) & (
        np.abs(local_n) <= max(0.050, 1.4 * support_radius)
    )
    if int(candidate.sum()) < 8:
        candidate = radial <= max(0.120, 2.8 * support_radius)

    fit_u = local_u[candidate]
    fit_v = local_v[candidate]
    fit_n = local_n[candidate]
    fit_radius = radial[candidate]
    if len(fit_n) < 6:
        return anchor + local_points[:, 0, None] * tangent_u + local_points[:, 1, None] * tangent_v

    weights = np.exp(-0.5 * (fit_radius / max(0.040, 1.35 * support_radius)) ** 2)
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
    u = local_points[:, 0]
    v = local_points[:, 1]
    fitted_n = coeffs[0] + coeffs[1] * u + coeffs[2] * v + coeffs[3] * u * u + coeffs[4] * u * v + coeffs[5] * v * v
    return (
        anchor
        + local_points[:, 0, None] * tangent_u
        + local_points[:, 1, None] * tangent_v
        + fitted_n[:, None] * normal
    ).astype(np.float32)


def clamp_to_constant_radius(local_points: np.ndarray, support_radius: float, eps: float = 1e-6) -> np.ndarray:
    out = local_points.astype(np.float32).copy()
    radial = np.linalg.norm(out, axis=1)
    outside = radial > support_radius
    out[outside] *= (support_radius / np.maximum(radial[outside], eps))[:, None]
    return out


def apply_analytic_gravity(
    local_points: np.ndarray,
    profile_heights: np.ndarray,
    radial_weight: np.ndarray,
    variant: PhysicsVariant,
    gravity_local: np.ndarray,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if variant.gravity_scale <= 0.0:
        return local_points.astype(np.float32).copy(), profile_heights.astype(np.float32).copy()

    tangent_gravity = np.array([gravity_local[0], gravity_local[2]], dtype=np.float32)
    norm = float(np.linalg.norm(tangent_gravity))
    if norm <= 1e-8:
        tangent_gravity = np.array([0.0, -1.0], dtype=np.float32)
    else:
        tangent_gravity /= norm

    profile_norm = profile_heights / max(float(profile_heights.max()), 1e-8)
    mobility = np.clip((1.0 - radial_weight * radial_weight) * (0.30 + 0.70 * profile_norm), 0.0, 1.0)
    sag = support_radius * variant.sag_gain * variant.gravity_scale * mobility
    out_points = local_points + sag[:, None] * tangent_gravity[None, :]
    out_points = clamp_to_constant_radius(out_points, support_radius)
    out_heights = profile_heights * (1.0 - variant.compression * mobility)
    return out_points.astype(np.float32), np.maximum(out_heights, 0.0).astype(np.float32)


def apply_mass_spring_gravity(
    local_points: np.ndarray,
    profile_heights: np.ndarray,
    radial_weight: np.ndarray,
    faces: np.ndarray,
    ring_index: np.ndarray,
    variant: PhysicsVariant,
    gravity_local: np.ndarray,
    support_radius: float,
    height: float,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    rest = np.column_stack([local_points[:, 0], profile_heights, local_points[:, 1]]).astype(np.float32)
    pin_mask = np.zeros(len(rest), dtype=bool)
    pin_mask[ring_index == radial_segments] = True
    frame = GrowthFrame(
        index=0,
        phase=variant.label,
        phase_slug=variant.method_id,
        growth_t=1.0,
        height=float(height),
        support_radius=float(support_radius),
        roundness=float(variant.roundness),
        lesion_blend=0.42,
        gravity_scale=float(variant.gravity_scale),
        shape_memory=float(variant.shape_memory),
        contact_adhesion=float(variant.contact_adhesion),
    )
    simulated = simulate_soft_body_surface(
        rest,
        faces,
        pin_mask,
        frame,
        gravity_local.astype(np.float32),
        support_radius=support_radius,
    )
    out_points = clamp_to_constant_radius(simulated[:, [0, 2]].astype(np.float32), support_radius)
    out_heights = np.maximum(simulated[:, 1], 0.0).astype(np.float32)
    return out_points, out_heights


def interpolated_lesion_colors(
    skin_colors: np.ndarray,
    heights: np.ndarray,
    radial_weight: np.ndarray,
) -> np.ndarray:
    height_norm = heights / max(float(np.max(heights)), 1e-8)
    center = np.clip(1.0 - radial_weight, 0.0, 1.0)
    warm_tint = np.array([10.0, 3.0, 0.0], dtype=np.float32)
    vascular_tint = np.array([5.0, -1.5, -2.0], dtype=np.float32)
    rgb = skin_colors.astype(np.float32)
    rgb = rgb * (0.985 + 0.045 * height_norm[:, None])
    rgb += warm_tint[None, :] * (0.45 * height_norm[:, None])
    rgb += vascular_tint[None, :] * (0.22 * center[:, None])
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def build_variant_mesh(
    variant: PhysicsVariant,
    base_xyz: np.ndarray,
    base_faces: np.ndarray,
    base_rgb: np.ndarray,
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    gravity_local: np.ndarray,
    support_radius: float,
    height: float,
    radial_segments: int,
    angular_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    local_points, profile_heights, radial_weight, ring_index, faces = lesion_template(
        support_radius,
        height,
        variant.roundness,
        radial_segments,
        angular_segments,
    )
    if variant.solver.startswith("analytic"):
        deformed_points, deformed_heights = apply_analytic_gravity(
            local_points,
            profile_heights,
            radial_weight,
            variant,
            gravity_local,
            support_radius,
        )
    else:
        deformed_points, deformed_heights = apply_mass_spring_gravity(
            local_points,
            profile_heights,
            radial_weight,
            faces,
            ring_index,
            variant,
            gravity_local,
            support_radius,
            height,
            radial_segments,
            angular_segments,
        )
        if variant.sag_gain > 0.0:
            deformed_points, deformed_heights = apply_analytic_gravity(
                deformed_points,
                deformed_heights,
                radial_weight,
                variant,
                gravity_local,
                support_radius,
            )

    skin_points = quadratic_skin_points(
        deformed_points,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        base_xyz,
        base_faces,
        support_radius,
    )
    _, skin_colors = sample_skin_and_color(
        deformed_points,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        base_xyz,
        base_faces,
        base_rgb,
        support_radius,
    )
    xyz = skin_points + deformed_heights[:, None] * normal
    rgb = interpolated_lesion_colors(skin_colors, deformed_heights, radial_weight)
    face_arr = remove_degenerate_faces(xyz.astype(np.float32), faces)

    tangent_gravity = np.array([gravity_local[0], gravity_local[2]], dtype=np.float32)
    norm = float(np.linalg.norm(tangent_gravity))
    if norm > 1e-8:
        tangent_gravity /= norm
    weights = np.maximum(deformed_heights, 1e-6)
    center_shift = float(np.average(deformed_points @ tangent_gravity, weights=weights)) if norm > 1e-8 else 0.0
    radial_extent = float(np.max(np.linalg.norm(deformed_points, axis=1)))
    peak_height = float(np.max(deformed_heights))
    mean_height = float(np.average(deformed_heights, weights=np.maximum(1.0 - radial_weight, 0.05)))
    triangles = xyz[face_arr]
    area = float(np.sum(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2.0))
    metrics = {
        "support_radius_m": float(support_radius),
        "radial_extent_m": radial_extent,
        "peak_height_m": peak_height,
        "mean_height_m": mean_height,
        "gravity_sag_m": center_shift,
        "surface_area_m2": area,
    }
    return xyz.astype(np.float32), face_arr.astype(np.int32), rgb, metrics


def make_patch_figure(
    local_xyz: np.ndarray,
    local_faces: np.ndarray,
    local_rgb: np.ndarray,
    title: str,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> go.Figure:
    fig = go.Figure(
        data=[
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
    )
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


def annotate_png(path: Path, label: str, metrics: dict[str, float]) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    text = f"{label} | peak {metrics['peak_height_m'] * 1000:.0f} mm | sag {metrics['gravity_sag_m'] * 1000:.1f} mm"
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.rounded_rectangle((18, 18, 36 + width, 36 + height), radius=7, fill=(255, 255, 255, 224))
    draw.text((27, 25), text, font=font, fill=(36, 39, 44, 255))
    image.save(path)


def render_method_gif(
    records: list[dict[str, object]],
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
    images = []
    with tempfile.TemporaryDirectory(prefix="physics_constant_radius_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for idx, record in enumerate(records):
            xyz, faces, rgb = read_colored_ply(DATASET_ROOT / str(record["mesh"]))
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
            fig = make_patch_figure(
                local_xyz,
                local_faces,
                local_rgb,
                str(record["label"]),
                half_width,
                half_height,
                depth_after,
            )
            png_path = tmp_dir / f"method_{idx:02d}.png"
            fig.write_image(png_path, scale=1)
            annotate_png(png_path, str(record["label"]), record["metrics"])
            images.append(imageio.imread(png_path))
    imageio.mimsave(gif_path, images, duration=1.15, loop=0)


def make_notebook_figure(
    records: list[dict[str, object]],
    anchor: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    half_width: float,
    half_height: float,
    depth_after: float,
) -> go.Figure:
    first = records[0]
    xyz, faces, rgb = read_colored_ply(DATASET_ROOT / str(first["mesh"]))
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
    frames = []
    for record in records:
        xyz, faces, rgb = read_colored_ply(DATASET_ROOT / str(record["mesh"]))
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
        frames.append(
            go.Frame(
                name=str(record["label"]),
                data=[
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
                ],
                traces=[0],
                layout=go.Layout(title_text=f"Constant-radius gravity comparison - {record['label']}"),
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
    fig = go.Figure(data=[trace], frames=frames)
    fig.update_layout(
        title=f"Constant-radius gravity comparison - {first['label']}",
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
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "len": 0.88,
                "steps": steps,
            }
        ],
    )
    return fig


def write_notebook(
    notebook_path: Path,
    records: list[dict[str, object]],
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
    figure = make_notebook_figure(records, anchor, normal, tangent_u, tangent_v, half_width, half_height, depth_after)
    payload = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    records_json = json.dumps(records, indent=2)
    metadata_json = json.dumps(metadata_record, indent=2)
    setup_code = f"""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData

ROOT_CANDIDATES = []
for parent in (Path.cwd(), *Path.cwd().parents):
    ROOT_CANDIDATES.append(parent / 'data' / 'synthetic' / 'physics_aug_constant_radius')
ROOT_CANDIDATES.append(Path.cwd())
ROOT = next((path for path in ROOT_CANDIDATES if (path / 'data' / 'manifest.json').exists()), ROOT_CANDIDATES[0])
RECORDS = {records_json}
METADATA = {metadata_json}
"""
    cells = [
        nbf.v4.new_markdown_cell("# Constant-radius gravity method comparison"),
        nbf.v4.new_markdown_cell(
            "This executed notebook compares constant-radius lesion meshes under different gravity/solver terms. Lesion colors are interpolated from the local skin surface."
        ),
        nbf.v4.new_code_cell(setup_code),
        nbf.v4.new_code_cell("fig"),
    ]
    cells[2]["execution_count"] = 1
    cells[2]["outputs"] = []
    cells[3]["execution_count"] = 2
    cells[3]["outputs"] = [
        nbf.v4.new_output(
            output_type="display_data",
            data={
                "application/vnd.plotly.v1+json": payload,
                "text/plain": "<Plotly Figure: constant-radius gravity comparison>",
            },
            metadata={},
        )
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    nbf.write(notebook, notebook_path)


def write_metrics_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_id",
        "label",
        "solver",
        "support_radius_m",
        "radial_extent_m",
        "peak_height_m",
        "mean_height_m",
        "gravity_sag_m",
        "surface_area_m2",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            variant = record["variant"]
            metrics = record["metrics"]
            row = {
                "method_id": variant["method_id"],
                "label": variant["label"],
                "solver": variant["solver"],
                "notes": variant["notes"],
            }
            row.update(metrics)
            writer.writerow(row)


def build_dataset(args: argparse.Namespace) -> None:
    scan_id = args.scan_id
    base_mesh_path = HSR_MESH_ROOT / f"{scan_id}_closed_textured_mesh.ply"
    if not base_mesh_path.exists():
        raise FileNotFoundError(f"Missing closed HSR mesh: {base_mesh_path}")

    data_root = DATASET_ROOT / "data"
    visualization_root = DATASET_ROOT / "visualizations"
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

    records: list[dict[str, object]] = []
    for variant in variant_plan():
        lesion_xyz, lesion_faces, lesion_rgb, metrics = build_variant_mesh(
            variant,
            base_xyz,
            base_faces,
            base_rgb,
            anchor,
            normal,
            tangent_u,
            tangent_v,
            gravity_local,
            support_radius=args.support_radius,
            height=args.height,
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
            args.support_radius,
            args.height,
        )
        stem = f"{scan_id}_constant_radius_{variant.method_id}"
        lesion_path = data_root / "lesion_meshes" / f"{stem}_lesion.ply"
        mesh_path = data_root / "meshes" / f"{stem}_hsr_lesion.ply"
        write_colored_ply(lesion_path, lesion_xyz, lesion_faces, lesion_rgb)
        write_colored_ply(mesh_path, combined_xyz, combined_faces, combined_rgb)
        record = {
            "scan_id": scan_id,
            "stem": stem,
            "method_id": variant.method_id,
            "label": variant.label,
            "mesh": str(mesh_path.relative_to(DATASET_ROOT)),
            "lesion_mesh": str(lesion_path.relative_to(DATASET_ROOT)),
            "variant": asdict(variant),
            "metrics": metrics,
        }
        records.append(record)
        print(
            f"{variant.method_id:28s} radius={metrics['support_radius_m']:.4f} "
            f"peak={metrics['peak_height_m']:.4f} sag={metrics['gravity_sag_m']:.4f}"
        )

    metadata_record: dict[str, object] = {
        "dataset": "physics_aug_constant_radius",
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
        },
        "constant_geometry": {
            "support_radius_m": args.support_radius,
            "nominal_height_m": args.height,
            "radial_segments": args.radial_segments,
            "angular_segments": args.angular_segments,
        },
        "coloring": {
            "method": "local_skin_barycentric_interpolation_with_subtle_warm_height_tint",
            "description": "Lesion vertex colors are sampled from nearby skin triangles and adjusted only slightly by local protrusion height.",
        },
        "variants": [record["variant"] for record in records],
        "metrics": [dict(method_id=record["method_id"], **record["metrics"]) for record in records],
    }
    metadata_path = data_root / "metadata" / f"{scan_id}_constant_radius_gravity_metadata.json"
    metadata_path.write_text(json.dumps(metadata_record, indent=2), encoding="utf-8")
    metrics_path = data_root / "metrics.csv"
    write_metrics_csv(metrics_path, records)

    manifest = {
        "dataset": "physics_aug_constant_radius",
        "scan_id": scan_id,
        "metadata": str(metadata_path.relative_to(DATASET_ROOT)),
        "metrics": str(metrics_path.relative_to(DATASET_ROOT)),
        "gif": f"visualizations/gifs/{scan_id}_constant_radius_gravity_methods.gif",
        "notebook": "visualizations/plotly/constant_radius_gravity_comparison.ipynb",
        "records": records,
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (visualization_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    depth_after = max(0.090, args.height + 0.040)
    gif_path = visualization_root / "gifs" / f"{scan_id}_constant_radius_gravity_methods.gif"
    render_method_gif(
        records,
        gif_path,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width=args.view_half_width,
        half_height=args.view_half_height,
        depth_after=depth_after,
    )
    notebook_path = visualization_root / "plotly" / "constant_radius_gravity_comparison.ipynb"
    write_notebook(
        notebook_path,
        records,
        metadata_record,
        anchor,
        normal,
        tangent_u,
        tangent_v,
        half_width=args.view_half_width,
        half_height=args.view_half_height,
        depth_after=depth_after,
    )
    print(gif_path)
    print(notebook_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", default="HSR0018-Body-070")
    parser.add_argument("--target-x", type=float, default=-0.09)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=1.09)
    parser.add_argument("--target-window", type=float, default=0.040)
    parser.add_argument("--support-radius", type=float, default=0.046)
    parser.add_argument("--height", type=float, default=0.050)
    parser.add_argument("--radial-segments", type=int, default=34)
    parser.add_argument("--angular-segments", type=int, default=112)
    parser.add_argument("--view-half-width", type=float, default=0.105)
    parser.add_argument("--view-half-height", type=float, default=0.115)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()

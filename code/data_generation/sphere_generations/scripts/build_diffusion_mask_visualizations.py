#!/usr/bin/env python3
"""Build visualizations for sphere diffusion inpaint masks."""

from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import nbformat as nbf
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder


ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "data" / "synthetic" / "single_lesion" / "body_parts" / "sphere_generations_textured_diffusion"
DATA_ROOT = DATASET_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "inpaint_inputs"
MASK_ROOT = DATA_ROOT / "inpaint_masks"
OBJ_ROOT = DATA_ROOT / "objs"
TEXTURE_ROOT = DATA_ROOT / "textures"
VIS_ROOT = ROOT / "data" / "synthetic" / "single_lesion" / "visualization" / "sphere_generations_textured_diffusion" / "inpaint_masks"
PREVIEW_ROOT = VIS_ROOT / "previews"
GIF_ROOT = VIS_ROOT / "gifs"
PLOTLY_ROOT = VIS_ROOT / "plotly"


def root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mask_items() -> list[dict[str, Path | str]]:
    rows = []
    for mask_path in sorted(MASK_ROOT.glob("*_inpaint_mask.png")):
        stem = mask_path.name.removesuffix("_inpaint_mask.png")
        input_path = INPUT_ROOT / f"{stem}_inpaint_init.png"
        texture_path = TEXTURE_ROOT / f"{stem}.png"
        obj_path = OBJ_ROOT / f"{stem}.obj"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing inpaint input for {mask_path.name}: {input_path}")
        if not texture_path.exists():
            raise FileNotFoundError(f"Missing generated texture for {mask_path.name}: {texture_path}")
        if not obj_path.exists():
            raise FileNotFoundError(f"Missing textured OBJ for {mask_path.name}: {obj_path}")
        rows.append({"stem": stem, "mask": mask_path, "input": input_path, "texture": texture_path, "obj": obj_path})
    if not rows:
        raise FileNotFoundError(f"No inpaint masks found in {MASK_ROOT}")
    return rows


def read_obj_uv_triangles(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    uv = []
    faces = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("vt "):
                parts = line.split()
                uv.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:4]:
                    pieces = token.split("/")
                    texture_idx = int(pieces[1]) - 1 if len(pieces) > 1 and pieces[1] else int(pieces[0]) - 1
                    face.append(texture_idx)
                faces.append(face)
    if not uv or not faces:
        raise ValueError(f"OBJ does not contain UV triangle data: {obj_path}")
    return np.asarray(uv, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def uv_mesh_image(obj_path: Path, size: int = 512) -> Image.Image:
    uv, faces = read_obj_uv_triangles(obj_path)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for face in faces:
        points = []
        for idx in face:
            u, v = uv[idx]
            x = float(np.clip(u, 0.0, 1.0)) * (size - 1)
            y = (1.0 - float(np.clip(v, 0.0, 1.0))) * (size - 1)
            points.append((x, y))
        draw.line([points[0], points[1], points[2], points[0]], fill=(20, 92, 170), width=1)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(30, 35, 45), width=2)
    return image


def labeled_tile(image: Image.Image, label: str, size: int = 192) -> Image.Image:
    tile = Image.new("RGB", (size, size + 28), "white")
    thumb = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    tile.paste(thumb, (0, 28))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 7), label[:34], fill=(20, 30, 45))
    return tile


def build_contact_sheet(rows: list[dict[str, Path | str]]) -> Path:
    tile_size = 192
    gap = 10
    columns = 3
    width = columns * tile_size + (columns + 1) * gap
    height = len(rows) * (tile_size + 28) + (len(rows) + 1) * gap
    sheet = Image.new("RGB", (width, height), (246, 248, 251))

    for row_idx, row in enumerate(rows):
        texture_image = Image.open(row["texture"]).convert("RGB")
        mask_image = Image.open(row["mask"]).convert("L")
        mask_rgb = Image.merge("RGB", (mask_image, mask_image, mask_image))
        uv_image = uv_mesh_image(Path(row["obj"]))
        tiles = [
            labeled_tile(texture_image, "2D diffusion texture", tile_size),
            labeled_tile(mask_rgb, "2D binary mask", tile_size),
            labeled_tile(uv_image, f"2D UV triangles: {row['stem']}", tile_size),
        ]
        y = gap + row_idx * (tile_size + 28 + gap)
        for col_idx, tile in enumerate(tiles):
            x = gap + col_idx * (tile_size + gap)
            sheet.paste(tile, (x, y))

    out_path = PREVIEW_ROOT / "sphere_diffusion_texture_mask_uv_contact_sheet.png"
    sheet.save(out_path)
    return out_path


def build_gif(rows: list[dict[str, Path | str]]) -> Path:
    frames = []
    for row in rows:
        texture_image = Image.open(row["texture"]).convert("RGB")
        mask_image = Image.open(row["mask"]).convert("L")
        mask_rgb = Image.merge("RGB", (mask_image, mask_image, mask_image))
        uv_image = uv_mesh_image(Path(row["obj"]))

        frame = Image.new("RGB", (384 * 3, 430), (246, 248, 251))
        tiles = [
            labeled_tile(texture_image, "2D diffusion texture", 384),
            labeled_tile(mask_rgb, "2D binary mask", 384),
            labeled_tile(uv_image, f"2D UV triangles: {row['stem']}", 384),
        ]
        for idx, tile in enumerate(tiles):
            frame.paste(tile, (idx * 384, 0))
        frames.append(np.asarray(frame))

    out_path = GIF_ROOT / "sphere_diffusion_texture_mask_uv_layouts.gif"
    imageio.mimsave(out_path, frames, duration=0.9, loop=0)
    return out_path


def image_to_data_uri(path: Path) -> str:
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_sample_figure(row: dict[str, Path | str]) -> go.Figure:
    texture_path = Path(row["texture"])
    mask_path = Path(row["mask"])
    obj_path = Path(row["obj"])
    uv, faces = read_obj_uv_triangles(obj_path)

    x_lines = []
    y_lines = []
    for face in faces:
        tri = uv[face]
        x_lines.extend([float(tri[0, 0]), float(tri[1, 0]), float(tri[2, 0]), float(tri[0, 0]), None])
        y_lines.extend([float(tri[0, 1]), float(tri[1, 1]), float(tri[2, 1]), float(tri[0, 1]), None])

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("2D diffusion texture", "2D binary mask", "2D mesh triangles in UV space"),
        horizontal_spacing=0.045,
    )
    fig.add_trace(go.Image(source=image_to_data_uri(texture_path), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Image(source=image_to_data_uri(mask_path), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color="rgb(20,92,170)", width=0.7),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_layout(
        title=str(row["stem"]),
        width=1200,
        height=430,
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="white",
    )
    fig.update_xaxes(visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(visible=False, constrain="domain", row=1, col=2)
    fig.update_yaxes(visible=False, scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(range=[0, 1], visible=True, title="u", constrain="domain", row=1, col=3)
    fig.update_yaxes(range=[0, 1], visible=True, title="v", scaleanchor="x3", scaleratio=1, row=1, col=3)
    return fig


def build_plotly_notebook(rows: list[dict[str, Path | str]], contact_sheet_path: Path) -> Path:
    records = [
        {
            "stem": str(row["stem"]),
            "input": root_relative(Path(row["input"])),
            "mask": root_relative(Path(row["mask"])),
            "texture": root_relative(Path(row["texture"])),
            "obj": root_relative(Path(row["obj"])),
        }
        for row in rows
    ]

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        nbf.v4.new_markdown_cell("# Sphere diffusion 2D texture, mask, and UV triangle layouts"),
        nbf.v4.new_code_cell("records = " + json.dumps(records, indent=2) + "\nrecords"),
    ]
    nb.cells[1]["execution_count"] = 1
    nb.cells[1]["outputs"] = [
        nbf.v4.new_output(
            output_type="execute_result",
            data={"text/plain": repr(records)},
            execution_count=1,
            metadata={},
        )
    ]
    for idx, row in enumerate(rows, start=2):
        fig = make_sample_figure(row)
        payload = json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))
        cell = nbf.v4.new_code_cell(f"# {row['stem']}\nfig")
        cell["execution_count"] = idx
        cell["outputs"] = [
            nbf.v4.new_output(
                output_type="display_data",
                data={
                    "application/vnd.plotly.v1+json": payload,
                    "text/plain": f"<Plotly Figure: {row['stem']}>",
                },
                metadata={},
            )
        ]
        nb.cells.append(cell)

    out_path = PLOTLY_ROOT / "sphere_diffusion_texture_mask_uv_viewer.ipynb"
    nbf.write(nb, out_path)
    return out_path


def main() -> None:
    if VIS_ROOT.exists():
        shutil.rmtree(VIS_ROOT)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    GIF_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTLY_ROOT.mkdir(parents=True, exist_ok=True)

    rows = mask_items()
    contact_sheet_path = build_contact_sheet(rows)
    gif_path = build_gif(rows)
    notebook_path = build_plotly_notebook(rows, contact_sheet_path)
    manifest = {
        "records": [
            {
                "stem": str(row["stem"]),
                "input": root_relative(Path(row["input"])),
                "mask": root_relative(Path(row["mask"])),
                "texture": root_relative(Path(row["texture"])),
                "obj": root_relative(Path(row["obj"])),
            }
            for row in rows
        ],
        "contact_sheet": root_relative(contact_sheet_path),
        "gif": root_relative(gif_path),
        "notebook": root_relative(notebook_path),
    }
    manifest_path = VIS_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(contact_sheet_path)
    print(gif_path)
    print(notebook_path)
    print(manifest_path)


if __name__ == "__main__":
    main()

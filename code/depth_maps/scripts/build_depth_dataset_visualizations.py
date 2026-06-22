from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat
import numpy as np
from nbclient import NotebookClient
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]


def resolve_dataset_root(dataset_root: str) -> Path:
    path = Path(dataset_root)
    return path if path.is_absolute() else ROOT / path


def find_manifest(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "data" / "manifest.csv",
        dataset_root / "manifest.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing manifest below {dataset_root}")


def load_manifest(dataset_root: Path) -> tuple[Path, list[dict[str, str]]]:
    manifest_path = find_manifest(dataset_root)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return manifest_path, list(csv.DictReader(handle))


def find_summary(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "data" / "summary.json",
        dataset_root / "summary.json",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def resolve_row_path(dataset_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    candidates = [
        dataset_root / path,
        dataset_root / "data" / path,
        dataset_root.parent / path,
    ]

    parts = path.parts
    if parts[:1] == (dataset_root.name,):
        candidates.append(dataset_root.joinpath(*parts[1:]))
        if len(parts) > 2 and parts[1] in {"images", "depth", "depth_vis", "metadata"}:
            candidates.append(dataset_root / "images" / Path(*parts[1:]))
    elif parts[:1] in {("images",), ("depth",), ("depth_vis",), ("metadata",)}:
        candidates.append(dataset_root / "images" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def evenly_spaced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indices]


def default_visualization_root(dataset_root: Path) -> Path:
    return dataset_root / "visualizations"


def save_preview_gif(
    dataset_root: Path,
    visualization_root: Path,
    rows: list[dict[str, str]],
    frame_count: int,
    tile_size: int,
) -> Path:
    output_dir = visualization_root / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_root.name}_rgb_depth_preview.gif"

    frames = []
    for row in evenly_spaced_rows(rows, frame_count):
        rgb = (
            Image.open(resolve_row_path(dataset_root, row["image_path"]))
            .convert("RGB")
            .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        )
        depth = (
            Image.open(resolve_row_path(dataset_root, row["depth_vis_path"]))
            .convert("L")
            .resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            .convert("RGB")
        )
        frame = Image.new("RGB", (tile_size * 2, tile_size), "white")
        frame.paste(rgb, (0, 0))
        frame.paste(depth, (tile_size, 0))
        frames.append(np.asarray(frame))

    imageio.mimsave(output_path, frames, duration=0.18, loop=0)
    return output_path


def notebook_source(dataset_root: Path) -> list[nbformat.NotebookNode]:
    dataset_root_text = str(dataset_root)
    return [
        nbformat.v4.new_markdown_cell(
            f"# {dataset_root.name} depth surfaces\n\n"
            "Executed Plotly notebook for inspecting RGB/depth pairs as interactive 3D surfaces."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv, json\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"DATASET_ROOT = Path({dataset_root_text!r})\n"
            "MANIFEST = next(path for path in [DATASET_ROOT / 'data' / 'manifest.csv', DATASET_ROOT / 'manifest.csv'] if path.exists())\n"
            "SUMMARY = next((path for path in [DATASET_ROOT / 'data' / 'summary.json', DATASET_ROOT / 'summary.json'] if path.exists()), None)\n"
            "rows = list(csv.DictReader(MANIFEST.open(newline='', encoding='utf-8')))\n"
            "summary = json.loads(SUMMARY.read_text(encoding='utf-8')) if SUMMARY else {'sample_count': len(rows)}\n"
            "def resolve_row_path(path_value):\n"
            "    path = Path(path_value)\n"
            "    candidates = [DATASET_ROOT / path, DATASET_ROOT / 'data' / path, DATASET_ROOT.parent / path]\n"
            "    parts = path.parts\n"
            "    if parts[:1] == (DATASET_ROOT.name,):\n"
            "        candidates.append(DATASET_ROOT.joinpath(*parts[1:]))\n"
            "        if len(parts) > 2 and parts[1] in {'images', 'depth', 'depth_vis', 'metadata'}:\n"
            "            candidates.append(DATASET_ROOT / 'images' / Path(*parts[1:]))\n"
            "    elif parts[:1] in {('images',), ('depth',), ('depth_vis',), ('metadata',)}:\n"
            "        candidates.append(DATASET_ROOT / 'images' / path)\n"
            "    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])\n"
            "summary['sample_count'], rows[0]['sample_id'], rows[-1]['sample_id']"
        ),
        nbformat.v4.new_code_cell(
            "row = rows[len(rows) // 2]\n"
            "depth = np.load(resolve_row_path(row['depth_npy_path'])).astype(float)\n"
            "rgb = np.array(Image.open(resolve_row_path(row['image_path'])).convert('RGB'))\n\n"
            "stride = 4\n"
            "z = depth[::stride, ::stride]\n"
            "z[~np.isfinite(z) | (z <= 0)] = np.nan\n"
            "yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]\n\n"
            "fig = make_subplots(\n"
            "    rows=1,\n"
            "    cols=2,\n"
            "    specs=[[{'type': 'surface'}, {'type': 'xy'}]],\n"
            "    subplot_titles=(f\"{row['sample_id']} GT depth surface\", 'RGB render'),\n"
            "    column_widths=[0.62, 0.38],\n"
            ")\n"
            "fig.add_trace(\n"
            "    go.Surface(\n"
            "        x=xx,\n"
            "        y=-yy,\n"
            "        z=z,\n"
            "        surfacecolor=z,\n"
            "        colorscale='Viridis',\n"
            "        colorbar={'title': 'meters'},\n"
            "        connectgaps=False,\n"
            "    ),\n"
            "    row=1,\n"
            "    col=1,\n"
            ")\n"
            "fig.add_trace(go.Image(z=rgb), row=1, col=2)\n"
            "fig.update_layout(\n"
            "    width=1050,\n"
            "    height=560,\n"
            "    title=f\"{DATASET_ROOT.name}: interactive RGB / GT depth pair\",\n"
            "    scene={\n"
            "        'xaxis_title': 'x pixels / stride',\n"
            "        'yaxis_title': 'y pixels / stride',\n"
            "        'zaxis_title': 'camera z distance (m)',\n"
            "        'aspectmode': 'data',\n"
            "    },\n"
            ")\n"
            "fig.update_xaxes(showticklabels=False, row=1, col=2)\n"
            "fig.update_yaxes(showticklabels=False, row=1, col=2)\n"
            "fig"
        ),
    ]


def save_plotly_notebook(dataset_root: Path, visualization_root: Path) -> Path:
    output_dir = visualization_root / "plotly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_root.name}_depth_surfaces.ipynb"

    notebook = nbformat.v4.new_notebook(cells=notebook_source(dataset_root))
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    NotebookClient(notebook, timeout=180, kernel_name="python3").execute()
    nbformat.write(notebook, output_path)
    return output_path


def build_visualizations(dataset_root: Path, frame_count: int, tile_size: int, visualization_root: Path | None = None) -> dict[str, Any]:
    visualization_root = visualization_root or default_visualization_root(dataset_root)
    manifest_path, rows = load_manifest(dataset_root)
    summary_path = find_summary(dataset_root)
    gif_path = save_preview_gif(dataset_root, visualization_root, rows, frame_count, tile_size)
    notebook_path = save_plotly_notebook(dataset_root, visualization_root)
    result: dict[str, Any] = {
        "dataset_root": str(dataset_root.relative_to(ROOT)),
        "visualization_root": str(visualization_root.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "sample_count": str(len(rows)),
        "gif": str(gif_path.relative_to(ROOT)),
        "notebook": str(notebook_path.relative_to(ROOT)),
    }
    if summary_path is not None:
        result["summary"] = str(summary_path.relative_to(ROOT))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GIF and executed Plotly notebook visualizations for a depth dataset.")
    parser.add_argument("dataset_root", nargs="+", help="Dataset root containing data/manifest.csv.")
    parser.add_argument(
        "--visualization-root",
        nargs="*",
        default=None,
        help="Optional visualization root for each dataset root. Defaults to <dataset_root>/visualizations.",
    )
    parser.add_argument("--frame_count", type=int, default=32)
    parser.add_argument("--tile_size", type=int, default=192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.visualization_root is None:
        visualization_roots = [None] * len(args.dataset_root)
    else:
        if len(args.visualization_root) != len(args.dataset_root):
            raise ValueError("--visualization-root must be provided once per dataset_root")
        visualization_roots = [resolve_dataset_root(path) for path in args.visualization_root]
    results = []
    for dataset_root, visualization_root in zip(args.dataset_root, visualization_roots):
        results.append(
            build_visualizations(resolve_dataset_root(dataset_root), args.frame_count, args.tile_size, visualization_root)
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

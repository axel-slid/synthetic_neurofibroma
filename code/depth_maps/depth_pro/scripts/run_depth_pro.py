#!/usr/bin/env python3
"""Run Apple Depth Pro on one image and save Plotly depth visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from PIL import Image
from transformers import pipeline


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "predictions" / "depth_pro_single"
MODEL_ID = "apple/DepthPro-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input RGB image.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Deprecated alias for --output-root.")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def predict_depth(image: Image.Image, device: str) -> np.ndarray:
    device_arg = 0 if device.startswith("cuda") and torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=MODEL_ID, device=device_arg)
    with torch.inference_mode():
        depth = pipe(image)["predicted_depth"]
    if isinstance(depth, torch.Tensor):
        return depth.detach().float().cpu().numpy()
    return np.asarray(depth, dtype=np.float32)


def make_surface_figure(depth: np.ndarray) -> go.Figure:
    stride = max(1, max(depth.shape) // 120)
    z = depth[::stride, ::stride].astype(np.float32)
    h, w = z.shape
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    z = z - np.nanmedian(z)
    scale = np.nanpercentile(np.abs(z), 95) or 1.0
    z = np.clip(z / scale, -1.5, 1.5)
    surface = go.Surface(
        x=x,
        y=y,
        z=-z,
        surfacecolor=-z,
        colorscale="Viridis",
        showscale=False,
        hoverinfo="skip",
    )
    fig = go.Figure(surface)
    fig.update_layout(
        title="Depth Pro surface",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
        margin=dict(l=0, r=0, t=42, b=0),
    )
    return fig


def main() -> None:
    args = parse_args()
    output_root = (args.output_dir or args.output_root).resolve()
    data_root = output_root / "data"
    visualizations_root = output_root / "visualizations" / "plotly"
    data_root.mkdir(parents=True, exist_ok=True)
    visualizations_root.mkdir(parents=True, exist_ok=True)

    image = load_image(args.image, args.max_side)
    depth = predict_depth(image, args.device)

    stem = args.image.stem
    depth_npy = data_root / f"{stem}_depthpro_depth.npy"
    plotly_json = visualizations_root / f"{stem}_depthpro_surface.plotly.json"
    metadata_path = data_root / f"{stem}_depthpro_metadata.json"
    np.save(depth_npy, depth.astype(np.float32))
    fig = make_surface_figure(depth)
    plotly_json.write_text(pio.to_json(fig), encoding="utf-8")
    metadata = {
        "model": MODEL_ID,
        "input": str(args.image),
        "resized_width": image.width,
        "resized_height": image.height,
        "depth_npy": str(depth_npy),
        "plotly_json": str(plotly_json),
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "depth_median": float(np.nanmedian(depth)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "input": str(args.image),
                "data": str(data_root),
                "visualizations": str(visualizations_root.parent),
                "outputs": {
                    "depth_npy": str(depth_npy),
                    "plotly_json": str(plotly_json),
                    "metadata": str(metadata_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {depth_npy}")
    print(f"wrote {plotly_json}")
    print(f"wrote {metadata_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

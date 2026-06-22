from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
DEPTH_ROOT = ROOT / "data" / "depth_maps"
BASE_ROOT = DEPTH_ROOT / "base"


def resolve_depth_path(relative_path: str) -> Path:
    path = Path(relative_path)
    candidates = [
        BASE_ROOT / path,
        DEPTH_ROOT / path,
    ]
    parts = path.parts
    if parts[:1] == ("base",):
        candidates.append(BASE_ROOT.joinpath(*parts[1:]))
        if len(parts) > 2 and parts[1] in {"images", "depth", "depth_vis", "metadata"}:
            candidates.append(BASE_ROOT / "images" / Path(*parts[1:]))
    elif parts[:1] in {("images",), ("depth",), ("depth_vis",), ("metadata",)}:
        candidates.append(BASE_ROOT / "images" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def main() -> None:
    manifest_path = BASE_ROOT / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        sample_id = row["sample_id"]
        depth = np.load(resolve_depth_path(row["depth_npy_path"]))
        rgb = imageio.imread(resolve_depth_path(row["image_path"]))
        depth_vis = near_bright_depth_visual(depth)
        imageio.imwrite(resolve_depth_path(row["depth_vis_path"]), depth_vis)
        save_comparison_plot(rgb, depth_vis, resolve_depth_path(row["plot_path"]))
        print(f"updated {sample_id}")

    print(f"updated {len(rows)} depth visualizations and plots")


if __name__ == "__main__":
    main()

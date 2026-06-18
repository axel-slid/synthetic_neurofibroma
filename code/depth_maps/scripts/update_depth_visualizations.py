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


def save_comparison_plot(rgb: np.ndarray, depth_vis: np.ndarray, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("2D render")
    axes[0].axis("off")
    axes[1].imshow(depth_vis, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("depth: near bright, far dark")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    manifest_path = DEPTH_ROOT / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        sample_id = row["sample_id"]
        depth = np.load(DEPTH_ROOT / row["depth_npy_path"])
        rgb = imageio.imread(DEPTH_ROOT / row["image_path"])
        depth_vis = near_bright_depth_visual(depth)
        imageio.imwrite(DEPTH_ROOT / row["depth_vis_path"], depth_vis)
        save_comparison_plot(rgb, depth_vis, sample_id, DEPTH_ROOT / row["plot_path"])
        print(f"updated {sample_id}")

    print(f"updated {len(rows)} depth visualizations and plots")


if __name__ == "__main__":
    main()

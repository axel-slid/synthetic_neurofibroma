#!/usr/bin/env python3
"""Fine-tune Depth Pro's fusion/depth head on local RGB/depth-map pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat
import numpy as np
import torch
import torch.nn.functional as F
from nbclient import NotebookClient
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import DepthProForDepthEstimation, DepthProImageProcessor


ROOT = Path(__file__).resolve().parents[4]
DEPTH_MAPS_ROOT = ROOT / "data" / "depth_maps"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "predictions" / "depth_pro_finetuned_synthetic"
MODEL_ID = "apple/DepthPro-hf"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source: str
    body_part: str
    image_path: Path
    depth_path: Path
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        default=[
            DEPTH_MAPS_ROOT / "synthetic" / "manifest.csv",
            DEPTH_MAPS_ROOT / "base" / "manifest.csv",
        ],
        help="Depth-map manifests to train on, or train/evaluate on when --val-manifests is omitted.",
    )
    parser.add_argument(
        "--val-manifests",
        type=Path,
        nargs="+",
        default=None,
        help="Optional fixed validation manifests. When set, --manifests are used only for training.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=50.0,
        help="Weight for edge-aware prediction smoothness regularization. Use 0 to disable.",
    )
    parser.add_argument(
        "--smoothness-edge-weight",
        type=float,
        default=8.0,
        help="RGB edge sensitivity for the smoothness term; higher values preserve image edges more strongly.",
    )
    parser.add_argument(
        "--smoothness-curvature-weight",
        type=float,
        default=0.5,
        help="Relative weight of second-order depth curvature inside the smoothness term.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to the CUDA device with the most free memory.")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trainable", choices=["head", "fusion_head"], default="fusion_head")
    parser.add_argument(
        "--target-space",
        choices=["metric_depth", "inverse_depth"],
        default="metric_depth",
        help="Train against camera z distance or Depth Pro's natural inverse-depth polarity.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--min-valid-fraction", type=float, default=0.001)
    parser.add_argument("--viz-count", type=int, default=36)
    parser.add_argument("--eval-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return torch.device(f"cuda:{best_idx}")


def amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or name == "fp32":
        return None
    if name == "bf16" and torch.cuda.is_bf16_supported(device):
        return torch.bfloat16
    if name == "bf16":
        return torch.float16
    return torch.float16


def resolve_manifest_path(manifest_path: Path, relative_path: str) -> Path:
    if not relative_path:
        raise FileNotFoundError(f"Empty manifest path value in {manifest_path}")
    root = manifest_path.parent
    path = Path(relative_path)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Could not resolve {relative_path!r} from {manifest_path}")

    candidates = [
        root / path,
        DEPTH_MAPS_ROOT / path,
        DEPTH_MAPS_ROOT / "body_parts" / path,
        ROOT / path,
    ]

    parts = path.parts
    body_parts_root = next((parent for parent in [root, *root.parents] if parent.name == "body_parts"), None)
    if body_parts_root is not None:
        candidates.append(body_parts_root / path)

    if root.name == "base":
        if parts[:1] == ("base",):
            candidates.append(root.joinpath(*parts[1:]))
            if len(parts) > 2 and parts[1] in {"images", "depth", "depth_vis", "metadata"}:
                candidates.append(root / "images" / Path(*parts[1:]))
        elif parts[:1] in {("images",), ("depth",), ("depth_vis",), ("metadata",)}:
            candidates.append(root / "images" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {relative_path!r} from {manifest_path}")


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_samples(manifest_paths: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source = row.get("source_folder") or row.get("body_part") or manifest_path.parent.name
                body_part = row.get("body_part") or source
                sample_id = row.get("sample_id") or row.get("pair_id")
                if not sample_id:
                    raise ValueError(f"Manifest row is missing sample_id/pair_id in {manifest_path}: {row}")
                depth_path_value = row.get("depth_npy_path") or row.get("depth_path")
                if not depth_path_value:
                    raise ValueError(f"Manifest row is missing depth_npy_path/depth_path in {manifest_path}: {row}")
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        source=source,
                        body_part=body_part,
                        image_path=resolve_manifest_path(manifest_path, row["image_path"]),
                        depth_path=resolve_manifest_path(manifest_path, depth_path_value),
                        width=int(float(row.get("width", 256))),
                        height=int(float(row.get("height", 256))),
                    )
                )
    if not samples:
        raise ValueError("No samples found in manifests")
    return samples


def valid_depth_fraction(depth_path: Path) -> float:
    depth = np.load(depth_path, mmap_mode="r")
    valid = np.isfinite(depth) & (depth > 0.0)
    return float(valid.mean())


def filter_valid_samples(samples: list[Sample], min_valid_fraction: float) -> tuple[list[Sample], list[dict[str, Any]]]:
    kept: list[Sample] = []
    skipped: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="checking valid depth", leave=False):
        fraction = valid_depth_fraction(sample.depth_path)
        if fraction >= min_valid_fraction:
            kept.append(sample)
        else:
            skipped.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "depth_path": repo_relative(sample.depth_path),
                    "valid_fraction": fraction,
                }
            )
    return kept, skipped


def split_samples(
    samples: list[Sample],
    val_fraction: float,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
    if max_val_samples is not None:
        val_samples = val_samples[:max_val_samples]
    return train_samples, val_samples


class DepthMapDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[Sample], train: bool, flip_probability: float = 0.5) -> None:
        self.samples = samples
        self.train = train
        self.flip_probability = flip_probability

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        depth = np.load(sample.depth_path).astype(np.float32)
        if self.train and random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = np.ascontiguousarray(np.fliplr(depth))
        return {
            "sample": sample,
            "image": image,
            "depth": depth,
        }


def target_from_metric_depth(depth: torch.Tensor, mask: torch.Tensor, target_space: str) -> torch.Tensor:
    if target_space == "metric_depth":
        return depth
    if target_space == "inverse_depth":
        target = torch.zeros_like(depth)
        target[mask] = 1.0 / depth.clamp_min(1e-3)[mask]
        return target
    raise ValueError(f"Unsupported target space: {target_space}")


def prediction_to_metric_depth(prediction: np.ndarray, target_space: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    if target_space == "metric_depth":
        return prediction
    if target_space == "inverse_depth":
        valid = np.isfinite(prediction) & (prediction > 1e-6)
        metric = np.zeros(prediction.shape, dtype=np.float32)
        metric[valid] = 1.0 / prediction[valid]
        return metric
    raise ValueError(f"Unsupported target space: {target_space}")


def make_collate(processor: DepthProImageProcessor, target_space: str = "metric_depth"):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        metric_depths = [torch.from_numpy(item["depth"]) for item in batch]
        rgb_tensors: list[torch.Tensor] = []
        for image, depth in zip(images, metric_depths, strict=True):
            height, width = depth.shape
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            rgb = np.asarray(resized, dtype=np.float32) / 255.0
            rgb_tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))
        processed = processor(images=images, return_tensors="pt")
        metric_depth = torch.stack(metric_depths, dim=0).float()
        mask = torch.isfinite(metric_depth) & (metric_depth > 0.0)
        metric_depth = torch.nan_to_num(metric_depth, nan=0.0, posinf=0.0, neginf=0.0)
        target = target_from_metric_depth(metric_depth, mask, target_space)
        return {
            "pixel_values": processed["pixel_values"],
            "depth": target,
            "metric_depth": metric_depth,
            "mask": mask,
            "rgb": torch.stack(rgb_tensors, dim=0).float(),
            "samples": [item["sample"] for item in batch],
        }

    return collate


def set_trainable_modules(model: DepthProForDepthEstimation, trainable: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    # DepthPro-hf ships fp16 weights. Keep the frozen encoder in fp16 for memory,
    # but optimize the small adapter modules in fp32 to avoid half-precision AdamW
    # updates producing NaNs.
    model.fusion_stage.float()
    model.head.float()

    modules: list[torch.nn.Module] = [model.head]
    if trainable == "fusion_head":
        modules.insert(0, model.fusion_stage)

    trainable_parameters: list[torch.nn.Parameter] = []
    for module in modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True
            trainable_parameters.append(parameter)

    model.depth_pro.eval()
    if getattr(model, "fov_model", None) is not None:
        model.fov_model.eval()
    return trainable_parameters


def adapter_state(model: DepthProForDepthEstimation, trainable: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "trainable": trainable,
        "head": {key: value.detach().cpu() for key, value in model.head.state_dict().items()},
    }
    if trainable == "fusion_head":
        state["fusion_stage"] = {key: value.detach().cpu() for key, value in model.fusion_stage.state_dict().items()}
    return state


def load_adapter_state(model: DepthProForDepthEstimation, state: dict[str, Any]) -> None:
    if "fusion_stage" in state:
        model.fusion_stage.load_state_dict(state["fusion_stage"])
    model.head.load_state_dict(state["head"])


def predict_depth(
    model: DepthProForDepthEstimation,
    pixel_values: torch.Tensor,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    device_type = pixel_values.device.type
    autocast_enabled = autocast_dtype is not None
    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_enabled):
        with torch.no_grad():
            depth_pro_outputs = model.depth_pro(
                pixel_values=pixel_values,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            fusion_dtype = next(model.fusion_stage.parameters()).dtype
            features = [feature.detach().to(dtype=fusion_dtype) for feature in depth_pro_outputs.features]
        fused_hidden_states = model.fusion_stage(features)
        predicted_depth = model.head(fused_hidden_states[-1])
    return predicted_depth.float()


def downsample_prediction(predicted_depth: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        predicted_depth.unsqueeze(1),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
    valid = valid & torch.isfinite(values)
    if int(valid.sum().detach().cpu()) == 0:
        return None
    return values[valid].mean()


def smoothness_loss(
    prediction: torch.Tensor,
    mask: torch.Tensor,
    rgb: torch.Tensor | None,
    edge_weight: float,
    curvature_weight: float,
) -> torch.Tensor:
    if prediction.ndim != 3:
        raise ValueError(f"Expected BxHxW prediction, got {tuple(prediction.shape)}")
    log_prediction = prediction.clamp_min(1e-3).log()

    dx = (log_prediction[:, :, 1:] - log_prediction[:, :, :-1]).abs()
    dy = (log_prediction[:, 1:, :] - log_prediction[:, :-1, :]).abs()
    valid_x = mask[:, :, 1:] & mask[:, :, :-1]
    valid_y = mask[:, 1:, :] & mask[:, :-1, :]

    weight_x: torch.Tensor | float = 1.0
    weight_y: torch.Tensor | float = 1.0
    if rgb is not None:
        rgb = rgb.to(dtype=log_prediction.dtype)
        weight_x = torch.exp(-edge_weight * (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=1))
        weight_y = torch.exp(-edge_weight * (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=1))
        dx = dx * weight_x.detach()
        dy = dy * weight_y.detach()

    terms = [term for term in [_masked_mean(dx, valid_x), _masked_mean(dy, valid_y)] if term is not None]

    if curvature_weight > 0.0:
        curvature_x = (log_prediction[:, :, 2:] - 2.0 * log_prediction[:, :, 1:-1] + log_prediction[:, :, :-2]).abs()
        curvature_y = (log_prediction[:, 2:, :] - 2.0 * log_prediction[:, 1:-1, :] + log_prediction[:, :-2, :]).abs()
        valid_curvature_x = mask[:, :, 2:] & mask[:, :, 1:-1] & mask[:, :, :-2]
        valid_curvature_y = mask[:, 2:, :] & mask[:, 1:-1, :] & mask[:, :-2, :]
        if isinstance(weight_x, torch.Tensor):
            curvature_x = curvature_x * (0.5 * (weight_x[:, :, 1:] + weight_x[:, :, :-1])).detach()
        if isinstance(weight_y, torch.Tensor):
            curvature_y = curvature_y * (0.5 * (weight_y[:, 1:, :] + weight_y[:, :-1, :])).detach()
        curvature_terms = [
            term for term in [_masked_mean(curvature_x, valid_curvature_x), _masked_mean(curvature_y, valid_curvature_y)] if term is not None
        ]
        if curvature_terms:
            terms.append(curvature_weight * torch.stack(curvature_terms).mean())

    if not terms:
        return prediction.sum() * 0.0
    return torch.stack(terms).mean()


def depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rgb: torch.Tensor | None = None,
    smoothness_weight: float = 0.0,
    smoothness_edge_weight: float = 8.0,
    smoothness_curvature_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid_count = mask.sum()
    if int(valid_count.detach().cpu()) == 0:
        zero = prediction.sum() * 0.0
        return zero, {
            "loss": 0.0,
            "data_loss": 0.0,
            "log_l1": 0.0,
            "silog": 0.0,
            "abs_rel_loss": 0.0,
            "smoothness_loss": 0.0,
            "smoothness_weighted_loss": 0.0,
            "valid_pixels": 0.0,
        }
    pred = prediction.clamp_min(1e-3)
    tgt = target.clamp_min(1e-3)
    log_diff = (pred.log() - tgt.log())[mask]
    log_l1 = log_diff.abs().mean()
    silog = torch.sqrt((log_diff.square().mean() - 0.85 * log_diff.mean().square()).clamp_min(0.0) + 1e-8)
    abs_rel = ((pred - tgt).abs() / tgt.clamp_min(1e-3))[mask].mean()
    data_loss = log_l1 + 0.15 * silog + 0.02 * abs_rel
    smooth = smoothness_loss(pred, mask, rgb, smoothness_edge_weight, smoothness_curvature_weight)
    smooth_weighted = smoothness_weight * smooth
    loss = data_loss + smooth_weighted
    return loss, {
        "loss": float(loss.detach().cpu()),
        "data_loss": float(data_loss.detach().cpu()),
        "log_l1": float(log_l1.detach().cpu()),
        "silog": float(silog.detach().cpu()),
        "abs_rel_loss": float(abs_rel.detach().cpu()),
        "smoothness_loss": float(smooth.detach().cpu()),
        "smoothness_weighted_loss": float(smooth_weighted.detach().cpu()),
        "valid_pixels": float(valid_count.detach().cpu()),
    }


def update_metric_sums(
    sums: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_parts: dict[str, float],
) -> None:
    pred = prediction.detach().float().clamp_min(1e-3)
    tgt = target.detach().float().clamp_min(1e-3)
    valid = mask.detach()
    count = int(valid.sum().item())
    if count == 0:
        return
    pred_v = pred[valid]
    tgt_v = tgt[valid]
    ratio = torch.maximum(pred_v / tgt_v, tgt_v / pred_v)
    sums["pixels"] += count
    sums["batches"] += 1
    sums["loss"] += loss_parts["loss"]
    sums["data_loss"] += loss_parts.get("data_loss", loss_parts["loss"])
    sums["smoothness_loss"] += loss_parts.get("smoothness_loss", 0.0)
    sums["smoothness_weighted_loss"] += loss_parts.get("smoothness_weighted_loss", 0.0)
    sums["abs_rel"] += float(((pred_v - tgt_v).abs() / tgt_v).sum().cpu())
    sums["rmse_sq"] += float((pred_v - tgt_v).square().sum().cpu())
    sums["log_mae"] += float((pred_v.log() - tgt_v.log()).abs().sum().cpu())
    sums["delta1"] += float((ratio < 1.25).float().sum().cpu())


def finalize_metric_sums(sums: dict[str, float]) -> dict[str, float]:
    pixels = max(1.0, sums["pixels"])
    batches = max(1.0, sums["batches"])
    return {
        "loss": sums["loss"] / batches,
        "data_loss": sums["data_loss"] / batches,
        "smoothness_loss": sums["smoothness_loss"] / batches,
        "smoothness_weighted_loss": sums["smoothness_weighted_loss"] / batches,
        "abs_rel": sums["abs_rel"] / pixels,
        "rmse": math.sqrt(sums["rmse_sq"] / pixels),
        "log_mae": sums["log_mae"] / pixels,
        "delta1": sums["delta1"] / pixels,
        "pixels": pixels,
        "batches": batches,
    }


def empty_metric_sums() -> dict[str, float]:
    return {
        "pixels": 0.0,
        "batches": 0.0,
        "loss": 0.0,
        "data_loss": 0.0,
        "smoothness_loss": 0.0,
        "smoothness_weighted_loss": 0.0,
        "abs_rel": 0.0,
        "rmse_sq": 0.0,
        "log_mae": 0.0,
        "delta1": 0.0,
    }


@torch.no_grad()
def evaluate_with_groups(
    model: DepthProForDepthEstimation,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    desc: str,
    smoothness_weight: float = 0.0,
    smoothness_edge_weight: float = 8.0,
    smoothness_curvature_weight: float = 0.5,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.depth_pro.eval()
    model.fusion_stage.eval()
    model.head.eval()
    sums = empty_metric_sums()
    group_sums: dict[str, dict[str, float]] = {}
    for batch in tqdm(loader, desc=desc, leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        prediction = predict_depth(model, pixel_values, autocast_dtype)
        prediction = downsample_prediction(prediction, target.shape[-2:])
        _loss, loss_parts = depth_loss(
            prediction,
            target,
            mask,
            rgb,
            smoothness_weight,
            smoothness_edge_weight,
            smoothness_curvature_weight,
        )
        update_metric_sums(sums, prediction, target, mask, loss_parts)

        for sample_index, sample in enumerate(batch["samples"]):
            sample_prediction = prediction[sample_index : sample_index + 1]
            sample_target = target[sample_index : sample_index + 1]
            sample_mask = mask[sample_index : sample_index + 1]
            sample_rgb = rgb[sample_index : sample_index + 1]
            _sample_loss, sample_loss_parts = depth_loss(
                sample_prediction,
                sample_target,
                sample_mask,
                sample_rgb,
                smoothness_weight,
                smoothness_edge_weight,
                smoothness_curvature_weight,
            )
            group_key = sample.body_part or sample.source
            if group_key not in group_sums:
                group_sums[group_key] = empty_metric_sums()
            update_metric_sums(
                group_sums[group_key],
                sample_prediction,
                sample_target,
                sample_mask,
                sample_loss_parts,
            )

    return finalize_metric_sums(sums), {
        group_key: finalize_metric_sums(values) for group_key, values in sorted(group_sums.items())
    }


@torch.no_grad()
def evaluate(
    model: DepthProForDepthEstimation,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    desc: str,
    smoothness_weight: float = 0.0,
    smoothness_edge_weight: float = 8.0,
    smoothness_curvature_weight: float = 0.5,
) -> dict[str, float]:
    metrics, _group_metrics = evaluate_with_groups(
        model,
        loader,
        device,
        autocast_dtype,
        desc,
        smoothness_weight,
        smoothness_edge_weight,
        smoothness_curvature_weight,
    )
    return metrics


def train_one_epoch(
    model: DepthProForDepthEstimation,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    epoch: int,
    smoothness_weight: float,
    smoothness_edge_weight: float,
    smoothness_curvature_weight: float,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.depth_pro.eval()
    model.fusion_stage.train()
    model.head.train()
    sums = empty_metric_sums()
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = predict_depth(model, pixel_values, autocast_dtype)
        prediction = downsample_prediction(prediction, target.shape[-2:])
        loss, loss_parts = depth_loss(
            prediction,
            target,
            mask,
            rgb,
            smoothness_weight,
            smoothness_edge_weight,
            smoothness_curvature_weight,
        )
        if not torch.isfinite(loss):
            print("skipping non-finite loss batch", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()

        update_metric_sums(sums, prediction, target, mask, loss_parts)
        progress.set_postfix(
            loss=f"{loss_parts['loss']:.4f}",
            data=f"{loss_parts['data_loss']:.4f}",
            smooth=f"{loss_parts['smoothness_loss']:.4f}",
        )
    return finalize_metric_sums(sums)


def write_samples_csv(samples: list[Sample], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "source", "body_part", "image_path", "depth_path", "width", "height"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "body_part": sample.body_part,
                    "image_path": repo_relative(sample.image_path),
                    "depth_path": repo_relative(sample.depth_path),
                    "width": sample.width,
                    "height": sample.height,
                }
            )


def write_metrics_csv(metrics_rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "phase",
        "epoch",
        "loss",
        "data_loss",
        "smoothness_loss",
        "smoothness_weighted_loss",
        "abs_rel",
        "rmse",
        "log_mae",
        "delta1",
        "pixels",
        "batches",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_group_metrics_csv(
    metrics_by_phase: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fieldnames = [
        "phase",
        "epoch",
        "body_part",
        "loss",
        "data_loss",
        "smoothness_loss",
        "smoothness_weighted_loss",
        "abs_rel",
        "rmse",
        "log_mae",
        "delta1",
        "pixels",
        "batches",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_by_phase:
            phase = row["phase"]
            epoch = row["epoch"]
            for body_part, metrics in sorted(row["metrics"].items()):
                writer.writerow(
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "body_part": body_part,
                        **{key: metrics.get(key, "") for key in fieldnames if key not in {"phase", "epoch", "body_part"}},
                    }
                )


def depth_visual(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0.0)
    else:
        mask = mask & np.isfinite(depth) & (depth > 0.0)
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


def paste_panel(tile: Image.Image, image: Image.Image, x: int, y: int, size: int) -> None:
    tile.paste(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), (x, y))


def make_comparison_tile(
    rgb_path: Path,
    gt_vis_path: Path,
    baseline_vis_path: Path,
    finetuned_vis_path: Path,
    size: int,
    label_height: int,
    sample_id: str | None = None,
) -> Image.Image:
    labels = ["RGB", "GT depth", "Depth Pro", "Fine-tuned"]
    tile = Image.new("RGB", (size * 4, size + label_height), "white")
    draw = ImageDraw.Draw(tile)
    for idx, label in enumerate(labels):
        draw.text((idx * size + 4, 3), label, fill=(0, 0, 0))
    if sample_id:
        draw.text((4, max(14, label_height - 14)), sample_id, fill=(0, 0, 0))
    paste_panel(tile, Image.open(rgb_path), 0, label_height, size)
    paste_panel(tile, Image.open(gt_vis_path).convert("L"), size, label_height, size)
    paste_panel(tile, Image.open(baseline_vis_path).convert("L"), size * 2, label_height, size)
    paste_panel(tile, Image.open(finetuned_vis_path).convert("L"), size * 3, label_height, size)
    return tile


def build_montage(rows: list[dict[str, Any]], output_path: Path, size: int = 112, columns: int = 3) -> None:
    label_height = 24
    tiles = [
        make_comparison_tile(
            Path(row["image_path"]),
            Path(row["gt_depth_vis_path"]),
            Path(row["baseline_depth_vis_path"]),
            Path(row["finetuned_depth_vis_path"]),
            size,
            label_height,
        )
        for row in rows
    ]
    rows_count = int(math.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * size * 4, rows_count * (size + label_height)), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile.width
        y = (idx // columns) * tile.height
        montage.paste(tile, (x, y))
    montage.save(output_path)


def build_gif(rows: list[dict[str, Any]], output_path: Path, size: int = 150) -> None:
    frames = []
    for row in rows:
        tile = make_comparison_tile(
            Path(row["image_path"]),
            Path(row["gt_depth_vis_path"]),
            Path(row["baseline_depth_vis_path"]),
            Path(row["finetuned_depth_vis_path"]),
            size,
            36,
            row["sample_id"],
        )
        frames.append(np.asarray(tile))
    imageio.mimsave(output_path, frames, duration=0.18, loop=0)


@torch.no_grad()
def predict_for_visual_rows(
    model: DepthProForDepthEstimation,
    processor: DepthProImageProcessor,
    samples: list[Sample],
    output_data_root: Path,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    initial_state: dict[str, Any],
    best_state: dict[str, Any],
    target_space: str,
) -> list[dict[str, Any]]:
    pred_root = output_data_root / "validation_predictions"
    baseline_depth_root = pred_root / "baseline_depth"
    baseline_vis_root = pred_root / "baseline_depth_vis"
    finetuned_depth_root = pred_root / "finetuned_depth"
    finetuned_vis_root = pred_root / "finetuned_depth_vis"
    gt_vis_root = pred_root / "gt_depth_vis"
    for path in [baseline_depth_root, baseline_vis_root, finetuned_depth_root, finetuned_vis_root, gt_vis_root]:
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    collate = make_collate(processor, target_space)
    for sample in tqdm(samples, desc="validation visual predictions", leave=False):
        depth = np.load(sample.depth_path).astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0.0)
        gt_vis_path = gt_vis_root / f"{sample.sample_id}_gt_depth_vis.png"
        imageio.imwrite(gt_vis_path, depth_visual(depth, mask))

        batch = collate([{"sample": sample, "image": Image.open(sample.image_path).convert("RGB"), "depth": depth}])
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        predictions: dict[str, np.ndarray] = {}
        for label, state in [("baseline", initial_state), ("finetuned", best_state)]:
            load_adapter_state(model, state)
            model.fusion_stage.eval()
            model.head.eval()
            pred = predict_depth(model, pixel_values, autocast_dtype)
            pred = downsample_prediction(pred, depth.shape).squeeze(0).detach().cpu().numpy().astype(np.float32)
            predictions[label] = prediction_to_metric_depth(pred, target_space)

        baseline_path = baseline_depth_root / f"{sample.sample_id}_baseline_depth.npy"
        finetuned_path = finetuned_depth_root / f"{sample.sample_id}_finetuned_depth.npy"
        baseline_vis_path = baseline_vis_root / f"{sample.sample_id}_baseline_depth_vis.png"
        finetuned_vis_path = finetuned_vis_root / f"{sample.sample_id}_finetuned_depth_vis.png"
        np.save(baseline_path, predictions["baseline"])
        np.save(finetuned_path, predictions["finetuned"])
        imageio.imwrite(baseline_vis_path, depth_visual(predictions["baseline"], mask))
        imageio.imwrite(finetuned_vis_path, depth_visual(predictions["finetuned"], mask))
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "target_space": target_space,
                "image_path": str(sample.image_path),
                "gt_depth_path": str(sample.depth_path),
                "gt_depth_vis_path": str(gt_vis_path),
                "baseline_depth_path": str(baseline_path),
                "baseline_depth_vis_path": str(baseline_vis_path),
                "finetuned_depth_path": str(finetuned_path),
                "finetuned_depth_vis_path": str(finetuned_vis_path),
            }
        )
    return rows


def write_visual_manifest(rows: list[dict[str, Any]], output_path: Path, output_root: Path) -> None:
    fieldnames = [
        "sample_id",
        "source",
        "target_space",
        "image_path",
        "gt_depth_path",
        "gt_depth_vis_path",
        "baseline_depth_path",
        "baseline_depth_vis_path",
        "finetuned_depth_path",
        "finetuned_depth_vis_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in fieldnames:
                if key.endswith("_path"):
                    out[key] = repo_relative(Path(out[key]))
            writer.writerow(out)


def write_notebook(output_root: Path, visual_manifest_path: Path, notebook_path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Fine-tuned Depth Pro validation surfaces\n\n"
            "This executed notebook loads validation predictions and displays interactive Plotly surfaces "
            "for ground truth, baseline Depth Pro, and the fine-tuned adapter."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import numpy as np\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"ROOT = Path({str(ROOT)!r})\n"
            f"OUTPUT_ROOT = Path({str(output_root)!r})\n"
            f"VISUAL_MANIFEST = Path({str(visual_manifest_path)!r})\n"
            "with VISUAL_MANIFEST.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n"
            "print(f'Loaded {len(rows)} validation visual rows from {VISUAL_MANIFEST}')"
        ),
        nbformat.v4.new_code_cell(
            "def resolve(path_value):\n"
            "    path = Path(path_value)\n"
            "    return path if path.is_absolute() else ROOT / path\n\n"
            "def surface_values(depth, mask=None, max_points=96):\n"
            "    if mask is None:\n"
            "        mask = np.isfinite(depth) & (depth > 0.0)\n"
            "    else:\n"
            "        mask = mask & np.isfinite(depth) & (depth > 0.0)\n"
            "    z = np.where(mask, depth, np.nan).astype(np.float32)\n"
            "    stride = max(1, int(np.ceil(max(z.shape) / max_points)))\n"
            "    z = z[::stride, ::stride]\n"
            "    if np.isfinite(z).any():\n"
            "        median = float(np.nanmedian(z))\n"
            "        scale = float(np.nanpercentile(np.abs(z - median), 95)) or 1.0\n"
            "        z = np.clip((z - median) / scale, -1.5, 1.5)\n"
            "        z = np.nan_to_num(z, nan=1.5)\n"
            "    else:\n"
            "        z = np.zeros_like(z)\n"
            "    h, w = z.shape\n"
            "    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))\n"
            "    return x, y, -z\n\n"
            "def make_surface_comparison(row):\n"
            "    arrays = [\n"
            "        ('GT depth', np.load(resolve(row['gt_depth_path']))),\n"
            "        ('Depth Pro', np.load(resolve(row['baseline_depth_path']))),\n"
            "        ('Fine-tuned', np.load(resolve(row['finetuned_depth_path']))),\n"
            "    ]\n"
            "    gt_mask = np.isfinite(arrays[0][1]) & (arrays[0][1] > 0.0)\n"
            "    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]], subplot_titles=[name for name, _ in arrays])\n"
            "    for col, (_name, depth) in enumerate(arrays, start=1):\n"
            "        x, y, z = surface_values(depth, gt_mask)\n"
            "        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale='Viridis', showscale=False, hoverinfo='skip'), row=1, col=col)\n"
            "    fig.update_layout(title=f\"{row['sample_id']}: baseline vs fine-tuned\", height=560, margin=dict(l=0, r=0, t=52, b=0))\n"
            "    fig.update_scenes(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data')\n"
            "    return fig"
        ),
        nbformat.v4.new_code_cell(
            "sample_index = 0\n"
            "fig = make_surface_comparison(rows[sample_index])\n"
            "fig"
        ),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    for cell in executed.cells:
        if cell.get("cell_type") == "code":
            cell["source"] = ""
    nbformat.write(executed, notebook_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    data_root = output_root / "data"
    visualizations_root = output_root / "visualizations"
    checkpoint_root = data_root / "checkpoints"
    for path in [data_root, visualizations_root, checkpoint_root]:
        path.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    autocast_dtype = amp_dtype(args.amp_dtype, device)
    print(f"using device={device} amp_dtype={autocast_dtype}", flush=True)

    loaded_train_samples = load_samples(args.manifests)
    if args.val_manifests:
        loaded_val_samples = load_samples(args.val_manifests)
        train_samples, skipped_train_samples = filter_valid_samples(loaded_train_samples, args.min_valid_fraction)
        val_samples, skipped_val_samples = filter_valid_samples(loaded_val_samples, args.min_valid_fraction)
        skipped_samples = skipped_train_samples + skipped_val_samples
        rng = random.Random(args.seed)
        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        if args.max_train_samples is not None:
            train_samples = train_samples[: args.max_train_samples]
        if args.max_val_samples is not None:
            val_samples = val_samples[: args.max_val_samples]
        samples = train_samples + val_samples
    else:
        samples, skipped_samples = filter_valid_samples(loaded_train_samples, args.min_valid_fraction)
        train_samples, val_samples = split_samples(
            samples,
            args.val_fraction,
            args.seed,
            args.max_train_samples,
            args.max_val_samples,
        )
    if not train_samples:
        raise ValueError("No training samples remain after filtering/sampling")
    if not val_samples:
        raise ValueError("No validation samples remain after filtering/sampling")
    if skipped_samples:
        skipped_path = data_root / "skipped_invalid_depth_samples.csv"
        with skipped_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "depth_path", "valid_fraction"])
            writer.writeheader()
            writer.writerows(skipped_samples)
        print(f"skipped {len(skipped_samples)} samples with too little valid depth", flush=True)
    write_samples_csv(train_samples, data_root / "train_manifest.csv")
    write_samples_csv(val_samples, data_root / "val_manifest.csv")
    print(f"loaded samples: train={len(train_samples)} val={len(val_samples)}", flush=True)

    processor = DepthProImageProcessor.from_pretrained(args.model_id)
    model = DepthProForDepthEstimation.from_pretrained(args.model_id).to(device)
    trainable_parameters = set_trainable_modules(model, args.trainable)
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"trainable parameters: {trainable_count:,} ({args.trainable})", flush=True)

    initial_state = adapter_state(model, args.trainable)
    torch.save(initial_state, checkpoint_root / "initial_fusion_head.pt")

    train_loader = DataLoader(
        DepthMapDataset(train_samples, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate(processor, args.target_space),
    )
    val_loader = DataLoader(
        DepthMapDataset(val_samples, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate(processor, args.target_space),
    )

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    metrics_rows: list[dict[str, Any]] = []
    group_metrics_rows: list[dict[str, Any]] = []

    baseline_metrics, baseline_by_body_part = evaluate_with_groups(
        model,
        val_loader,
        device,
        autocast_dtype,
        desc="baseline val",
        smoothness_weight=args.smoothness_weight,
        smoothness_edge_weight=args.smoothness_edge_weight,
        smoothness_curvature_weight=args.smoothness_curvature_weight,
    )
    metrics_rows.append({"phase": "baseline_val", "epoch": 0, **baseline_metrics})
    group_metrics_rows.append({"phase": "baseline_val", "epoch": 0, "metrics": baseline_by_body_part})
    print(f"baseline val: {baseline_metrics}", flush=True)

    best_loss = baseline_metrics["loss"]
    best_state = initial_state
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            autocast_dtype,
            epoch,
            smoothness_weight=args.smoothness_weight,
            smoothness_edge_weight=args.smoothness_edge_weight,
            smoothness_curvature_weight=args.smoothness_curvature_weight,
        )
        metrics_rows.append({"phase": "train", "epoch": epoch, **train_metrics})
        print(f"epoch {epoch} train: {train_metrics}", flush=True)

        if args.eval_every_epoch or epoch == args.epochs:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                autocast_dtype,
                desc=f"epoch {epoch} val",
                smoothness_weight=args.smoothness_weight,
                smoothness_edge_weight=args.smoothness_edge_weight,
                smoothness_curvature_weight=args.smoothness_curvature_weight,
            )
            metrics_rows.append({"phase": "val", "epoch": epoch, **val_metrics})
            print(f"epoch {epoch} val: {val_metrics}", flush=True)
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_state = adapter_state(model, args.trainable)
                torch.save(
                    {
                        "model_id": args.model_id,
                        "target_space": args.target_space,
                        "smoothness_weight": args.smoothness_weight,
                        "smoothness_edge_weight": args.smoothness_edge_weight,
                        "smoothness_curvature_weight": args.smoothness_curvature_weight,
                        "epoch": epoch,
                        "metrics": val_metrics,
                        "adapter": best_state,
                    },
                    checkpoint_root / "best_fusion_head_adapter.pt",
                )

        torch.save(
            {
                "model_id": args.model_id,
                "target_space": args.target_space,
                "smoothness_weight": args.smoothness_weight,
                "smoothness_edge_weight": args.smoothness_edge_weight,
                "smoothness_curvature_weight": args.smoothness_curvature_weight,
                "epoch": epoch,
                "adapter": adapter_state(model, args.trainable),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_root / "last_training_state.pt",
        )
        write_metrics_csv(metrics_rows, data_root / "metrics.csv")

    if not (checkpoint_root / "best_fusion_head_adapter.pt").exists():
        torch.save(
            {
                "model_id": args.model_id,
                "target_space": args.target_space,
                "smoothness_weight": args.smoothness_weight,
                "smoothness_edge_weight": args.smoothness_edge_weight,
                "smoothness_curvature_weight": args.smoothness_curvature_weight,
                "epoch": 0,
                "metrics": baseline_metrics,
                "adapter": best_state,
            },
            checkpoint_root / "best_fusion_head_adapter.pt",
        )

    load_adapter_state(model, best_state)
    final_metrics, best_by_body_part = evaluate_with_groups(
        model,
        val_loader,
        device,
        autocast_dtype,
        desc="best val",
        smoothness_weight=args.smoothness_weight,
        smoothness_edge_weight=args.smoothness_edge_weight,
        smoothness_curvature_weight=args.smoothness_curvature_weight,
    )
    metrics_rows.append({"phase": "best_val", "epoch": "best", **final_metrics})
    write_metrics_csv(metrics_rows, data_root / "metrics.csv")
    write_group_metrics_csv(group_metrics_rows + [{"phase": "best_val", "epoch": "best", "metrics": best_by_body_part}], data_root / "metrics_by_body_part.csv")

    visual_samples = val_samples[: min(args.viz_count, len(val_samples))]
    visual_rows = predict_for_visual_rows(
        model,
        processor,
        visual_samples,
        data_root,
        device,
        autocast_dtype,
        initial_state,
        best_state,
        args.target_space,
    )
    visual_manifest_path = data_root / "validation_visual_manifest.csv"
    write_visual_manifest(visual_rows, visual_manifest_path, output_root)
    montage_path = visualizations_root / "validation_montage_rgb_gt_baseline_finetuned.png"
    gif_path = visualizations_root / "validation_baseline_vs_finetuned.gif"
    notebook_path = visualizations_root / "plot_finetuned_depth_pro_surfaces.ipynb"
    build_montage(visual_rows, montage_path)
    build_gif(visual_rows, gif_path)
    write_notebook(output_root, visual_manifest_path, notebook_path)

    summary = {
        "model_id": args.model_id,
        "trainable": args.trainable,
        "target_space": args.target_space,
        "device": str(device),
        "amp_dtype": str(autocast_dtype),
        "sample_count": len(samples),
        "skipped_invalid_depth_count": len(skipped_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "smoothness_weight": args.smoothness_weight,
        "smoothness_edge_weight": args.smoothness_edge_weight,
        "smoothness_curvature_weight": args.smoothness_curvature_weight,
        "baseline_val": baseline_metrics,
        "baseline_val_by_body_part": baseline_by_body_part,
        "best_val": final_metrics,
        "best_val_by_body_part": best_by_body_part,
        "outputs": {
            "train_manifest": repo_relative(data_root / "train_manifest.csv"),
            "val_manifest": repo_relative(data_root / "val_manifest.csv"),
            "metrics": repo_relative(data_root / "metrics.csv"),
            "metrics_by_body_part": repo_relative(data_root / "metrics_by_body_part.csv"),
            "best_adapter": repo_relative(checkpoint_root / "best_fusion_head_adapter.pt"),
            "last_training_state": repo_relative(checkpoint_root / "last_training_state.pt"),
            "visual_manifest": repo_relative(visual_manifest_path),
            "montage": repo_relative(montage_path),
            "gif": repo_relative(gif_path),
            "notebook": repo_relative(notebook_path),
        },
    }
    (data_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

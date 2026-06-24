from __future__ import annotations

import csv
import importlib
import json
import math
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


Point = tuple[float, float]
ScalePoints = tuple[Point, Point]


@dataclass(frozen=True)
class LesionVolumeResult:
    image_path: str
    output_dir: str
    pixels_per_cm: float
    total_volume_cm3: float
    lesions: list[dict[str, Any]]
    outputs: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "output_dir": self.output_dir,
            "pixels_per_cm": self.pixels_per_cm,
            "total_volume_cm3": self.total_volume_cm3,
            "lesions": self.lesions,
            "outputs": self.outputs,
            "warnings": self.warnings,
        }


class LesionVolumePipeline:
    """Compute neurofibroma lesion volume from an RGB image and lesion outlines.

    The public entry point is :meth:`compute_volume`. It predicts a Depth Pro
    depth map, fits a local background plane around each lesion outline, and
    integrates the positive lesion relief above that fitted surface.
    """

    def __init__(
        self,
        model_id: str = "apple/DepthPro-hf",
        device: str = "auto",
        auto_install_depthpro: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.auto_install_depthpro = auto_install_depthpro
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def compute_volume(
        self,
        image_path: str | Path,
        lesions: Sequence[Any],
        scale_points: Sequence[Sequence[float]],
        output_dir: str | Path | None = None,
        generate_visuals: bool = False,
        visuals: Iterable[str] | None = None,
        point_radius_px: int = 14,
        ring_width_px: int | None = None,
        max_height_cm: float | None = None,
        depth_override_m: np.ndarray | None = None,
    ) -> LesionVolumeResult:
        """Run the lesion volume pipeline.

        Args:
            image_path: RGB image path.
            lesions: Lesion outlines. Each item can be a polygon list
                ``[(x, y), ...]``, a dict with ``points``/``polygon``, or a dict
                with ``center`` and optional ``radius_px``.
            scale_points: Two image coordinates that mark exactly 1 cm.
            output_dir: Directory for tables, arrays, masks, and visuals.
            generate_visuals: If true, write default GIF and montage outputs.
            visuals: Optional iterable containing any of ``gif``, ``montage``,
                and ``mov``.
            point_radius_px: Fallback disk radius when only a lesion center is
                provided.
            ring_width_px: Background ring width around each lesion. By default
                this is derived from lesion size.
            max_height_cm: Optional clip for integrated relief height.
            depth_override_m: Optional depth map in meters for tests or callers
                that already ran Depth Pro.
        """

        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        output_root = Path(output_dir) if output_dir else image_path.parent / f"{image_path.stem}_lesion_volume"
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        scale_pair = _parse_scale_points(scale_points)
        pixels_per_cm = _pixels_per_cm(scale_pair)
        pixel_area_cm2 = 1.0 / (pixels_per_cm * pixels_per_cm)

        lesion_specs = [_parse_lesion_spec(lesion, idx, point_radius_px) for idx, lesion in enumerate(lesions, start=1)]
        if not lesion_specs:
            raise ValueError("At least one lesion outline or lesion center is required.")

        masks: list[np.ndarray] = []
        for spec in lesion_specs:
            mask = _mask_from_spec(spec, width, height)
            if int(mask.sum()) == 0:
                raise ValueError(f"Lesion {spec['id']} produced an empty mask.")
            masks.append(mask)

        union_mask = np.zeros((height, width), dtype=bool)
        for mask in masks:
            union_mask |= mask

        warnings_list: list[str] = []
        if depth_override_m is None:
            depth_m = self._predict_depth_m(image)
        else:
            depth_m = np.asarray(depth_override_m, dtype=np.float32)
            if depth_m.ndim != 2:
                raise ValueError("depth_override_m must be a 2D array in meters.")
            if depth_m.shape != (height, width):
                depth_m = _resize_depth_nearest(depth_m, (width, height))

        depth_m = np.nan_to_num(depth_m.astype(np.float32), nan=np.nanmedian(depth_m), posinf=np.nanmedian(depth_m), neginf=np.nanmedian(depth_m))
        depth_path = output_root / f"{image_path.stem}_depth_m.npy"
        np.save(depth_path, depth_m)

        mask_dir = output_root / "masks"
        mask_dir.mkdir(exist_ok=True)
        lesion_rows: list[dict[str, Any]] = []
        for spec, mask in zip(lesion_specs, masks):
            width_px = ring_width_px or max(8, min(48, int(round(math.sqrt(float(mask.sum())) * 0.65))))
            ring = _ring_mask(mask, union_mask, width_px)
            if int(ring.sum()) < 24:
                wider_ring = _ring_mask(mask, union_mask, width_px * 2)
                if int(wider_ring.sum()) >= 24:
                    ring = wider_ring
            if int(ring.sum()) < 3:
                raise ValueError(f"Lesion {spec['id']} does not have enough nearby background pixels for plane fitting.")

            surface_m = _fit_plane_surface(depth_m, ring)
            deviation_m = depth_m - surface_m
            target_deviation_m = deviation_m[mask]
            direction = -1.0 if float(np.nanmedian(target_deviation_m)) < 0 else 1.0
            height_cm = np.maximum(direction * target_deviation_m, 0.0) * 100.0
            if max_height_cm is not None:
                height_cm = np.minimum(height_cm, float(max_height_cm))

            volume_cm3 = float(height_cm.sum() * pixel_area_cm2)
            area_px = int(mask.sum())
            row = {
                "lesion_id": spec["id"],
                "name": spec["name"],
                "area_px": area_px,
                "area_cm2": float(area_px * pixel_area_cm2),
                "median_height_cm": float(np.nanmedian(height_cm)) if height_cm.size else 0.0,
                "mean_height_cm": float(np.nanmean(height_cm)) if height_cm.size else 0.0,
                "max_height_cm": float(np.nanmax(height_cm)) if height_cm.size else 0.0,
                "volume_cm3": volume_cm3,
                "relief_direction": "toward_camera" if direction < 0 else "away_from_camera",
                "ring_px": int(ring.sum()),
            }
            lesion_rows.append(row)

            mask_path = mask_dir / f"{image_path.stem}_{spec['id']}_mask.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

        union_path = mask_dir / f"{image_path.stem}_lesion_union_mask.png"
        Image.fromarray((union_mask.astype(np.uint8) * 255), mode="L").save(union_path)

        csv_path = output_root / "lesion_volumes_cm3.csv"
        _write_csv(csv_path, lesion_rows, pixels_per_cm)

        outputs: dict[str, str] = {
            "depth_npy": str(depth_path),
            "lesion_csv": str(csv_path),
            "union_mask": str(union_path),
        }

        visual_root = output_root / "visualizations"
        visual_root.mkdir(exist_ok=True)
        depth_vis_path = visual_root / f"{image_path.stem}_depth.png"
        _depth_visual(depth_m).save(depth_vis_path)
        outputs["depth_png"] = str(depth_vis_path)

        requested_visuals = set(visuals or [])
        if generate_visuals and not requested_visuals:
            requested_visuals = {"gif", "montage"}
        requested_visuals = {item.lower() for item in requested_visuals}
        unknown_visuals = requested_visuals - {"gif", "montage", "mov"}
        if unknown_visuals:
            raise ValueError(f"Unknown visual output(s): {sorted(unknown_visuals)}")

        if requested_visuals:
            visual_outputs, visual_warnings = _write_visuals(
                image=image,
                depth_m=depth_m,
                union_mask=union_mask,
                lesion_specs=lesion_specs,
                scale_points=scale_pair,
                output_root=visual_root,
                image_stem=image_path.stem,
                requested=requested_visuals,
            )
            outputs.update(visual_outputs)
            warnings_list.extend(visual_warnings)

        summary = {
            "image_path": str(image_path),
            "pixels_per_cm": pixels_per_cm,
            "scale_points": scale_pair,
            "total_volume_cm3": float(sum(row["volume_cm3"] for row in lesion_rows)),
            "lesions": lesion_rows,
            "outputs": outputs,
            "warnings": warnings_list,
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs["summary_json"] = str(summary_path)

        return LesionVolumeResult(
            image_path=str(image_path),
            output_dir=str(output_root),
            pixels_per_cm=pixels_per_cm,
            total_volume_cm3=summary["total_volume_cm3"],
            lesions=lesion_rows,
            outputs=outputs,
            warnings=warnings_list,
        )

    def _predict_depth_m(self, image: Image.Image) -> np.ndarray:
        self._ensure_depthpro_dependencies()
        torch = self._torch
        if torch is None:
            import torch as torch_module

            torch = torch_module
            self._torch = torch

        device = _select_device(torch, self.device)
        if self._processor is None or self._model is None:
            transformers = importlib.import_module("transformers")
            try:
                processor_cls = getattr(transformers, "DepthProImageProcessor")
                model_cls = getattr(transformers, "DepthProForDepthEstimation")
            except AttributeError as exc:
                if self.auto_install_depthpro:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
                    importlib.invalidate_caches()
                    transformers = importlib.import_module("transformers")
                    processor_cls = getattr(transformers, "DepthProImageProcessor")
                    model_cls = getattr(transformers, "DepthProForDepthEstimation")
                else:
                    raise RuntimeError(
                        "This transformers install does not include Depth Pro classes. "
                        "Install a newer transformers version or enable auto_install_depthpro."
                    ) from exc
            self._processor = processor_cls.from_pretrained(self.model_id)
            self._model = model_cls.from_pretrained(self.model_id).to(device)
            self._model.eval()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        post = self._processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        predicted = post[0].get("predicted_depth", post[0].get("depth"))
        if predicted is None:
            raise RuntimeError("Depth Pro post-processing did not return a depth tensor.")
        return predicted.detach().cpu().numpy().astype(np.float32)

    def _ensure_depthpro_dependencies(self) -> None:
        missing: list[str] = []
        if importlib.util.find_spec("torch") is None:
            missing.append("torch")
        if importlib.util.find_spec("transformers") is None:
            missing.append("transformers")
        if missing:
            if not self.auto_install_depthpro:
                raise RuntimeError(
                    "Depth Pro dependencies are missing: "
                    + ", ".join(missing)
                    + ". Install them or enable auto_install_depthpro."
                )
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            importlib.invalidate_caches()
        self._torch = importlib.import_module("torch")


def _parse_scale_points(scale_points: Sequence[Sequence[float]]) -> ScalePoints:
    arr = np.asarray(scale_points, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError("scale_points must contain exactly two [x, y] coordinates that mark 1 cm.")
    return ((float(arr[0, 0]), float(arr[0, 1])), (float(arr[1, 0]), float(arr[1, 1])))


def _pixels_per_cm(scale_points: ScalePoints) -> float:
    (x1, y1), (x2, y2) = scale_points
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance <= 0:
        raise ValueError("The two scale coordinates must be different points.")
    return float(distance)


def _parse_lesion_spec(lesion: Any, index: int, point_radius_px: int) -> dict[str, Any]:
    lesion_id = f"lesion_{index:03d}"
    name = lesion_id
    radius_px = point_radius_px
    points: Any

    if isinstance(lesion, Mapping):
        lesion_id = str(lesion.get("id", lesion_id))
        name = str(lesion.get("name", lesion_id))
        radius_px = int(lesion.get("radius_px", point_radius_px))
        if "points" in lesion:
            points = lesion["points"]
        elif "polygon" in lesion:
            points = lesion["polygon"]
        elif "center" in lesion:
            center = _as_point(lesion["center"])
            return {"id": lesion_id, "name": name, "kind": "center", "center": center, "radius_px": radius_px}
        else:
            raise ValueError(f"Lesion {index} dict must include points, polygon, or center.")
    else:
        points = lesion

    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1 and arr.shape == (2,):
        return {"id": lesion_id, "name": name, "kind": "center", "center": (float(arr[0]), float(arr[1])), "radius_px": radius_px}
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError(f"Lesion {index} points must be a polygon with at least three [x, y] coordinates.")
    polygon = [(float(x), float(y)) for x, y in arr]
    return {"id": lesion_id, "name": name, "kind": "polygon", "polygon": polygon}


def _as_point(value: Any) -> Point:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("A lesion center must be one [x, y] coordinate.")
    return (float(arr[0]), float(arr[1]))


def _mask_from_spec(spec: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if spec["kind"] == "polygon":
        draw.polygon(list(spec["polygon"]), fill=255)
    elif spec["kind"] == "center":
        x, y = spec["center"]
        radius = float(spec["radius_px"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    else:
        raise ValueError(f"Unsupported lesion spec kind: {spec['kind']}")
    return np.asarray(mask, dtype=np.uint8) > 0


def _ring_mask(mask: np.ndarray, union_mask: np.ndarray, width_px: int) -> np.ndarray:
    width_px = max(1, int(width_px))
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    dilated = mask_image.filter(ImageFilter.MaxFilter(width_px * 2 + 1))
    dilated_mask = np.asarray(dilated, dtype=np.uint8) > 0
    return dilated_mask & ~union_mask


def _fit_plane_surface(depth_m: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(depth_m.shape, dtype=np.float32)
    z = depth_m[fit_mask].astype(np.float64)
    x = xx[fit_mask].astype(np.float64)
    y = yy[fit_mask].astype(np.float64)
    finite = np.isfinite(z)
    if int(finite.sum()) < 3:
        raise ValueError("Plane fitting needs at least three finite background depth samples.")
    x = x[finite]
    y = y[finite]
    z = z[finite]
    x_center = float(x.mean())
    y_center = float(y.mean())
    x_scale = float(max(x.std(), 1.0))
    y_scale = float(max(y.std(), 1.0))
    design = np.column_stack(((x - x_center) / x_scale, (y - y_center) / y_scale, np.ones_like(x)))
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    full_design = np.stack(((xx - x_center) / x_scale, (yy - y_center) / y_scale, np.ones_like(xx)), axis=-1)
    surface = full_design @ coeffs
    return surface.astype(np.float32)


def _resize_depth_nearest(depth_m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(depth_m.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _select_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _write_csv(path: Path, rows: list[dict[str, Any]], pixels_per_cm: float) -> None:
    fields = [
        "lesion_id",
        "name",
        "pixels_per_cm",
        "area_px",
        "area_cm2",
        "median_height_cm",
        "mean_height_cm",
        "max_height_cm",
        "volume_cm3",
        "relief_direction",
        "ring_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["pixels_per_cm"] = pixels_per_cm
            writer.writerow(out)


def _depth_visual(depth_m: np.ndarray) -> Image.Image:
    finite = depth_m[np.isfinite(depth_m)]
    if finite.size == 0:
        norm = np.zeros(depth_m.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            high = low + 1.0
        norm = np.clip((depth_m - low) / (high - low), 0, 1)
        norm = (255 - (norm * 255)).astype(np.uint8)
    return Image.fromarray(norm, mode="L").convert("RGB")


def _write_visuals(
    image: Image.Image,
    depth_m: np.ndarray,
    union_mask: np.ndarray,
    lesion_specs: list[dict[str, Any]],
    scale_points: ScalePoints,
    output_root: Path,
    image_stem: str,
    requested: set[str],
) -> tuple[dict[str, str], list[str]]:
    outputs: dict[str, str] = {}
    warnings_list: list[str] = []
    overlay = _overlay_image(image, union_mask, lesion_specs, scale_points, alpha=0.45)
    depth_vis = _depth_visual(depth_m)

    if "montage" in requested:
        montage_path = output_root / f"{image_stem}_lesion_volume_montage.png"
        montage = Image.new("RGB", (image.width * 3, image.height), "white")
        montage.paste(image, (0, 0))
        montage.paste(depth_vis, (image.width, 0))
        montage.paste(overlay, (image.width * 2, 0))
        montage.save(montage_path)
        outputs["montage_png"] = str(montage_path)

    if "gif" in requested or "mov" in requested:
        frames = []
        for alpha in [0.18, 0.3, 0.45, 0.62, 0.45, 0.3]:
            frames.append(np.asarray(_overlay_image(image, union_mask, lesion_specs, scale_points, alpha=alpha)))
        if "gif" in requested:
            gif_path = output_root / f"{image_stem}_lesion_volume.gif"
            _write_animation(gif_path, frames, fps=6)
            outputs["gif"] = str(gif_path)
        if "mov" in requested:
            mov_path = output_root / f"{image_stem}_lesion_volume.mov"
            try:
                _write_animation(mov_path, frames, fps=6)
                outputs["mov"] = str(mov_path)
            except Exception as exc:  # pragma: no cover - depends on ffmpeg availability.
                warnings_list.append(f"Could not write MOV because ffmpeg/imageio failed: {exc}")

    return outputs, warnings_list


def _overlay_image(
    image: Image.Image,
    union_mask: np.ndarray,
    lesion_specs: list[dict[str, Any]],
    scale_points: ScalePoints,
    alpha: float,
) -> Image.Image:
    base = image.convert("RGBA")
    tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
    tint_arr = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    tint_arr[union_mask] = (226, 48, 48, int(255 * alpha))
    tint = Image.fromarray(tint_arr, mode="RGBA")
    combined = Image.alpha_composite(base, tint)
    draw = ImageDraw.Draw(combined)
    for spec in lesion_specs:
        if spec["kind"] == "polygon":
            points = list(spec["polygon"])
            if points:
                draw.line(points + [points[0]], fill=(255, 0, 0, 255), width=3)
        elif spec["kind"] == "center":
            x, y = spec["center"]
            radius = float(spec["radius_px"])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 0, 0, 255), width=3)
    (x1, y1), (x2, y2) = scale_points
    draw.line((x1, y1, x2, y2), fill=(20, 20, 20, 255), width=4)
    draw.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), fill=(20, 20, 20, 255))
    draw.ellipse((x2 - 4, y2 - 4, x2 + 4, y2 + 4), fill=(20, 20, 20, 255))
    return combined.convert("RGB")


def _write_animation(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for GIF/MOV visual outputs.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imageio.mimsave(path, frames, fps=fps)

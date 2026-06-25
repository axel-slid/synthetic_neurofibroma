from __future__ import annotations

import csv
import importlib
import json
import math
import subprocess
import sys
import warnings
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


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
        show_progress: bool = True,
        visual_ruler_count: int = 0,
    ) -> LesionVolumeResult:
        """Run the lesion volume pipeline.

        Args:
            image_path: RGB image path.
            lesions: Lesion outlines. Each item can be a polygon list
                ``[(x, y), ...]``, a dict with ``points``/``polygon``, or a dict
                with ``center`` and optional ``radius_px``.
            scale_points: Two image coordinates that mark exactly 1 cm.
            output_dir: Directory for tables, arrays, masks, and visuals.
            generate_visuals: If true, write default GIF, PNG, and MOV outputs.
            visuals: Optional iterable containing any of ``gif``, ``png``,
                ``mov``, and the legacy ``montage`` alias.
            point_radius_px: Fallback disk radius when only a lesion center is
                provided.
            ring_width_px: Background ring width around each lesion. By default
                this is derived from lesion size.
            max_height_cm: Optional clip for integrated relief height.
            depth_override_m: Optional depth map in meters for tests or callers
                that already ran Depth Pro.
            show_progress: Show a tqdm progress bar for major pipeline stages.
            visual_ruler_count: Number of one-centimeter ruler markers to draw
                on the visual depth panel. ``0`` preserves the no-ruler style.
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

        requested_visuals = set(visuals or [])
        if generate_visuals and not requested_visuals:
            requested_visuals = {"gif", "png", "mov"}
        requested_visuals = {item.lower() for item in requested_visuals}
        unknown_visuals = requested_visuals - {"gif", "png", "montage", "mov"}
        if unknown_visuals:
            raise ValueError(f"Unknown visual output(s): {sorted(unknown_visuals)}")
        visual_ruler_count = _validate_ruler_count(visual_ruler_count)

        lesion_specs = [_parse_lesion_spec(lesion, idx, point_radius_px) for idx, lesion in enumerate(lesions, start=1)]
        if not lesion_specs:
            raise ValueError("At least one lesion outline or lesion center is required.")

        progress = _make_progress(show_progress, total=4 + len(lesion_specs), desc="lesion volume")
        with progress as bar:
            bar.set_description("Build lesion masks")
            masks: list[np.ndarray] = []
            for spec in lesion_specs:
                mask = _mask_from_spec(spec, width, height)
                if int(mask.sum()) == 0:
                    raise ValueError(f"Lesion {spec['id']} produced an empty mask.")
                masks.append(mask)

            union_mask = np.zeros((height, width), dtype=bool)
            for mask in masks:
                union_mask |= mask
            bar.update(1)

            warnings_list: list[str] = []
            bar.set_description("Predict depth")
            if depth_override_m is None:
                depth_m = self._predict_depth_m(image)
            else:
                depth_m = np.asarray(depth_override_m, dtype=np.float32)
                if depth_m.ndim != 2:
                    raise ValueError("depth_override_m must be a 2D array in meters.")
                if depth_m.shape != (height, width):
                    depth_m = _resize_depth_nearest(depth_m, (width, height))
            bar.update(1)

            depth_m = np.nan_to_num(depth_m.astype(np.float32), nan=np.nanmedian(depth_m), posinf=np.nanmedian(depth_m), neginf=np.nanmedian(depth_m))
            depth_path = output_root / f"{image_path.stem}_depth_m.npy"
            np.save(depth_path, depth_m)

            mask_dir = output_root / "masks"
            mask_dir.mkdir(exist_ok=True)
            lesion_rows: list[dict[str, Any]] = []
            for spec, mask in zip(lesion_specs, masks):
                bar.set_description(f"Measure {spec['id']}")
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
                bar.update(1)

            bar.set_description("Write outputs")
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
            bar.update(1)

            bar.set_description("Write visualizations")
            if requested_visuals:
                visual_outputs, visual_warnings = _write_visuals(
                    image=image,
                    depth_m=depth_m,
                    lesion_rows=lesion_rows,
                    lesion_specs=lesion_specs,
                    output_root=visual_root,
                    image_stem=image_path.stem,
                    requested=requested_visuals,
                    show_progress=show_progress,
                    pixels_per_cm=pixels_per_cm,
                    ruler_count=visual_ruler_count,
                )
                outputs.update(visual_outputs)
                warnings_list.extend(visual_warnings)
            bar.update(1)

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

    def compute_from_json(
        self,
        lesions_json: str | Path,
        image_path: str | Path | None = None,
        scale_points: Sequence[Sequence[float]] | None = None,
        output_dir: str | Path | None = None,
        generate_visuals: bool = False,
        visuals: Iterable[str] | None = None,
        show_progress: bool = True,
        **compute_kwargs: Any,
    ) -> LesionVolumeResult:
        """Compute volume from a JSON file that stores image, lesions, and scale."""

        json_path = Path(lesions_json).expanduser().resolve()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("lesions_json must be an object with lesions and scale_points.")

        lesions = payload.get("lesions")
        if not lesions:
            raise ValueError("lesions_json must include a non-empty lesions list.")

        selected_scale_points = scale_points if scale_points is not None else payload.get("scale_points")
        if selected_scale_points is None:
            raise ValueError("scale_points must be provided in the JSON or as an override.")

        selected_image_path = image_path if image_path is not None else payload.get("image_path")
        if selected_image_path is None:
            raise ValueError("image_path must be provided in the JSON or as an override.")
        selected_image_path = _resolve_json_image_path(selected_image_path, json_path.parent)

        return self.compute_volume(
            image_path=selected_image_path,
            lesions=lesions,
            scale_points=selected_scale_points,
            output_dir=output_dir,
            generate_visuals=generate_visuals,
            visuals=visuals,
            show_progress=show_progress,
            **compute_kwargs,
        )

    def compute_from_table(
        self,
        annotations_csv: str | Path,
        output_dir: str | Path | None = None,
        image_root: str | Path | None = None,
        generate_visuals: bool = False,
        visuals: Iterable[str] | None = None,
        show_progress: bool = True,
        **compute_kwargs: Any,
    ) -> list[LesionVolumeResult]:
        """Compute volumes from the expected tabular annotation schema.

        Required columns are ``image_path``, ``ai_cnf_contours``,
        ``ruler_location``, ``ruler_distance_cm``, and ``lesion_id``.
        ``ruler_location`` is two image coordinates whose real-world length is
        ``ruler_distance_cm``.
        """

        annotations_path = Path(annotations_csv).expanduser().resolve()
        rows = _read_annotation_table(annotations_path)
        if not rows:
            raise ValueError(f"No annotation rows found in {annotations_path}")

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            image_key = row.get("image_path", "").strip()
            if not image_key:
                raise ValueError("Every annotation row must include image_path.")
            grouped.setdefault(image_key, []).append(row)

        base_output = Path(output_dir).expanduser().resolve() if output_dir else None
        root = Path(image_root).expanduser().resolve() if image_root else annotations_path.parent
        results: list[LesionVolumeResult] = []

        for image_key, image_rows in grouped.items():
            image_path = _resolve_table_image_path(image_key, root)
            scale_points = _scale_points_from_table_row(image_rows[0])
            lesions = [_lesion_from_table_row(row, idx) for idx, row in enumerate(image_rows, start=1)]
            if base_output is None or len(grouped) == 1:
                image_output = base_output
            else:
                image_output = base_output / image_path.stem
            results.append(
                self.compute_volume(
                    image_path=image_path,
                    lesions=lesions,
                    scale_points=scale_points,
                    output_dir=image_output,
                    generate_visuals=generate_visuals,
                    visuals=visuals,
                    show_progress=show_progress,
                    **compute_kwargs,
                )
            )
        return results

    def compute_from_csv(
        self,
        annotations_csv: str | Path,
        output_dir: str | Path | None = None,
        image_root: str | Path | None = None,
        generate_visuals: bool = False,
        visuals: Iterable[str] | None = None,
        show_progress: bool = True,
        **compute_kwargs: Any,
    ) -> list[LesionVolumeResult]:
        """Compute volumes from the expected one-row-per-lesion CSV schema."""

        return self.compute_from_table(
            annotations_csv=annotations_csv,
            output_dir=output_dir,
            image_root=image_root,
            generate_visuals=generate_visuals,
            visuals=visuals,
            show_progress=show_progress,
            **compute_kwargs,
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


def _validate_ruler_count(value: int) -> int:
    count = int(value)
    if count < 0:
        raise ValueError("visual_ruler_count must be greater than or equal to 0.")
    if count > 64:
        raise ValueError("visual_ruler_count must be 64 or fewer markers.")
    return count


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


def _read_annotation_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve_table_image_path(image_key: str, image_root: Path) -> Path:
    image_path = Path(image_key).expanduser()
    if not image_path.is_absolute():
        image_path = image_root / image_path
    return image_path.resolve()


def _resolve_json_image_path(image_key: str | Path, json_root: Path) -> Path:
    image_path = Path(image_key).expanduser()
    if not image_path.is_absolute():
        image_path = json_root / image_path
    return image_path.resolve()


def _scale_points_from_table_row(row: Mapping[str, str]) -> ScalePoints:
    if _missing(row.get("ruler_location")):
        raise ValueError("ruler_location is required and must contain two [x, y] coordinates.")
    ruler = np.asarray(_parse_table_literal(row["ruler_location"]), dtype=float)
    if ruler.shape != (2, 2):
        raise ValueError("ruler_location must be [[x1, y1], [x2, y2]].")
    distance_cm = float(row.get("ruler_distance_cm") or 1.0)
    if distance_cm <= 0:
        raise ValueError("ruler_distance_cm must be greater than zero.")
    x1, y1 = ruler[0]
    x2, y2 = ruler[1]
    return ((float(x1), float(y1)), (float(x1 + (x2 - x1) / distance_cm), float(y1 + (y2 - y1) / distance_cm)))


def _lesion_from_table_row(row: Mapping[str, str], index: int) -> dict[str, Any]:
    lesion_id = str(row.get("lesion_id") or f"lesion_{index:03d}")
    if not _missing(row.get("ai_cnf_contours")):
        points = _polygon_from_table_value(row["ai_cnf_contours"])
        return {"id": lesion_id, "name": lesion_id, "points": points}
    if not _missing(row.get("ai_cnf_points")):
        center = _parse_table_literal(row["ai_cnf_points"])
        return {"id": lesion_id, "name": lesion_id, "center": center}
    raise ValueError(f"Row {index} must include ai_cnf_contours or ai_cnf_points.")


def _polygon_from_table_value(value: str) -> list[list[float]]:
    parsed = _parse_table_literal(value)
    arr = np.asarray(parsed, dtype=float)
    if arr.ndim == 3:
        arr = max(arr, key=lambda contour: len(contour))
        arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError("ai_cnf_contours must be a polygon like [[x, y], [x, y], ...].")
    return [[float(x), float(y)] for x, y in arr]


def _parse_table_literal(value: str) -> Any:
    text = value.strip()
    if not text:
        raise ValueError("Cannot parse an empty table value.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _missing(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "nan", "NaN", "None", "null"}


class _NoopProgress:
    def __enter__(self) -> "_NoopProgress":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def set_description(self, _description: str) -> None:
        return None

    def update(self, _count: int = 1) -> None:
        return None


def _make_progress(enabled: bool, total: int, desc: str) -> Any:
    if not enabled:
        return _NoopProgress()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NoopProgress()
    return tqdm(total=total, desc=desc, unit="step")


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


_VISUAL_BACKGROUND = (245, 247, 250)
_VISUAL_TEXT = (28, 34, 42)
_PANEL_W = 340
_PANEL_H = 300
_PANEL_GAP = 12
_CAPTION_H = 34
_LEGEND_GAP = 20
_LEGEND_W = 168
_CM3_LABEL = "cm\u00b3"
_VIRIDIS_STOPS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)
_MAGMA_STOPS = (
    (0.00, (0, 0, 4)),
    (0.25, (72, 16, 110)),
    (0.50, (183, 55, 121)),
    (0.75, (251, 135, 97)),
    (1.00, (252, 253, 191)),
)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf" if bold else "/usr/share/fonts/truetype/lato/Lato-Medium.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _interpolate_color(value: float, stops: Sequence[tuple[float, tuple[int, int, int]]]) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    for index in range(1, len(stops)):
        left_t, left_rgb = stops[index - 1]
        right_t, right_rgb = stops[index]
        if value <= right_t:
            frac = (value - left_t) / max(right_t - left_t, 1e-12)
            return tuple(
                int(round(left_rgb[channel] + frac * (right_rgb[channel] - left_rgb[channel])))
                for channel in range(3)
            )
    return stops[-1][1]


def _volume_color(norm: float) -> tuple[int, int, int]:
    return _interpolate_color(norm, _VIRIDIS_STOPS)


def _apply_colormap(norm: np.ndarray, stops: Sequence[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    clipped = np.clip(norm.astype(np.float32), 0.0, 1.0)
    colors = np.zeros((*clipped.shape, 3), dtype=np.float32)
    for index in range(1, len(stops)):
        left_t, left_rgb = stops[index - 1]
        right_t, right_rgb = stops[index]
        mask = (clipped >= left_t) & (clipped <= right_t if index == len(stops) - 1 else clipped < right_t)
        if not np.any(mask):
            continue
        frac = (clipped[mask] - left_t) / max(right_t - left_t, 1e-12)
        left_arr = np.asarray(left_rgb, dtype=np.float32)
        right_arr = np.asarray(right_rgb, dtype=np.float32)
        colors[mask] = left_arr + frac[:, None] * (right_arr - left_arr)
    return np.clip(np.round(colors), 0, 255).astype(np.uint8)


def _depth_heatmap(depth_m: np.ndarray) -> Image.Image:
    finite = np.isfinite(depth_m)
    norm = np.zeros(depth_m.shape, dtype=np.float32)
    if np.any(finite):
        low, high = np.percentile(depth_m[finite], [2, 98])
        if high <= low:
            high = low + 1e-6
        norm[finite] = np.clip((depth_m[finite] - low) / (high - low), 0.0, 1.0)
    rgb = _apply_colormap(norm, _MAGMA_STOPS)
    return Image.fromarray(rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=1.1))


def _fit_visual_panel(image: Image.Image) -> Image.Image:
    panel = Image.new("RGB", (_PANEL_W, _PANEL_H), _VISUAL_BACKGROUND)
    contained = ImageOps.contain(image.convert("RGB"), (_PANEL_W, _PANEL_H), Image.Resampling.LANCZOS)
    panel.paste(contained, ((_PANEL_W - contained.width) // 2, (_PANEL_H - contained.height) // 2))
    return panel


def _visual_panel_geometry(image_size: tuple[int, int]) -> tuple[float, int, int, int, int]:
    image_w, image_h = image_size
    scale = min(_PANEL_W / max(image_w, 1), _PANEL_H / max(image_h, 1))
    fitted_w = int(round(image_w * scale))
    fitted_h = int(round(image_h * scale))
    offset_x = (_PANEL_W - fitted_w) // 2
    offset_y = (_PANEL_H - fitted_h) // 2
    return scale, offset_x, offset_y, fitted_w, fitted_h


def _draw_visual_rulers(
    panel: Image.Image,
    image_size: tuple[int, int],
    pixels_per_cm: float,
    ruler_count: int,
) -> Image.Image:
    if ruler_count <= 0:
        return panel

    scale, offset_x, offset_y, fitted_w, fitted_h = _visual_panel_geometry(image_size)
    ruler_px = max(1, int(round(float(pixels_per_cm) * scale)))
    cols = int(math.ceil(math.sqrt(ruler_count)))
    rows = int(math.ceil(ruler_count / max(cols, 1)))

    panel = panel.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    white = (255, 255, 255, 255)
    shadow = (0, 0, 0, 210)
    line_width = 3 if ruler_px >= 14 else 2
    shadow_width = line_width + 2
    cap_half = 10 if ruler_px >= 14 else 8

    marker_index = 0
    for row in range(rows):
        remaining = ruler_count - marker_index
        row_count = min(cols, remaining)
        if row_count <= 0:
            break
        center_y = int(round(offset_y + fitted_h * (row + 1) / (rows + 1)))
        for col in range(row_count):
            center_x = int(round(offset_x + fitted_w * (col + 1) / (row_count + 1)))
            x0 = int(round(center_x - ruler_px / 2))
            x1 = int(round(center_x + ruler_px / 2))
            draw.line((x0, center_y, x1, center_y), fill=shadow, width=shadow_width)
            draw.line((x0, center_y, x1, center_y), fill=white, width=line_width)
            for x in (x0, x1):
                draw.line((x, center_y - cap_half, x, center_y + cap_half), fill=shadow, width=shadow_width)
                draw.line((x, center_y - cap_half, x, center_y + cap_half), fill=white, width=line_width)
            marker_index += 1
    return panel


def _build_label_map(image_size: tuple[int, int], lesion_specs: list[dict[str, Any]]) -> np.ndarray:
    width, height = image_size
    label_map = np.zeros((height, width), dtype=np.int32)
    for index, spec in enumerate(lesion_specs, start=1):
        mask = _mask_from_spec(spec, width, height)
        open_pixels = mask & (label_map == 0)
        label_map[open_pixels] = index
        if not np.any(open_pixels) and np.any(mask):
            label_map[mask] = index
    return label_map


def _volume_log_range(volumes: Mapping[int, float]) -> tuple[float, float]:
    max_volume = max([float(value) for value in volumes.values() if float(value) > 0.0], default=1.0)
    high = math.log1p(max(max_volume, 1e-9))
    return 0.0, high if high > 0.0 else 1.0


def _volume_norm(volume: float, log_range: tuple[float, float]) -> float:
    low, high = log_range
    if high <= low:
        return 0.0
    return float(np.clip((math.log1p(max(float(volume), 0.0)) - low) / (high - low), 0.0, 1.0))


def _format_volume(value: float) -> str:
    value = max(0.0, float(value))
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value >= 0.01:
        return f"{value:.2g}"
    if value > 0:
        return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return "0"


def _label_overlay(
    image: Image.Image,
    label_map: np.ndarray,
    volumes: Mapping[int, float],
    lesion_specs: list[dict[str, Any]],
    log_range: tuple[float, float],
    alpha: float,
) -> Image.Image:
    base = image.convert("RGBA")
    if label_map.size:
        max_label = int(label_map.max())
    else:
        max_label = 0
    lut = np.zeros((max_label + 1, 4), dtype=np.uint8)
    for label_id in range(1, max_label + 1):
        color = _volume_color(_volume_norm(volumes.get(label_id, 0.0), log_range))
        lut[label_id, :3] = color
        lut[label_id, 3] = int(round(255 * np.clip(alpha, 0.0, 1.0)))
    overlay = Image.fromarray(lut[label_map], mode="RGBA") if max_label else Image.new("RGBA", image.size, (0, 0, 0, 0))
    combined = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(combined, "RGBA")
    outline_width = max(2, int(round(min(image.size) / 220)))
    for index, spec in enumerate(lesion_specs, start=1):
        color = _volume_color(_volume_norm(volumes.get(index, 0.0), log_range))
        outline = (*color, 235)
        shadow = (30, 20, 36, 150)
        if spec["kind"] == "polygon":
            points = [(float(x), float(y)) for x, y in spec["polygon"]]
            if len(points) >= 2:
                closed = points + [points[0]]
                draw.line(closed, fill=shadow, width=outline_width + 2, joint="curve")
                draw.line(closed, fill=outline, width=outline_width, joint="curve")
        elif spec["kind"] == "center":
            x, y = spec["center"]
            radius = float(spec["radius_px"])
            box = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(box, outline=shadow, width=outline_width + 2)
            draw.ellipse(box, outline=outline, width=outline_width)
    return combined.convert("RGB")


def _build_heatmap_overlay_panel(
    image: Image.Image,
    label_map: np.ndarray,
    volumes: Mapping[int, float],
    lesion_specs: list[dict[str, Any]],
    log_range: tuple[float, float],
    alpha: float,
) -> Image.Image:
    return _fit_visual_panel(_label_overlay(image, label_map, volumes, lesion_specs, log_range, alpha))


def _build_surface_style_panel(
    image: Image.Image,
    depth_m: np.ndarray,
    depth_heatmap: Image.Image,
    label_map: np.ndarray,
    volumes: Mapping[int, float],
    lesion_specs: list[dict[str, Any]],
    log_range: tuple[float, float],
    alpha: float,
    yaw_angle_rad: float = 0.0,
) -> Image.Image:
    depth_rgb = depth_heatmap.resize(image.size, Image.Resampling.BILINEAR)
    shaded = Image.blend(image.convert("RGB"), depth_rgb, 0.18)
    highlighted = _label_overlay(shaded, label_map, volumes, lesion_specs, log_range, alpha)
    return _render_depth_surface_panel(highlighted, depth_m, label_map, volumes, log_range, yaw_angle_rad)


def _render_depth_surface_panel(
    texture: Image.Image,
    depth_m: np.ndarray,
    label_map: np.ndarray,
    volumes: Mapping[int, float],
    log_range: tuple[float, float],
    yaw_angle_rad: float,
) -> Image.Image:
    side = 112
    panel = Image.new("RGB", (_PANEL_W, _PANEL_H), _VISUAL_BACKGROUND)
    depth_grid = _resize_float_grid(depth_m, (side, side))
    label_grid = np.asarray(
        Image.fromarray(label_map.astype(np.int32), mode="I").resize((side, side), Image.Resampling.NEAREST),
        dtype=np.int32,
    )
    colors = np.asarray(texture.convert("RGB").resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32)

    finite = np.isfinite(depth_grid)
    if np.any(finite):
        fill_value = float(np.nanmedian(depth_grid[finite]))
    else:
        fill_value = 0.0
    depth_grid = np.nan_to_num(depth_grid.astype(np.float32), nan=fill_value, posinf=fill_value, neginf=fill_value)
    centered = depth_grid - float(np.median(depth_grid))
    scale = float(np.percentile(np.abs(centered), 95)) or 1.0
    relief = np.clip(centered / scale, -1.0, 1.0) * 0.32

    for label_id in np.unique(label_grid[label_grid > 0]):
        relief[label_grid == label_id] += 0.06 * _volume_norm(volumes.get(int(label_id), 0.0), log_range)

    image_w, image_h = texture.size
    max_dim = max(image_w, image_h)
    half_x = image_w / max_dim
    half_y = image_h / max_dim
    x_axis = np.linspace(-half_x, half_x, side, dtype=np.float32)
    y_axis = np.linspace(-half_y, half_y, side, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    cos_yaw = math.cos(float(yaw_angle_rad))
    sin_yaw = math.sin(float(yaw_angle_rad))
    x_rot = x_grid * cos_yaw + relief * sin_yaw
    z_rot = -x_grid * sin_yaw + relief * cos_yaw

    camera_distance = 3.1
    perspective = camera_distance / np.maximum(camera_distance - z_rot, 0.25)
    content_scale = min(
        (_PANEL_W * 0.88) / max(2.0 * half_x, 1e-6),
        (_PANEL_H * 0.78) / max(2.0 * half_y, 1e-6),
    )
    px = (_PANEL_W / 2.0) + x_rot * content_scale * perspective
    py = (_PANEL_H / 2.0) + y_grid * content_scale * perspective

    relief_norm = (relief - float(relief.min())) / max(float(relief.max() - relief.min()), 1e-6)
    x_light = np.linspace(0.96, 1.05, side, dtype=np.float32)[None, :]
    shade = np.clip((0.80 + 0.24 * relief_norm) * x_light, 0.66, 1.18)
    shaded_colors = np.clip(colors * shade[..., None], 0, 255).astype(np.uint8)

    z_faces = (
        z_rot[:-1, :-1]
        + z_rot[1:, :-1]
        + z_rot[:-1, 1:]
        + z_rot[1:, 1:]
    ) * 0.25
    draw = ImageDraw.Draw(panel)
    order = np.argsort(z_faces.reshape(-1))
    face_cols = side - 1
    for flat_index in order:
        row = int(flat_index // face_cols)
        col = int(flat_index % face_cols)
        polygon = (
            (float(px[row, col]), float(py[row, col])),
            (float(px[row, col + 1]), float(py[row, col + 1])),
            (float(px[row + 1, col + 1]), float(py[row + 1, col + 1])),
            (float(px[row + 1, col]), float(py[row + 1, col])),
        )
        rgb = np.mean(
            shaded_colors[row: row + 2, col: col + 2].reshape(-1, 3),
            axis=0,
        )
        draw.polygon(polygon, fill=tuple(int(value) for value in rgb))

    return panel.filter(ImageFilter.SMOOTH_MORE)


def _resize_float_grid(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32)


def _build_volume_legend(height: int, log_range: tuple[float, float]) -> Image.Image:
    legend = Image.new("RGB", (_LEGEND_W, height), _VISUAL_BACKGROUND)
    draw = ImageDraw.Draw(legend)
    title_font = _load_font(25, bold=True)
    label = f"lesion volume ({_CM3_LABEL})"
    label_box = draw.textbbox((0, 0), label, font=title_font)
    label_image = Image.new(
        "RGBA",
        (label_box[2] - label_box[0] + 8, label_box[3] - label_box[1] + 8),
        (0, 0, 0, 0),
    )
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((4 - label_box[0], 4 - label_box[1]), label, fill=(*_VISUAL_TEXT, 255), font=title_font)
    label_rotated = label_image.rotate(90, expand=True)

    bar_h = min(max(210, int(height * 0.48)), max(130, height - 118))
    bar_y0 = (height - bar_h) // 2
    bar_y1 = bar_y0 + bar_h
    bar_x0 = 62
    bar_x1 = 94
    legend.paste(label_rotated.convert("RGB"), (12, bar_y0 + (bar_h - label_rotated.height) // 2), label_rotated)

    for y in range(bar_y0, bar_y1):
        t = 1.0 - (y - bar_y0) / max(1, bar_y1 - bar_y0 - 1)
        draw.line((bar_x0, y, bar_x1, y), fill=_volume_color(t))
    draw.rectangle((bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1), outline=(60, 66, 74), width=2)

    log_min, log_max = log_range
    tick_labels = [
        (t, _format_volume(math.expm1(log_min + t * (log_max - log_min))))
        for t in (1.0, 0.5, 0.0)
    ]
    tick_size = 18
    tick_x = bar_x1 + 17
    while tick_size > 11:
        tick_font = _load_font(tick_size, bold=True)
        if all(draw.textbbox((0, 0), text, font=tick_font)[2] <= _LEGEND_W - tick_x - 2 for _, text in tick_labels):
            break
        tick_size -= 1
    tick_font = _load_font(tick_size, bold=True)

    for t, label_text in tick_labels:
        y = int(round(bar_y1 - t * (bar_y1 - bar_y0)))
        draw.line((bar_x1 + 3, y, bar_x1 + 13, y), fill=_VISUAL_TEXT, width=2)
        draw.text((tick_x, y - 10), label_text, fill=_VISUAL_TEXT, font=tick_font)
    return legend


def _draw_caption(frame: Image.Image, text: str, center_x: int, top_y: int) -> None:
    draw = ImageDraw.Draw(frame)
    font = _load_font(20, bold=True)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((center_x - width // 2, top_y + (_CAPTION_H - height) // 2 - box[1]), text, fill=_VISUAL_TEXT, font=font)


def _compose_volume_frame(
    overlay_panel: Image.Image,
    depth_panel: Image.Image,
    surface_panel: Image.Image,
    legend: Image.Image,
    total_volume_cm3: float,
) -> np.ndarray:
    row_width = _PANEL_W * 3 + _PANEL_GAP * 2
    width = row_width + _LEGEND_GAP + _LEGEND_W
    height = _PANEL_H + _CAPTION_H
    frame = Image.new("RGB", (width, height), _VISUAL_BACKGROUND)
    for index, panel in enumerate((overlay_panel, depth_panel, surface_panel)):
        frame.paste(panel, (index * (_PANEL_W + _PANEL_GAP), 0))
    frame.paste(legend, (row_width + _LEGEND_GAP, 0))
    surface_cx = 2 * (_PANEL_W + _PANEL_GAP) + _PANEL_W // 2
    _draw_caption(frame, f"total lesion volume: {_format_volume(total_volume_cm3)} {_CM3_LABEL}", surface_cx, _PANEL_H)
    return np.asarray(frame, dtype=np.uint8)


def _write_visuals(
    image: Image.Image,
    depth_m: np.ndarray,
    lesion_rows: list[dict[str, Any]],
    lesion_specs: list[dict[str, Any]],
    output_root: Path,
    image_stem: str,
    requested: set[str],
    show_progress: bool = False,
    pixels_per_cm: float = 1.0,
    ruler_count: int = 0,
) -> tuple[dict[str, str], list[str]]:
    outputs: dict[str, str] = {}
    warnings_list: list[str] = []

    label_map = _build_label_map(image.size, lesion_specs)
    volumes = {index: float(row.get("volume_cm3", 0.0)) for index, row in enumerate(lesion_rows, start=1)}
    total_volume_cm3 = float(sum(volumes.values()))
    log_range = _volume_log_range(volumes)
    legend = _build_volume_legend(_PANEL_H + _CAPTION_H, log_range)
    overlay_panel = _build_heatmap_overlay_panel(image, label_map, volumes, lesion_specs, log_range, alpha=0.42)
    depth_heatmap = _depth_heatmap(depth_m)
    depth_panel = _fit_visual_panel(depth_heatmap)
    depth_panel = _draw_visual_rulers(depth_panel, image.size, pixels_per_cm, ruler_count)

    frames: list[np.ndarray] = []
    frame_count = 72 if {"gif", "mov"} & requested else 1
    yaw_amplitude_rad = math.radians(18.0)
    progress = _make_progress(show_progress and frame_count > 1, total=frame_count, desc="Render visual frames")
    with progress as frame_bar:
        for frame_index in range(frame_count):
            yaw = yaw_amplitude_rad * math.sin(2.0 * math.pi * frame_index / frame_count)
            surface_panel = _build_surface_style_panel(
                image,
                depth_m,
                depth_heatmap,
                label_map,
                volumes,
                lesion_specs,
                log_range,
                alpha=0.48,
                yaw_angle_rad=yaw,
            )
            frames.append(_compose_volume_frame(overlay_panel, depth_panel, surface_panel, legend, total_volume_cm3))
            frame_bar.update(1)

    if "png" in requested:
        png_path = output_root / f"{image_stem}_lesion_volume.png"
        Image.fromarray(frames[0], mode="RGB").save(png_path, compress_level=1)
        outputs["png"] = str(png_path)

    if "montage" in requested:
        montage_path = output_root / f"{image_stem}_lesion_volume_montage.png"
        Image.fromarray(frames[0], mode="RGB").save(montage_path, compress_level=1)
        outputs["montage_png"] = str(montage_path)

    if "gif" in requested:
        gif_path = output_root / f"{image_stem}_lesion_volume.gif"
        _write_animation(gif_path, frames, fps=24)
        outputs["gif"] = str(gif_path)

    if "mov" in requested:
        mov_path = output_root / f"{image_stem}_lesion_volume.mov"
        try:
            _write_animation(mov_path, frames, fps=24)
            outputs["mov"] = str(mov_path)
        except Exception as exc:  # pragma: no cover - depends on local video encoders.
            warnings_list.append(f"Could not write MOV because no usable video writer was available: {exc}")

    return outputs, warnings_list


def _write_animation(path: Path, frames: list[np.ndarray], fps: int) -> None:
    suffix = path.suffix.lower()
    if suffix == ".mov":
        _write_mov(path, frames, fps)
        return

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required for GIF visual outputs.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imageio.mimsave(path, frames, fps=fps)


def _write_mov(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if writer.isOpened():
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            if path.exists() and path.stat().st_size > 0:
                return
        writer.release()
    except Exception:
        pass

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("MOV output requires opencv-python or imageio with an ffmpeg backend.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imageio.mimsave(path, frames, fps=fps)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from synthetic_nf.lesion_volume import LesionVolumePipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute lesion volume from an image, lesion outlines, and a 1 cm pixel scale.")
    parser.add_argument("--image", required=True, help="Path to the input RGB image.")
    parser.add_argument(
        "--lesions-json",
        required=True,
        help=(
            "Path to a JSON file or an inline JSON string. The JSON can be a list of lesions "
            "or an object with a 'lesions' list and optional 'scale_points'."
        ),
    )
    parser.add_argument(
        "--scale-points",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Two image coordinates that denote exactly 1 cm. Overrides scale_points in the JSON file.",
    )
    parser.add_argument("--output-dir", help="Directory for depth maps, masks, tables, and visuals.")
    parser.add_argument(
        "--visual",
        action="append",
        choices=["gif", "png", "mov", "montage"],
        default=[],
        help="Write an optional visual output. Repeat the flag for multiple outputs. 'montage' is a legacy PNG alias.",
    )
    parser.add_argument("--all-visuals", action="store_true", help="Write GIF, PNG, and MOV outputs.")
    parser.add_argument("--model-id", default="apple/DepthPro-hf", help="Hugging Face Depth Pro model id.")
    parser.add_argument("--device", default="auto", help="Depth Pro device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--no-auto-install-depthpro",
        action="store_true",
        help="Fail instead of installing missing torch/transformers dependencies.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress output.")
    args = parser.parse_args(argv)

    payload = _load_json_arg(args.lesions_json)
    if isinstance(payload, dict):
        lesions = payload.get("lesions")
        scale_points = payload.get("scale_points")
    else:
        lesions = payload
        scale_points = None

    if not lesions:
        parser.error("--lesions-json must provide at least one lesion.")

    if args.scale_points:
        scale_points = ((args.scale_points[0], args.scale_points[1]), (args.scale_points[2], args.scale_points[3]))
    if scale_points is None:
        parser.error("Scale points are required either in --scale-points or in the JSON payload as scale_points.")

    visuals = set(args.visual)
    if args.all_visuals:
        visuals = {"gif", "png", "mov"}

    pipeline = LesionVolumePipeline(
        model_id=args.model_id,
        device=args.device,
        auto_install_depthpro=not args.no_auto_install_depthpro,
    )
    result = pipeline.compute_volume(
        image_path=args.image,
        lesions=lesions,
        scale_points=scale_points,
        output_dir=args.output_dir,
        generate_visuals=bool(visuals),
        visuals=visuals,
        show_progress=not args.quiet,
    )
    json.dump(result.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _load_json_arg(value: str) -> Any:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(value)


if __name__ == "__main__":
    raise SystemExit(main())

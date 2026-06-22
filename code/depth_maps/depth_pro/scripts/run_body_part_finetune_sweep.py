#!/usr/bin/env python3
"""Run body-part combination fine-tuning experiments for Depth Pro."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nbformat
import numpy as np
from nbclient import NotebookClient
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_MANIFEST = ROOT / "data" / "depth_maps" / "body_parts" / "data" / "manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "predictions" / "depth_pro_body_part_sweep"
FINETUNE_SCRIPT = ROOT / "code" / "depth_maps" / "depth_pro" / "scripts" / "finetune_depth_pro_on_depth_maps.py"
BODY_PARTS = ["front", "back", "face", "arms", "hands", "legs", "feet"]

SCREENING_EXPERIMENTS: list[tuple[str, list[str]]] = [
    ("single_front", ["front"]),
    ("single_back", ["back"]),
    ("single_face", ["face"]),
    ("single_arms", ["arms"]),
    ("single_hands", ["hands"]),
    ("single_legs", ["legs"]),
    ("single_feet", ["feet"]),
    ("front_back", ["front", "back"]),
    ("arms_hands", ["arms", "hands"]),
    ("legs_feet", ["legs", "feet"]),
    ("hands_feet", ["hands", "feet"]),
    ("torso_face", ["front", "back", "face"]),
    ("limbs", ["arms", "hands", "legs", "feet"]),
    ("all_parts", BODY_PARTS),
]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_root_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if "body_part" not in fieldnames:
        raise ValueError(f"Manifest must include a body_part column: {path}")
    return fieldnames, rows


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_body_part(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {body_part: [] for body_part in BODY_PARTS}
    for row in rows:
        body_part = row["body_part"]
        if body_part in grouped:
            grouped[body_part].append(row)
    missing = [body_part for body_part, part_rows in grouped.items() if not part_rows]
    if missing:
        raise ValueError(f"Missing rows for body parts: {missing}")
    return grouped


def capped_rows(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    if limit is None:
        return rows
    return rows[:limit]


def split_body_part_rows(
    grouped: dict[str, list[dict[str, str]]],
    seed: int,
    max_val_samples_per_part: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    train_pool: dict[str, list[dict[str, str]]] = {}
    val_rows: dict[str, list[dict[str, str]]] = {}
    for index, body_part in enumerate(BODY_PARTS):
        part_rows = list(grouped[body_part])
        random.Random(seed + index * 1009).shuffle(part_rows)
        val_rows[body_part] = part_rows[:max_val_samples_per_part]
        train_pool[body_part] = part_rows[max_val_samples_per_part:]
    return train_pool, val_rows


def flatten(parts_to_rows: dict[str, list[dict[str, str]]], body_parts: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for body_part in body_parts:
        rows.extend(parts_to_rows[body_part])
    return rows


def build_split_manifests(
    source_manifest: Path,
    output_root: Path,
    max_train_samples_per_part: int | None,
    max_val_samples_per_part: int,
    seed: int,
    experiments: list[tuple[str, list[str]]],
) -> dict[str, Any]:
    fieldnames, rows = read_manifest(source_manifest)
    train_pool, val_by_part = split_body_part_rows(rows_by_body_part(rows), seed, max_val_samples_per_part)
    split_root = output_root / "data" / "splits"

    val_manifest = split_root / "validation_all_body_parts.csv"
    validation_rows = flatten(val_by_part, BODY_PARTS)
    write_manifest(val_manifest, fieldnames, validation_rows)

    experiment_manifests: dict[str, dict[str, Any]] = {}
    for experiment_name, body_parts in experiments:
        train_parts = {
            body_part: capped_rows(train_pool[body_part], max_train_samples_per_part)
            for body_part in body_parts
        }
        train_rows = flatten(train_parts, body_parts)
        train_manifest = split_root / f"train_{experiment_name}.csv"
        write_manifest(train_manifest, fieldnames, train_rows)
        experiment_manifests[experiment_name] = {
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "train_count": len(train_rows),
            "val_count": len(validation_rows),
            "train_counts_by_body_part": {body_part: len(train_parts[body_part]) for body_part in body_parts},
            "val_counts_by_body_part": {body_part: len(val_by_part[body_part]) for body_part in BODY_PARTS},
        }
    return {
        "fieldnames": fieldnames,
        "experiment_manifests": experiment_manifests,
        "validation_manifest": val_manifest,
        "validation_count": len(validation_rows),
    }


def experiments_from_args(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    experiments = SCREENING_EXPERIMENTS
    if args.limit_experiments is not None:
        experiments = experiments[: args.limit_experiments]
    if args.include_experiments:
        include = set(args.include_experiments)
        experiments = [experiment for experiment in experiments if experiment[0] in include]
    if not experiments:
        raise ValueError("No experiments selected")
    return experiments


def run_finetune_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    body_parts: list[str],
    train_manifest: Path,
    val_manifest: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    summary_path = experiment_root / "data" / "summary.json"
    log_path = experiment_root / "data" / "run.log"
    if summary_path.exists() and not args.overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        str(FINETUNE_SCRIPT),
        "--manifests",
        str(train_manifest),
        "--val-manifests",
        str(val_manifest),
        "--output-root",
        str(experiment_root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--smoothness-weight",
        str(args.smoothness_weight),
        "--smoothness-edge-weight",
        str(args.smoothness_edge_weight),
        "--smoothness-curvature-weight",
        str(args.smoothness_curvature_weight),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--amp-dtype",
        args.amp_dtype,
        "--trainable",
        args.trainable,
        "--target-space",
        args.target_space,
        "--viz-count",
        str(args.viz_count),
        "--overwrite",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "data").mkdir(parents=True, exist_ok=True)
    print(f"running {experiment_name}: {','.join(body_parts)}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"{experiment_name} failed with exit code {completed.returncode}\n{tail}")
    if not summary_path.exists():
        raise FileNotFoundError(f"{experiment_name} did not write summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def metric_delta(baseline: dict[str, Any], best: dict[str, Any], key: str, larger_is_better: bool = False) -> float:
    before = float(baseline[key])
    after = float(best[key])
    return after - before if larger_is_better else before - after


def optional_metric_delta(
    baseline: dict[str, Any],
    best: dict[str, Any],
    key: str,
    larger_is_better: bool = False,
) -> float | str:
    if key not in baseline or key not in best:
        return ""
    return metric_delta(baseline, best, key, larger_is_better)


def optional_float(metrics: dict[str, Any], key: str) -> float | str:
    if key not in metrics:
        return ""
    return float(metrics[key])


def summarize_experiment(
    experiment_name: str,
    body_parts: list[str],
    split_info: dict[str, Any],
    experiment_root: Path,
    summary: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = summary["baseline_val"]
    best = summary["best_val"]
    row = {
        "experiment": experiment_name,
        "train_body_parts": ",".join(body_parts),
        "train_count": split_info["train_count"],
        "val_count": split_info["val_count"],
        "output_root": repo_relative(experiment_root),
        "baseline_loss": float(baseline["loss"]),
        "best_loss": float(best["loss"]),
        "loss_reduction": metric_delta(baseline, best, "loss"),
        "baseline_data_loss": optional_float(baseline, "data_loss"),
        "best_data_loss": optional_float(best, "data_loss"),
        "data_loss_reduction": optional_metric_delta(baseline, best, "data_loss"),
        "baseline_smoothness_loss": optional_float(baseline, "smoothness_loss"),
        "best_smoothness_loss": optional_float(best, "smoothness_loss"),
        "smoothness_loss_reduction": optional_metric_delta(baseline, best, "smoothness_loss"),
        "baseline_abs_rel": float(baseline["abs_rel"]),
        "best_abs_rel": float(best["abs_rel"]),
        "abs_rel_reduction": metric_delta(baseline, best, "abs_rel"),
        "baseline_rmse": float(baseline["rmse"]),
        "best_rmse": float(best["rmse"]),
        "rmse_reduction": metric_delta(baseline, best, "rmse"),
        "baseline_delta1": float(baseline["delta1"]),
        "best_delta1": float(best["delta1"]),
        "delta1_gain": metric_delta(baseline, best, "delta1", larger_is_better=True),
        "best_adapter": summary["outputs"]["best_adapter"],
        "metrics": summary["outputs"]["metrics"],
        "metrics_by_body_part": summary["outputs"].get("metrics_by_body_part", ""),
        "gif": summary["outputs"].get("gif", ""),
        "notebook": summary["outputs"].get("notebook", ""),
    }

    body_part_rows: list[dict[str, Any]] = []
    baseline_by_part = summary.get("baseline_val_by_body_part", {})
    best_by_part = summary.get("best_val_by_body_part", {})
    for body_part in BODY_PARTS:
        if body_part not in baseline_by_part or body_part not in best_by_part:
            continue
        part_baseline = baseline_by_part[body_part]
        part_best = best_by_part[body_part]
        body_part_rows.append(
            {
                "experiment": experiment_name,
                "train_body_parts": ",".join(body_parts),
                "validation_body_part": body_part,
                "trained_on_body_part": body_part in body_parts,
                "baseline_loss": float(part_baseline["loss"]),
                "best_loss": float(part_best["loss"]),
                "loss_reduction": metric_delta(part_baseline, part_best, "loss"),
                "baseline_data_loss": optional_float(part_baseline, "data_loss"),
                "best_data_loss": optional_float(part_best, "data_loss"),
                "data_loss_reduction": optional_metric_delta(part_baseline, part_best, "data_loss"),
                "baseline_smoothness_loss": optional_float(part_baseline, "smoothness_loss"),
                "best_smoothness_loss": optional_float(part_best, "smoothness_loss"),
                "smoothness_loss_reduction": optional_metric_delta(part_baseline, part_best, "smoothness_loss"),
                "baseline_abs_rel": float(part_baseline["abs_rel"]),
                "best_abs_rel": float(part_best["abs_rel"]),
                "abs_rel_reduction": metric_delta(part_baseline, part_best, "abs_rel"),
                "baseline_delta1": float(part_baseline["delta1"]),
                "best_delta1": float(part_best["delta1"]),
                "delta1_gain": metric_delta(part_baseline, part_best, "delta1", larger_is_better=True),
            }
        )
    return row, body_part_rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_bar_frame(
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    output_size: tuple[int, int] = (1100, 720),
) -> np.ndarray:
    width, height = output_size
    image = Image.new("RGB", output_size, (248, 249, 251))
    draw = ImageDraw.Draw(image)
    sorted_rows = sorted(rows, key=lambda row: float(row[metric]), reverse=True)
    max_value = max(abs(float(row[metric])) for row in sorted_rows) or 1.0
    left = 300
    top = 72
    bar_height = 30
    gap = 12
    usable_width = width - left - 70

    draw.text((26, 24), title, fill=(18, 24, 32))
    draw.text((left, 46), "positive means fine-tuning improved over baseline", fill=(76, 86, 100))
    for index, row in enumerate(sorted_rows):
        y = top + index * (bar_height + gap)
        value = float(row[metric])
        label = row["experiment"]
        body_parts = row["train_body_parts"]
        draw.text((24, y + 7), label, fill=(18, 24, 32))
        draw.text((150, y + 7), body_parts[:22], fill=(76, 86, 100))
        bar_width = int(round((abs(value) / max_value) * usable_width))
        color = (37, 120, 94) if value >= 0 else (170, 64, 64)
        x0 = left
        if value >= 0:
            draw.rectangle((x0, y, x0 + bar_width, y + bar_height), fill=color)
        else:
            draw.rectangle((x0, y, x0 + bar_width, y + bar_height), fill=color)
        draw.text((left + usable_width + 12, y + 7), f"{value:.5f}", fill=(18, 24, 32))
    return np.asarray(image)


def save_results_gif(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        draw_bar_frame(rows, "abs_rel_reduction", "Depth Pro body-part sweep: abs-rel reduction"),
        draw_bar_frame(rows, "loss_reduction", "Depth Pro body-part sweep: validation-loss reduction"),
        draw_bar_frame(rows, "delta1_gain", "Depth Pro body-part sweep: delta1 gain"),
    ]
    imageio.mimsave(output_path, frames, duration=1.8, loop=0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    return float(value)


def notebook_relative_path(notebook_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path.resolve(), notebook_path.parent.resolve())).as_posix()


def resolve_output_asset(row: dict[str, Any], asset_name: str) -> Path:
    return resolve_root_path(row["output_root"]) / "visualizations" / asset_name


def make_top_results_markdown(rows: list[dict[str, str]], results_csv: Path, body_part_csv: Path) -> str:
    ranked = sorted(rows, key=lambda row: as_float(row["abs_rel_reduction"]), reverse=True)
    best = ranked[0]
    lines = [
        "# Depth Pro Fine-Tuning Report",
        "",
        "This executed report summarizes the body-part fine-tuning sweep, including quantitative metrics, "
        "per-body-part validation behavior, and qualitative baseline-vs-fine-tuned visualizations.",
        "",
        "## Best Run",
        "",
        f"- Best by abs-rel reduction: `{best['experiment']}` trained on `{best['train_body_parts']}`.",
        f"- Abs-rel: `{as_float(best['baseline_abs_rel']):.4f}` baseline to `{as_float(best['best_abs_rel']):.4f}` fine-tuned.",
        f"- RMSE: `{as_float(best['baseline_rmse']):.4f}` baseline to `{as_float(best['best_rmse']):.4f}` fine-tuned.",
        f"- Delta1: `{as_float(best['baseline_delta1']):.4f}` baseline to `{as_float(best['best_delta1']):.4f}` fine-tuned.",
        "",
        "## Data Files",
        "",
        f"- Experiment results CSV: `{repo_relative(results_csv)}`",
        f"- Body-part results CSV: `{repo_relative(body_part_csv)}`",
        "",
        "## Ranked Results",
        "",
        "| Rank | Experiment | Train body parts | Abs-rel reduction | Best abs-rel | Best RMSE | Best delta1 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | `{row['experiment']}` | `{row['train_body_parts']}` | "
            f"{as_float(row['abs_rel_reduction']):.4f} | "
            f"{as_float(row['best_abs_rel']):.4f} | "
            f"{as_float(row['best_rmse']):.4f} | "
            f"{as_float(row['best_delta1']):.4f} |"
        )
    return "\n".join(lines)


def make_qualitative_markdown(rows: list[dict[str, str]], output_path: Path, gif_path: Path) -> str:
    ranked = sorted(rows, key=lambda row: as_float(row["abs_rel_reduction"]), reverse=True)
    lines = [
        "## Qualitative Visualizations",
        "",
        "The panels below compare `RGB`, `GT depth`, baseline `Depth Pro`, and `Fine-tuned` predictions. "
        "Each experiment also includes an animated before/after GIF and a separate interactive surface notebook.",
        "",
        f"![Sweep ranking animation]({notebook_relative_path(output_path, gif_path)})",
        "",
    ]
    for row in ranked:
        montage = resolve_output_asset(row, "validation_montage_rgb_gt_baseline_finetuned.png")
        gif = resolve_root_path(row["gif"])
        notebook = resolve_root_path(row["notebook"])
        lines.extend(
            [
                f"### {row['experiment']}",
                "",
                f"Trained on `{row['train_body_parts']}`. "
                f"Abs-rel `{as_float(row['baseline_abs_rel']):.4f}` to `{as_float(row['best_abs_rel']):.4f}`.",
                "",
                f"![{row['experiment']} montage]({notebook_relative_path(output_path, montage)})",
                "",
                f"![{row['experiment']} animation]({notebook_relative_path(output_path, gif)})",
                "",
                f"Interactive surface notebook: [`plot_finetuned_depth_pro_surfaces.ipynb`]({notebook_relative_path(output_path, notebook)})",
                "",
            ]
        )
    return "\n".join(lines)


def notebook_cells(results_csv: Path, body_part_csv: Path, output_path: Path, gif_path: Path) -> list[nbformat.NotebookNode]:
    rows = read_rows(results_csv)
    return [
        nbformat.v4.new_markdown_cell(make_top_results_markdown(rows, results_csv, body_part_csv)),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import plotly.graph_objects as go\n"
            "from plotly.subplots import make_subplots\n\n"
            f"ROOT = Path({str(ROOT)!r})\n"
            f"RESULTS_CSV = Path({str(results_csv)!r})\n"
            f"BODY_PART_CSV = Path({str(body_part_csv)!r})\n"
            "def resolve(path_value):\n"
            "    path = Path(path_value)\n"
            "    return path if path.is_absolute() else ROOT / path\n\n"
            "def as_float(value, default=0.0):\n"
            "    return default if value in ('', None) else float(value)\n\n"
            "with RESULTS_CSV.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n"
            "with BODY_PART_CSV.open(newline='', encoding='utf-8') as handle:\n"
            "    body_rows = list(csv.DictReader(handle))\n"
            "for row in rows:\n"
            "    for key, value in list(row.items()):\n"
            "        if key not in {'experiment', 'train_body_parts', 'output_root', 'best_adapter', 'metrics', 'metrics_by_body_part', 'gif', 'notebook'}:\n"
            "            try:\n"
            "                row[key] = float(value)\n"
            "            except (TypeError, ValueError):\n"
            "                pass\n"
            "for row in body_rows:\n"
            "    for key, value in list(row.items()):\n"
            "        if key not in {'experiment', 'train_body_parts', 'validation_body_part', 'trained_on_body_part'}:\n"
            "            try:\n"
            "                row[key] = float(value)\n"
            "            except (TypeError, ValueError):\n"
            "                pass\n"
            "print(f'Loaded {len(rows)} experiments and {len(body_rows)} body-part rows')"
        ),
        nbformat.v4.new_code_cell(
            "candidate_metrics = [\n"
            "    ('abs_rel_reduction', 'Abs-rel reduction'),\n"
            "    ('rmse_reduction', 'RMSE reduction'),\n"
            "    ('loss_reduction', 'Validation-loss reduction'),\n"
            "    ('data_loss_reduction', 'Data-loss reduction'),\n"
            "    ('smoothness_loss_reduction', 'Smoothness-loss reduction'),\n"
            "    ('delta1_gain', 'Delta1 gain'),\n"
            "]\n"
            "metrics = []\n"
            "for metric, label in candidate_metrics:\n"
            "    if any(isinstance(row.get(metric), float) for row in rows):\n"
            "        metrics.append((metric, label))\n"
            "fig = go.Figure()\n"
            "for metric_index, (metric, label) in enumerate(metrics):\n"
            "    ordered = sorted(rows, key=lambda row: as_float(row.get(metric)), reverse=True)\n"
            "    fig.add_trace(\n"
            "        go.Bar(\n"
            "            x=[as_float(row.get(metric)) for row in ordered],\n"
            "            y=[row['experiment'] for row in ordered],\n"
            "            orientation='h',\n"
            "            text=[row['train_body_parts'] for row in ordered],\n"
            "            hovertemplate='%{y}<br>%{text}<br>' + label + ': %{x:.5f}<extra></extra>',\n"
            "            visible=metric_index == 0,\n"
            "        )\n"
            "    )\n"
            "buttons = []\n"
            "for metric_index, (_metric, label) in enumerate(metrics):\n"
            "    visible = [False] * len(metrics)\n"
            "    visible[metric_index] = True\n"
            "    buttons.append({'label': label, 'method': 'update', 'args': [{'visible': visible}, {'title': label}]})\n"
            "fig.update_layout(\n"
            "    title='Abs-rel reduction',\n"
            "    xaxis_title='Positive is better than baseline',\n"
            "    yaxis_title='Experiment',\n"
            "    yaxis={'autorange': 'reversed'},\n"
            "    height=620,\n"
            "    updatemenus=[{'buttons': buttons, 'direction': 'down', 'x': 0, 'y': 1.14}],\n"
            ")\n"
            "fig"
        ),
        nbformat.v4.new_code_cell(
            "fig = make_subplots(\n"
            "    rows=1,\n"
            "    cols=2,\n"
            "    subplot_titles=['Accuracy vs smoothness', 'Final validation metrics'],\n"
            ")\n"
            "fig.add_trace(\n"
            "    go.Scatter(\n"
            "        x=[as_float(row.get('best_smoothness_loss')) for row in rows],\n"
            "        y=[as_float(row.get('best_abs_rel')) for row in rows],\n"
            "        mode='markers+text',\n"
            "        text=[row['experiment'] for row in rows],\n"
            "        textposition='top center',\n"
            "        hovertemplate='%{text}<br>smoothness=%{x:.5f}<br>abs-rel=%{y:.5f}<extra></extra>',\n"
            "    ),\n"
            "    row=1,\n"
            "    col=1,\n"
            ")\n"
            "ordered = sorted(rows, key=lambda row: as_float(row.get('best_abs_rel')))\n"
            "fig.add_trace(go.Bar(x=[row['experiment'] for row in ordered], y=[as_float(row.get('best_abs_rel')) for row in ordered], name='abs-rel'), row=1, col=2)\n"
            "fig.add_trace(go.Bar(x=[row['experiment'] for row in ordered], y=[as_float(row.get('best_rmse')) for row in ordered], name='RMSE'), row=1, col=2)\n"
            "fig.update_layout(title='Fine-tuned validation quality', height=560, barmode='group')\n"
            "fig.update_xaxes(title_text='smoothness loss', row=1, col=1)\n"
            "fig.update_yaxes(title_text='best abs-rel', row=1, col=1)\n"
            "fig.update_xaxes(tickangle=-35, row=1, col=2)\n"
            "fig"
        ),
        nbformat.v4.new_code_cell(
            "experiments = [row['experiment'] for row in rows]\n"
            "parts = ['front', 'back', 'face', 'arms', 'hands', 'legs', 'feet']\n"
            "z = []\n"
            "for part in parts:\n"
            "    part_values = []\n"
            "    for experiment in experiments:\n"
            "        match = next(row for row in body_rows if row['experiment'] == experiment and row['validation_body_part'] == part)\n"
            "        part_values.append(as_float(match.get('abs_rel_reduction')))\n"
            "    z.append(part_values)\n"
            "heatmap = go.Figure(\n"
            "    go.Heatmap(\n"
            "        z=z,\n"
            "        x=experiments,\n"
            "        y=parts,\n"
            "        colorscale='RdBu',\n"
            "        zmid=0,\n"
            "        colorbar={'title': 'abs-rel reduction'},\n"
            "        hovertemplate='experiment=%{x}<br>validation part=%{y}<br>abs-rel reduction=%{z:.5f}<extra></extra>',\n"
            "    )\n"
            ")\n"
            "heatmap.update_layout(title='Per-body-part validation improvement', height=520, xaxis_tickangle=-35)\n"
            "heatmap"
        ),
        nbformat.v4.new_code_cell(
            "metric_rows = []\n"
            "for row in rows:\n"
            "    path = resolve(row['metrics'])\n"
            "    if not path.exists():\n"
            "        continue\n"
            "    with path.open(newline='', encoding='utf-8') as handle:\n"
            "        for metric_row in csv.DictReader(handle):\n"
            "            metric_row['experiment'] = row['experiment']\n"
            "            metric_rows.append(metric_row)\n"
            "fig = make_subplots(rows=1, cols=2, subplot_titles=['Validation loss over epochs', 'Validation abs-rel over epochs'])\n"
            "for experiment in [row['experiment'] for row in sorted(rows, key=lambda row: as_float(row.get('abs_rel_reduction')), reverse=True)]:\n"
            "    experiment_rows = [row for row in metric_rows if row['experiment'] == experiment and row['phase'] in {'baseline_val', 'val'}]\n"
            "    xs = [0 if row['phase'] == 'baseline_val' else int(row['epoch']) for row in experiment_rows]\n"
            "    fig.add_trace(go.Scatter(x=xs, y=[as_float(row.get('loss')) for row in experiment_rows], mode='lines+markers', name=experiment), row=1, col=1)\n"
            "    fig.add_trace(go.Scatter(x=xs, y=[as_float(row.get('abs_rel')) for row in experiment_rows], mode='lines+markers', name=experiment, showlegend=False), row=1, col=2)\n"
            "fig.update_layout(title='Training trajectories', height=560)\n"
            "fig.update_xaxes(title_text='epoch')\n"
            "fig.update_yaxes(title_text='loss', row=1, col=1)\n"
            "fig.update_yaxes(title_text='abs-rel', row=1, col=2)\n"
            "fig"
        ),
        nbformat.v4.new_markdown_cell(make_qualitative_markdown(rows, output_path, gif_path)),
    ]


def save_results_notebook(results_csv: Path, body_part_csv: Path, gif_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(cells=notebook_cells(results_csv, body_part_csv, output_path, gif_path))
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    executed = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    for cell in executed.cells:
        if cell.get("cell_type") == "code":
            cell["source"] = ""
    nbformat.write(executed, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=repo_relative(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-root", default=repo_relative(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-train-samples-per-part", type=int, default=24)
    parser.add_argument("--max-val-samples-per-part", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=50.0,
        help="Weight for edge-aware smoothness regularization inside each Depth Pro finetune.",
    )
    parser.add_argument(
        "--smoothness-edge-weight",
        type=float,
        default=8.0,
        help="RGB edge sensitivity for smoothness regularization.",
    )
    parser.add_argument(
        "--smoothness-curvature-weight",
        type=float,
        default=0.5,
        help="Relative weight of second-order depth curvature in the smoothness term.",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trainable", choices=["head", "fusion_head"], default="head")
    parser.add_argument("--target-space", choices=["metric_depth", "inverse_depth"], default="metric_depth")
    parser.add_argument("--viz-count", type=int, default=2)
    parser.add_argument("--limit-experiments", type=int, default=None)
    parser.add_argument("--include-experiments", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = resolve_root_path(args.output_root)
    source_manifest = resolve_root_path(args.source_manifest)
    experiments = experiments_from_args(args)

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    (output_root / "data" / "experiments").mkdir(parents=True, exist_ok=True)
    (output_root / "visualizations" / "gifs").mkdir(parents=True, exist_ok=True)
    (output_root / "visualizations" / "plotly").mkdir(parents=True, exist_ok=True)

    split_summary = build_split_manifests(
        source_manifest=source_manifest,
        output_root=output_root,
        max_train_samples_per_part=args.max_train_samples_per_part,
        max_val_samples_per_part=args.max_val_samples_per_part,
        seed=args.seed,
        experiments=experiments,
    )

    result_rows: list[dict[str, Any]] = []
    body_part_result_rows: list[dict[str, Any]] = []
    for experiment_name, body_parts in experiments:
        split_info = split_summary["experiment_manifests"][experiment_name]
        experiment_root = output_root / "data" / "experiments" / experiment_name
        summary = run_finetune_experiment(
            args=args,
            experiment_name=experiment_name,
            body_parts=body_parts,
            train_manifest=split_info["train_manifest"],
            val_manifest=split_info["val_manifest"],
            experiment_root=experiment_root,
        )
        result_row, part_rows = summarize_experiment(experiment_name, body_parts, split_info, experiment_root, summary)
        result_rows.append(result_row)
        body_part_result_rows.extend(part_rows)

        results_csv = output_root / "data" / "experiment_results.csv"
        body_part_csv = output_root / "data" / "experiment_body_part_results.csv"
        write_rows(results_csv, result_rows)
        write_rows(body_part_csv, body_part_result_rows)

    results_csv = output_root / "data" / "experiment_results.csv"
    body_part_csv = output_root / "data" / "experiment_body_part_results.csv"
    result_rows = sorted(result_rows, key=lambda row: float(row["abs_rel_reduction"]), reverse=True)
    write_rows(results_csv, result_rows)
    write_rows(body_part_csv, body_part_result_rows)

    gif_path = output_root / "visualizations" / "gifs" / "body_part_finetune_sweep_rankings.gif"
    notebook_path = output_root / "visualizations" / "plotly" / "body_part_finetune_sweep_results.ipynb"
    save_results_gif(result_rows, gif_path)
    save_results_notebook(results_csv, body_part_csv, gif_path, notebook_path)

    summary = {
        "source_manifest": repo_relative(source_manifest),
        "output_root": repo_relative(output_root),
        "experiment_count": len(result_rows),
        "body_parts": BODY_PARTS,
        "trainable": args.trainable,
        "target_space": args.target_space,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "smoothness_weight": args.smoothness_weight,
        "smoothness_edge_weight": args.smoothness_edge_weight,
        "smoothness_curvature_weight": args.smoothness_curvature_weight,
        "max_train_samples_per_part": args.max_train_samples_per_part,
        "max_val_samples_per_part": args.max_val_samples_per_part,
        "validation_manifest": repo_relative(split_summary["validation_manifest"]),
        "validation_count": split_summary["validation_count"],
        "best_by_abs_rel_reduction": result_rows[0],
        "outputs": {
            "results_csv": repo_relative(results_csv),
            "body_part_results_csv": repo_relative(body_part_csv),
            "gif": repo_relative(gif_path),
            "notebook": repo_relative(notebook_path),
        },
    }
    (output_root / "data" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

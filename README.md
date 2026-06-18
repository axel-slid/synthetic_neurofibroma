# Synthetic Neurofibroma

This repository is for synthetic neurofibroma data generation, HSR body-scan processing, depth-map experiments, and Depth Pro visualizations.

## Repository Layout

```text
synthetic_neurofibroma/
  code/
    data_generation/        Project scripts for generating and visualizing synthetic lesions
    depth_maps/             Scripts for generating base HSR RGB/depth pairs and plots
    depth_pro/              Minimal Depth Pro runner plus Plotly notebook visualizations
    external/               Third-party or collaborator GitHub/code drops
  data/
    depth_maps/             Generated RGB/depth examples and depth plots
    hsr/                    HSR scan inputs and HSR mesh/Plotly visualizations
    skin/                   Fitzpatrick neurofibromatosis images and manifest
    synthetic/              Synthetic sphere/gaussian lesion datasets and visualizations
  AGENTS.md                 Project-specific agent/data-folder rules
```

## Naming Conventions

Use lowercase snake case for project-owned code and data folders:

```text
code/depth_maps/
code/data_generation/sphere_generations/
data/synthetic/sphere_generations_textured_diffusion/
```

Use this convention for external GitHub repositories or other people’s code:

```text
code/external/<repo_owner>__<repo_name>/
```

Examples:

```text
code/external/apple__ml-depth-pro/
code/external/DepthAnything__Depth-Anything-V2/
code/external/facebookresearch__sam2/
```

If the code is not from a clean GitHub owner/repo source, use a clear folder name under `code/external/`, and add a README inside that folder describing where it came from.

## Current Main Components

### HSR Processing

HSR scan processing scripts are in:

```text
code/data_generation/hsr/scripts/
```

Generated HSR visualizations are in:

```text
data/hsr/visualizations/
```

### Synthetic Lesion Generation

Sphere and gaussian synthetic lesion scripts are in:

```text
code/data_generation/sphere_generations/scripts/
code/data_generation/gaussian_generations/scripts/
```

Generated datasets are in:

```text
data/synthetic/sphere_generations/
data/synthetic/sphere_generations_textured_interpolation/
data/synthetic/sphere_generations_textured_diffusion/
data/synthetic/gaussian_generations/
data/synthetic/gaussian_generations_textured_interpolation/
data/synthetic/gaussian_generations_textured_diffusion/
```

### Depth Maps

Depth-map generation and plot update scripts are in:

```text
code/depth_maps/scripts/
```

Generated depth outputs are in:

```text
data/depth_maps/
```

### Depth Pro

Depth Pro code is intentionally minimal:

```text
code/depth_pro/scripts/run_depth_pro.py
code/depth_pro/visualizations/plotly/plot_fitzpatrick_depth_surfaces.ipynb
```

Run Depth Pro on one image:

```bash
python code/depth_pro/scripts/run_depth_pro.py path/to/image.jpg
```

The Fitzpatrick notebook is already executed and contains Plotly 2D/3D Depth Pro visualizations:

```text
code/depth_pro/visualizations/plotly/plot_fitzpatrick_depth_surfaces.ipynb
```

## Data Folder Rules

For new generated datasets, use this pattern:

```text
data/<dataset_name>/
  data/              Machine-readable data, meshes, arrays, manifests, metadata
  visualizations/    GIFs, notebooks, Plotly outputs, previews
```

Project rule from `AGENTS.md`: visualization folders should not use HTML-only outputs as the main deliverable. Prefer GIFs and executed `.ipynb` notebooks with Plotly figures for interactive 3D data.

## External Code Policy

Put imported third-party or collaborator code under:

```text
code/external/
```

For every external code folder, keep a small README with:

- Source URL or person/project source
- Commit hash or download date, if available
- Install instructions
- Any local modifications
- Whether outputs from that code should be stored in this repo or regenerated elsewhere

Do not mix external source files into project-owned script folders unless they have been deliberately adapted.

## Quick Checks

Compile project Python scripts:

```bash
python -m py_compile \
  code/depth_pro/scripts/run_depth_pro.py \
  code/depth_maps/scripts/generate_base_depth_maps.py \
  code/depth_maps/scripts/update_depth_visualizations.py
```

List the current high-level tree:

```bash
find code data -maxdepth 2 -type d | sort
```

## Notes

This repo contains generated data and visual assets, so it can be large. Keep generated outputs organized under `data/` and keep reusable source code under `code/`.

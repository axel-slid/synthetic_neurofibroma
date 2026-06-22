# Data Layout

Generated data stays inside this repository under `data/`, but it is ignored by
normal Git commits because the outputs are large.

## Dataset Contract

Use this structure for every new dataset or model-output folder:

```text
data/<area>/<dataset_name>/
  data/              Machine-readable outputs: arrays, meshes, manifests, metadata
  visualizations/    Review outputs: GIFs, executed Plotly notebooks, previews
  summary.json       Dataset-level summary, when useful
```

For synthetic lesion data, lesion cardinality is the top-level split and each
split is organized by body part first, then lesion-generation method:

```text
data/synthetic/
  single_lesion/
    body_parts/<body_part>/<method>/
  multiple_lesion/
    body_parts/<body_part>/<method>/
```

The current body parts are `front`, `back`, `face`, `arms`, `hands`, `legs`,
and `feet`. Each body part should contain the same six method folders:

```text
gaussian/
gaussian_interpolation/
gaussian_diffusion/
spheres/
spheres_interpolation/
spheres_diffusion/
```

Each synthetic method folder contains the data and review outputs for that
body-part/method combination:

```text
data/synthetic/<single_lesion|multiple_lesion>/body_parts/<body_part>/<method>/
  data/settings.csv
  data/camera_depth_manifest.csv
  data/images/*_rgb.png
  data/depth/*_depth.npy
  data/depth/*_depth_mm.png
  data/depth_vis/*_depth_vis.png
  data/volumes/*_lesion_volume.ply
  summary.json
  visualization/plotly/<method>_closed_body_lesion_viewer.ipynb
  visualization/gifs/<method>_rgb_depth_preview.gif
```

Each method folder should have 1000 rows in `settings.csv`, 1000 rows in
`camera_depth_manifest.csv`, 1000 RGB images, 1000 depth arrays, 1000 depth PNGs,
and 1000 depth visualization PNGs. The Plotly notebook must be an executed
HSR-style combined dropdown viewer with a filled closed-body HSR mesh, baked
color sample overlay, and filled lesion volume meshes for that body-part region.

Do not create separate `data/synthetic/<split>/visualization/` folders. Older
method-first synthetic outputs are archived in
`data/synthetic/_legacy_pre_bodypart_restructure/`.

## Visualization Rule

Visualization folders should not use HTML files as the main deliverable. Prefer:

- GIF previews for quick review
- Executed `.ipynb` notebooks containing Plotly figures for interactive 3D review
- Small PNG previews only as supporting assets

If Plotly data or cached arrays are required by a notebook, keep them in the
dataset `data/` folder when they are machine-readable outputs, or in a clearly
named subfolder under `visualizations/` when they are only notebook cache files.

## Existing Data

Some older folders predate this contract and may still contain `plots/`,
top-level `metadata/`, or direct `images/` folders. Do not migrate large existing
datasets casually. Prefer updating scripts so future outputs use the contract,
then migrate older folders only when their manifests and scripts are updated
together.

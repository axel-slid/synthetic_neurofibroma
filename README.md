# Synthetic Neurofibroma

GitHub: <https://github.com/axel-slid/synthetic_neurofibroma>

## Install

```bash
git clone git@github.com:axel-slid/synthetic_neurofibroma.git
cd synthetic_neurofibroma
python -m pip install -e ".[volume]"
```

## Python

```python
from synthetic_nf import LesionVolumePipeline

results = LesionVolumePipeline().compute_from_csv(
    "sample_data/lesion_volume_sample/sample_annotations.csv",
    image_root="sample_data/lesion_volume_sample",
    output_dir="outputs/sample_volume",
    generate_visuals=True,
    visuals={"gif", "png", "mov"},
)

print(results[0].total_volume_cm3)
```

## CLI

```bash
synthetic-nf-volume \
  --image sample_data/lesion_volume_sample/sample_lesions.png \
  --lesions-json sample_data/lesion_volume_sample/lesions.json \
  --output-dir outputs/sample_volume \
  --visual gif \
  --visual png \
  --visual mov
```

## CSV Sample

```bash
python examples/run_sample_table.py
```

```text
image_path,ai_cnf_points,ai_cnf_contours,sensitivity_cnf_points,ai_cnf_stage,ruler_location,ruler_distance_cm,lesion_id
```

`sample_annotations.csv` has 519 real lesion rows for the sample image; the ruler scale is fake.

## Outputs

```text
outputs/sample_volume/
  lesion_volumes_cm3.csv
  summary.json
  sample_lesions_depth_m.npy
  masks/
  visualizations/
    sample_lesions_depth.png
    sample_lesions_lesion_volume.gif
    sample_lesions_lesion_volume.mov
    sample_lesions_lesion_volume.png
```

![Lesion volume heatmap montage](code/pipeline/visualizations/plots/added_polygons_largest_diameter_3cm_per_image_heatmaps_exports/per_quadrant_cm3_new_stacked_gif_all_five_patients.gif)

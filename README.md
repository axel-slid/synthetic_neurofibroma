# Synthetic Neurofibroma

![Lesion volume heatmap montage](code/pipeline/visualizations/plots/added_polygons_largest_diameter_3cm_per_image_heatmaps_exports/per_quadrant_cm3_new_stacked_gif_all_five_patients.gif)

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

result = LesionVolumePipeline().compute_volume(
    image_path="sample_data/lesion_volume_sample/sample_lesions.png",
    lesions=[
        {"points": [[247, 184], [241, 205], [226, 220], [205, 226], [184, 220], [169, 205], [163, 184], [169, 163], [184, 148], [205, 142], [226, 148], [241, 163]]},
        {"points": [[357, 230], [353, 246], [342, 257], [326, 261], [310, 257], [299, 246], [295, 230], [299, 214], [310, 203], [326, 199], [342, 203], [353, 214]]},
    ],
    scale_points=((48, 48), (128, 48)),
    generate_visuals=True,
)

print(result.total_volume_cm3)
```

## CLI

```bash
synthetic-nf-volume \
  --image sample_data/lesion_volume_sample/sample_lesions.png \
  --lesions-json sample_data/lesion_volume_sample/lesions.json \
  --output-dir outputs/sample_volume \
  --visual gif \
  --visual montage
```

## Outputs

```text
outputs/sample_volume/
  lesion_volumes_cm3.csv
  summary.json
  sample_lesions_depth_m.npy
  sample_lesions_depth.png
  masks/
  sample_lesions_lesion_volume.gif
  sample_lesions_lesion_volume_montage.png
```

![Lesion volume heatmap montage](code/pipeline/visualizations/plots/added_polygons_largest_diameter_3cm_per_image_heatmaps_exports/per_quadrant_cm3_new_stacked_gif_all_five_patients.gif)

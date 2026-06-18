# TODO

Status key:

- `[x]` Complete enough to use now
- `[~]` In progress / partially implemented
- `[ ]` Not started or not yet in this repo

## Synthetic Lesion Geometry

- [x] Add spherical-cap lesions to HSR scans.
  Sphere-cap placement metadata exists for HSR scans under `data/synthetic/sphere_generations/metadata`, with generation/visualization code under `code/data_generation/sphere_generations/scripts`.

- [x] Add gaussian-bump lesions to HSR scans.
  Gaussian lesion metadata exists under `data/synthetic/gaussian_generations/metadata`, with scripts under `code/data_generation/gaussian_generations/scripts`.

- [~] Make lesion morphology statistically and physically grounded.
  Current sphere/gaussian lesions are useful geometric prototypes. Next step is to parameterize location, radius, height, shape class, and growth using NF literature and real annotated data rather than hand-picked synthetic settings.

- [ ] Use NF paper findings to implement realistic lesion locations and growth.
  Extract anatomical distributions, growth rates, age/visit effects, and lesion-type frequencies from relevant NF studies, then convert them into sampling rules for synthetic generation.

- [ ] Use NF paper physics to define synthetic morphology.
  Translate lesion class definitions into geometry: flat, sessile, globular, and pedunculated. Define plausible height/diameter/neck/base relationships and deformation constraints.

## Ground-Truth Depth and Scale

- [x] Create base HSR GT depth maps.
  Base RGB/depth arrays and visualizations exist under `data/depth_maps/base` and `data/depth_maps/plots`, with scripts in `code/depth_maps/scripts`.

- [~] Create GT depth maps for synthetic lesions.
  HSR base depth maps exist, and synthetic 3D lesion geometry exists. The next step is to export paired synthetic RGB/depth/mask data for every generated lesion with explicit lesion-only protrusion depth.

- [ ] Define exact Depth Pro fine-tuning target.
  Decide whether the target should be absolute camera depth, local protrusion depth above skin baseline, normalized relative depth, or a multi-head target including mask/height/volume.

- [ ] Fine-tune Depth Pro on desired volume range.
  Build a training set with controlled lesion volumes and run a Depth Pro adapter/full fine-tune. Track performance by volume range, morphology class, and camera setting.

- [~] Use pixel/cm values to scale area, surface area, and volume calculations.
  External distance/regression code exists under `code/external/distance_model`, but it still needs to be connected to the lesion measurement pipeline and validated.

- [~] Obtain old skin distance regression model.
  Placeholder external folder exists at `code/external/distance_model`. Confirm source, weights, expected inputs, and output units.

## Texture, Coloring, and Rendering

- [x] Add texture by sampling/pulling colors from HSR texture maps.
  Interpolated texture datasets exist for sphere and gaussian generations under `data/synthetic/*_textured_interpolation`.

- [x] Implement interpolation-based lesion coloring.
  Interpolated OBJ/MTL outputs exist for synthetic sphere and gaussian lesions.

- [x] Implement diffusion-based lesion texturing.
  Text-conditioned diffusion textured datasets exist under `data/synthetic/sphere_generations_textured_diffusion` and `data/synthetic/gaussian_generations_textured_diffusion`.

- [x] Generate colored 3D versions.
  Colored OBJ/MTL outputs and texture assets exist for interpolation/diffusion textured synthetic lesions.

- [~] Generate colored 2D versions.
  HSR and synthetic visualization notebooks/GIFs exist, but a systematic 2D rendered image dataset for each colored 3D lesion still needs to be finalized.

- [~] Generate colored 2D versions with varied camera settings.
  Depth-map views exist for HSR, but synthetic lesion rendering still needs a controlled camera sweep over distance, focal length, pose, lighting, and crop.

## Annotated NF Image Data

- [~] Obtain all annotated NF images.
  Fitzpatrick neurofibromatosis images and manifest exist under `data/skin/fitzpatrick`; more annotated datasets should be added if available.

- [~] Organize already available NF images.
  Local Fitzpatrick NF image data exists, but dataset provenance, annotation completeness, and intended train/validation/test splits should be documented.

- [ ] Obtain lesion center locations `(x, y)`.
  Need point annotations or locator model outputs for lesion centers in NF images.

- [ ] Obtain lesion boundaries.
  Need segmentation masks or outlines to compute 2D area and support volume estimation.

- [ ] Classify lesions as flat, sessile, globular, or pedunculated.
  Need labels from annotation, model prediction, dermatologist review, or rules based on morphology.

- [ ] Decide realism metrics based on area and classification.
  Define quantitative realism checks: area distribution, lesion count per image/body region, morphology class proportions, color/texture distribution, and growth over visits.

## Literature and Clinical Review

- [ ] Consult literature.
  Collect quantitative distributions for cNF lesion size, shape class, anatomical location, growth, color, and volume where available.

- [ ] Consult dermatologist.
  Review synthetic outputs, class definitions, realism criteria, and failure modes with a domain expert.

## 3D Validation Data

- [ ] Obtain 3D validation data for above-the-skin NF.
  Target paired 2D images and 3D geometry over multiple visits if possible.

- [ ] Investigate SPECTRA scans.
  Determine access, export format, resolution, calibration, and whether cNF surfaces can be segmented.

- [ ] Investigate MRI validation data.
  Determine whether MRI is relevant for visible cutaneous lesions, whether volume labels exist, and how it aligns with 2D photos.

- [ ] Investigate ultrasound validation data.
  Determine whether ultrasound can provide lesion height/volume labels for above-skin lesions and how to register to images.

- [ ] Segment and scale NF volumes in validation data.
  Build a workflow to isolate lesion volumes, convert units, and store masks/meshes/measurements.

- [ ] Run the depth-estimation-to-volume pipeline.
  Use image input, lesion mask, scale estimate, depth map, local baseline fit, and volume integration.

- [ ] Compare predicted and reference volumes.
  Report error by lesion class, size, body region, imaging condition, and visit/timepoint.

## Heuristic Measurement Pipeline

- [~] Obtain locator model `(x, y)`.
  External locator code folder exists at `code/external/nf_locator`; verify model source, weights, and inference script.

- [~] Obtain outline model, likely SAM 2 or equivalent.
  External segmentation folder exists at `code/external/nf_segmenter`; connect it to locator outputs and NF images.

- [~] Obtain classification model.
  External classifier folder exists at `code/external/nf_classifier`; verify labels, weights, and expected input crops.

- [ ] Combine locator, outline, and classification outputs.
  Produce a single per-lesion table with image id, center, mask path, area, morphology class, confidence scores, and scale.

- [ ] Use heuristics to predict volume.
  Define formulas by morphology class, using area, estimated diameter, shape assumptions, and optional depth cues.

- [ ] Use pixels/cm model to scale volume/surface area/area.
  Convert image-space measurements into physical units and propagate scale uncertainty.

- [ ] Compare heuristic and depth-based results.
  Benchmark heuristic volume estimates against Depth Pro-based estimates and any available 3D validation labels.

- [ ] Make poster.
  Summarize data sources, synthetic generation, Depth Pro results, heuristic pipeline, validation comparisons, and limitations.

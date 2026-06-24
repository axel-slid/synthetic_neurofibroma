from __future__ import annotations

from pathlib import Path

from synthetic_nf import LesionVolumePipeline


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data" / "lesion_volume_sample"


def main() -> None:
    results = LesionVolumePipeline().compute_from_table(
        annotations_csv=SAMPLE_DIR / "sample_annotations.csv",
        image_root=SAMPLE_DIR,
        output_dir=ROOT / "outputs" / "sample_table",
        generate_visuals=True,
        visuals={"gif", "montage"},
    )
    for result in results:
        print(f"{Path(result.image_path).name}: {result.total_volume_cm3:.4f} cm^3")
        print(result.output_dir)


if __name__ == "__main__":
    main()

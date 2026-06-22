from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_nf.paths import DATA_ROOT


@dataclass(frozen=True)
class DatasetRoot:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def visualizations(self) -> Path:
        return self.root / "visualizations"

    @property
    def manifest_csv(self) -> Path:
        return self.data / "manifest.csv"

    @property
    def summary_json(self) -> Path:
        return self.root / "summary.json"

    def ensure(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.visualizations.mkdir(parents=True, exist_ok=True)


def dataset_root(*parts: str) -> DatasetRoot:
    return DatasetRoot(DATA_ROOT.joinpath(*parts))

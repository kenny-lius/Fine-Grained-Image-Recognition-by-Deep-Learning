"""Configuration objects for datasets and training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DatasetName = Literal["cub", "butterfly"]


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    display_name: str
    image_directory: Path
    metadata_file: Path
    level_names: tuple[str, str, str, str]
    default_image_size: int
    default_batch_size: int
    default_epochs: int
    default_learning_rate: float

    def resolve(self, data_root: Path) -> "ResolvedDatasetSpec":
        return ResolvedDatasetSpec(
            spec=self,
            image_directory=data_root / self.image_directory,
            metadata_file=data_root / self.metadata_file,
        )


@dataclass(frozen=True)
class ResolvedDatasetSpec:
    spec: DatasetSpec
    image_directory: Path
    metadata_file: Path


DATASETS: Final[dict[DatasetName, DatasetSpec]] = {
    "cub": DatasetSpec(
        name="cub",
        display_name="CUB-200-2011",
        image_directory=Path("CUB_200_2011/CUB_200_2011/images"),
        metadata_file=Path(
            "CUB_200_2011/CUB_200_2011/"
            "CUB_200_2011_train_test_multi_level_info.pkl"
        ),
        level_names=("order", "family", "genus", "species"),
        default_image_size=448,
        default_batch_size=32,
        default_epochs=20,
        default_learning_rate=0.005,
    ),
    "butterfly": DatasetSpec(
        name="butterfly",
        display_name="Butterfly-200",
        image_directory=Path("Butterfly200/Butterfly200/images"),
        metadata_file=Path(
            "Butterfly200/Butterfly200/Butterfly_train_test_multi_level_info.pkl"
        ),
        level_names=("family", "subfamily", "genus", "species"),
        default_image_size=224,
        default_batch_size=64,
        default_epochs=90,
        default_learning_rate=0.01,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]  # type: ignore[index]
    except KeyError as error:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset {name!r}. Choose one of: {choices}."
        ) from error

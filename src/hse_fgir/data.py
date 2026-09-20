"""Dataset loading and image preprocessing."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import ResolvedDatasetSpec

Split = Literal["train", "test"]


@dataclass(frozen=True)
class HierarchyMetadata:
    train_files: Sequence[str]
    train_labels: Sequence[Sequence[int]]
    test_files: Sequence[str]
    test_labels: Sequence[Sequence[int]]
    class_counts: tuple[int, int, int, int]
    auxiliary: tuple[Any, Any, Any, Any]
    root_to_level_1: np.ndarray
    level_1_to_2: np.ndarray
    level_2_to_3: np.ndarray
    level_3_to_4: np.ndarray


def load_hierarchy_metadata(path: Path) -> HierarchyMetadata:
    if not path.is_file():
        raise FileNotFoundError(
            f"Hierarchy metadata was not found at {path}. "
            "See the data layout in README.md."
        )

    with path.open("rb") as stream:
        values = []
        try:
            for _ in range(13):
                values.append(pickle.load(stream))
        except EOFError as error:
            raise ValueError(
                f"{path} does not contain the expected 13-object pickle stream."
            ) from error

    train_files, train_labels, test_files, test_labels = values[:4]
    class_counts = tuple(int(value) for value in values[4])
    if len(class_counts) != 4:
        raise ValueError(f"Expected four hierarchy levels, got {class_counts!r}.")
    if len(train_files) != len(train_labels):
        raise ValueError("Training file and label counts do not match.")
    if len(test_files) != len(test_labels):
        raise ValueError("Test file and label counts do not match.")

    transition_matrices = tuple(np.asarray(value) for value in values[9:13])
    expected_shapes = (
        (class_counts[0], class_counts[1]),
        (class_counts[1], class_counts[2]),
        (class_counts[2], class_counts[3]),
    )
    for matrix, expected_shape in zip(
        transition_matrices[1:], expected_shapes, strict=True
    ):
        if matrix.shape != expected_shape:
            raise ValueError(
                "Expected transition matrix shape "
                f"{expected_shape}, got {matrix.shape}."
            )

    return HierarchyMetadata(
        train_files=train_files,
        train_labels=train_labels,
        test_files=test_files,
        test_labels=test_labels,
        class_counts=class_counts,  # type: ignore[arg-type]
        auxiliary=tuple(values[5:9]),  # type: ignore[arg-type]
        root_to_level_1=transition_matrices[0],
        level_1_to_2=transition_matrices[1],
        level_2_to_3=transition_matrices[2],
        level_3_to_4=transition_matrices[3],
    )


def build_transform(image_size: int, *, training: bool) -> transforms.Compose:

    operations: list[Any]
    if training:
        operations = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        operations = [
            transforms.Resize(round(image_size * 8 / 7)),
            transforms.CenterCrop(image_size),
        ]
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class FineGrainedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: ResolvedDatasetSpec,
        metadata: HierarchyMetadata,
        split: Split,
        image_size: int,
    ) -> None:
        self.image_directory = dataset.image_directory
        if not self.image_directory.is_dir():
            raise FileNotFoundError(
                f"Image directory was not found at {self.image_directory}."
            )
        self.transform = build_transform(image_size, training=split == "train")
        if split == "train":
            self.files = metadata.train_files
            self.labels = metadata.train_labels
        else:
            self.files = metadata.test_files
            self.labels = metadata.test_labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_directory / self.files[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        labels = torch.as_tensor(self.labels[index], dtype=torch.long)
        if labels.shape != (4,):
            raise ValueError(
                "Expected four labels for "
                f"{image_path}, got shape {tuple(labels.shape)}."
            )
        return image_tensor, labels


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

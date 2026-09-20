"""Utilities for projecting predictions through a class hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import HierarchyMetadata


@dataclass(frozen=True)
class FinestLevelMappings:

    level_1: torch.Tensor
    level_2: torch.Tensor
    level_3: torch.Tensor

    @classmethod
    def from_metadata(cls, metadata: HierarchyMetadata) -> "FinestLevelMappings":
        level_1_to_2 = torch.as_tensor(metadata.level_1_to_2, dtype=torch.float32)
        level_2_to_3 = torch.as_tensor(metadata.level_2_to_3, dtype=torch.float32)
        level_3_to_4 = torch.as_tensor(metadata.level_3_to_4, dtype=torch.float32)
        return cls(
            level_1=level_1_to_2 @ level_2_to_3 @ level_3_to_4,
            level_2=level_2_to_3 @ level_3_to_4,
            level_3=level_3_to_4,
        )

    def to(self, device: torch.device) -> "FinestLevelMappings":
        return FinestLevelMappings(
            level_1=self.level_1.to(device),
            level_2=self.level_2.to(device),
            level_3=self.level_3.to(device),
        )


def fuse_predictions(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    mappings: FinestLevelMappings,
) -> torch.Tensor:

    probabilities = tuple(output.softmax(dim=1) for output in logits)
    return (
        probabilities[0] @ mappings.level_1
        + probabilities[1] @ mappings.level_2
        + probabilities[2] @ mappings.level_3
        + probabilities[3]
    )

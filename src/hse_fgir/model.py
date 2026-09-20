from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class HierarchicalResNet50(nn.Module):
    """A shared ResNet-50 trunk followed by one branch per hierarchy level.

    The first three branches consume a detached trunk representation, matching
    the behavior of the original project. Consequently, the finest-level loss
    is the only loss that updates the shared trunk.
    """

    def __init__(
        self,
        class_counts: Sequence[int],
        *,
        pretrained: bool = True,
        detach_auxiliary_branches: bool = True,
    ) -> None:
        super().__init__()
        if len(class_counts) != 4:
            raise ValueError("HierarchicalResNet50 requires exactly four class counts.")

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.trunk = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.branches = nn.ModuleList(copy.deepcopy(backbone.layer4) for _ in range(4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifiers = nn.ModuleList(
            nn.Linear(backbone.fc.in_features, count) for count in class_counts
        )
        self.detach_auxiliary_branches = detach_auxiliary_branches

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_features = self.trunk(images)
        outputs: list[torch.Tensor] = []
        for index, (branch, classifier) in enumerate(
            zip(self.branches, self.classifiers, strict=True)
        ):
            branch_input = shared_features
            if self.detach_auxiliary_branches and index < len(self.branches) - 1:
                branch_input = branch_input.detach()
            features = self.pool(branch(branch_input)).flatten(start_dim=1)
            outputs.append(classifier(features))
        return tuple(outputs)  # type: ignore[return-value]

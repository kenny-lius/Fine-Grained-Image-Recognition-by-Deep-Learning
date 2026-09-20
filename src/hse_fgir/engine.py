"""Training, evaluation, checkpointing, and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .hierarchy import FinestLevelMappings, fuse_predictions


@dataclass(frozen=True)
class EvaluationMetrics:
    level_accuracy: tuple[float, float, float, float]
    fused_accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:

    model.train()
    total_loss = 0.0
    sample_count = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = sum(
            criterion(output, labels[:, index])
            for index, output in enumerate(outputs)
        )
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("The training dataset is empty.")
    return total_loss / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    mappings: FinestLevelMappings,
    device: torch.device,
) -> EvaluationMetrics:

    model.eval()
    correct = [0, 0, 0, 0]
    fused_correct = 0
    sample_count = 0
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        for index, output in enumerate(outputs):
            correct[index] += (output.argmax(dim=1) == labels[:, index]).sum().item()
        fused = fuse_predictions(outputs, mappings)
        fused_correct += (fused.argmax(dim=1) == labels[:, -1]).sum().item()
        sample_count += images.shape[0]

    if sample_count == 0:
        raise ValueError("The evaluation dataset is empty.")
    return EvaluationMetrics(
        level_accuracy=tuple(  # type: ignore[arg-type]
            value / sample_count for value in correct
        ),
        fused_accuracy=fused_correct / sample_count,
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    configuration: dict[str, Any],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    state_dict = {
        name: value.detach().cpu()
        for name, value in unwrap_model(model).state_dict().items()
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "configuration": configuration,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def append_metrics(
    path: Path,
    *,
    epoch: int,
    learning_rate: float,
    training_loss: float,
    evaluation: EvaluationMetrics,
) -> None:

    record = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "training_loss": training_loss,
        **asdict(evaluation),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")

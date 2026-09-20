"""Command-line training entry point."""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import DATASETS, DatasetSpec, get_dataset_spec
from .data import FineGrainedDataset, load_hierarchy_metadata, seed_worker
from .engine import (
    append_metrics,
    evaluate,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from .hierarchy import FinestLevelMappings
from .model import HierarchicalResNet50

LOGGER = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical multi-branch ResNet-50 model."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cub")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        help="Cosine schedule floor (default: min(1e-4, learning_rate * 0.01)).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available by default.",
    )
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        help="CUDA device IDs. Multiple IDs enable torch.nn.DataParallel.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the ResNet-50 backbone from ImageNet weights.",
    )
    parser.add_argument(
        "--detach-auxiliary-branches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prevent the first three hierarchy losses from updating the trunk.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a checkpoint produced by this refactored trainer.",
    )
    return parser


def select_device(requested: str, device_ids: list[int] | None) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto" and not torch.cuda.is_available():
        return torch.device("cpu")
    primary_device = device_ids[0] if device_ids else 0
    return torch.device(f"cuda:{primary_device}")


def resolve_hyperparameters(
    args: argparse.Namespace, spec: DatasetSpec
) -> dict[str, int | float]:
    values: dict[str, int | float] = {
        "epochs": args.epochs if args.epochs is not None else spec.default_epochs,
        "batch_size": (
            args.batch_size
            if args.batch_size is not None
            else spec.default_batch_size
        ),
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else spec.default_learning_rate
        ),
        "image_size": (
            args.image_size
            if args.image_size is not None
            else spec.default_image_size
        ),
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be greater than zero.")
    return values


def validate_arguments(
    args: argparse.Namespace,
    hyperparameters: dict[str, int | float],
    minimum_learning_rate: float,
) -> None:
    if args.workers < 0:
        raise ValueError("workers cannot be negative.")
    if minimum_learning_rate < 0:
        raise ValueError("minimum learning rate cannot be negative.")
    if minimum_learning_rate > float(hyperparameters["learning_rate"]):
        raise ValueError("minimum learning rate cannot exceed the learning rate.")
    if args.device_ids:
        if any(device_id < 0 for device_id in args.device_ids):
            raise ValueError("CUDA device IDs cannot be negative.")
        if len(set(args.device_ids)) != len(args.device_ids):
            raise ValueError("CUDA device IDs must be unique.")


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def configuration_record(
    args: argparse.Namespace,
    hyperparameters: dict[str, int | float],
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "workers": args.workers,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "detach_auxiliary_branches": args.detach_auxiliary_branches,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        **hyperparameters,
    }


def run(args: argparse.Namespace) -> Path:
    spec = get_dataset_spec(args.dataset)
    resolved_dataset = spec.resolve(args.data_root)
    hyperparameters = resolve_hyperparameters(args, spec)
    minimum_learning_rate = (
        args.minimum_learning_rate
        if args.minimum_learning_rate is not None
        else min(1e-4, float(hyperparameters["learning_rate"]) * 0.01)
    )
    validate_arguments(args, hyperparameters, minimum_learning_rate)
    device = select_device(args.device, args.device_ids)
    seed_everything(args.seed)

    metadata = load_hierarchy_metadata(resolved_dataset.metadata_file)
    training_dataset = FineGrainedDataset(
        resolved_dataset,
        metadata,
        "train",
        int(hyperparameters["image_size"]),
    )
    evaluation_dataset = FineGrainedDataset(
        resolved_dataset,
        metadata,
        "test",
        int(hyperparameters["image_size"]),
    )

    loader_options: dict[str, Any] = {
        "batch_size": int(hyperparameters["batch_size"]),
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        shuffle=False,
        **loader_options,
    )

    model: nn.Module = HierarchicalResNet50(
        metadata.class_counts,
        pretrained=args.pretrained,
        detach_auxiliary_branches=args.detach_auxiliary_branches,
    ).to(device)
    if device.type == "cuda" and args.device_ids and len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    optimizer = SGD(
        model.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(hyperparameters["epochs"]),
        eta_min=minimum_learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    mappings = FinestLevelMappings.from_metadata(metadata).to(device)

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > int(hyperparameters["epochs"]):
            raise ValueError(
                "The resume checkpoint has already reached the requested epoch count."
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_directory = args.output_root / spec.name / timestamp
    experiment_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(resolved_dataset.metadata_file, experiment_directory / "metadata.pkl")
    metrics_path = experiment_directory / "metrics.jsonl"
    configuration = configuration_record(args, hyperparameters)
    configuration["minimum_learning_rate"] = minimum_learning_rate

    LOGGER.info(
        "Training %s on %s with %d training and %d evaluation images.",
        spec.display_name,
        device,
        len(training_dataset),
        len(evaluation_dataset),
    )

    for epoch in range(start_epoch, int(hyperparameters["epochs"]) + 1):
        training_loss = train_one_epoch(
            model, training_loader, optimizer, criterion, device
        )
        evaluation = evaluate(model, evaluation_loader, mappings, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        append_metrics(
            metrics_path,
            epoch=epoch,
            learning_rate=learning_rate,
            training_loss=training_loss,
            evaluation=evaluation,
        )

        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "configuration": configuration,
        }
        save_checkpoint(experiment_directory / "last-model.pth", **checkpoint_arguments)

        percentages = [100 * value for value in evaluation.level_accuracy]
        LOGGER.info(
            "Epoch %d/%d | loss %.4f | levels %s | fused %.2f%%",
            epoch,
            hyperparameters["epochs"],
            training_loss,
            ", ".join(f"{value:.2f}%" for value in percentages),
            100 * evaluation.fused_accuracy,
        )

    return experiment_directory


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

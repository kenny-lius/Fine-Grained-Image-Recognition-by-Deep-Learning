"""Command-line inference for trained hierarchical classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .config import DATASETS, get_dataset_spec
from .data import build_transform, load_hierarchy_metadata
from .hierarchy import FinestLevelMappings, fuse_predictions
from .model import HierarchicalResNet50
from .train import select_device

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict all hierarchy levels for one or more images."
    )
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cub")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def discover_images(inputs: list[Path]) -> list[Path]:
    images: set[Path] = set()
    for path in inputs:
        if path.is_dir():
            images.update(
                candidate
                for candidate in path.iterdir()
                if candidate.is_file()
                and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            images.add(path)
        else:
            raise FileNotFoundError(f"No supported image found at {path}.")
    return sorted(images)


@torch.inference_mode()
def run(args: argparse.Namespace) -> list[dict[str, object]]:
    spec = get_dataset_spec(args.dataset)
    resolved_dataset = spec.resolve(args.data_root)
    metadata = load_hierarchy_metadata(resolved_dataset.metadata_file)
    device = select_device(args.device, None)
    image_size = args.image_size or spec.default_image_size

    model = HierarchicalResNet50(metadata.class_counts, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    mappings = FinestLevelMappings.from_metadata(metadata).to(device)
    transform = build_transform(image_size, training=False)

    results: list[dict[str, object]] = []
    for image_path in discover_images(args.images):
        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        outputs = model(image_tensor)
        fused = fuse_predictions(outputs, mappings)
        predictions = {
            level_name: int(output.argmax(dim=1).item())
            for level_name, output in zip(spec.level_names, outputs, strict=True)
        }
        predictions["fused_species"] = int(fused.argmax(dim=1).item())
        results.append({"image": str(image_path), "predictions": predictions})
    return results


def main() -> None:
    args = build_parser().parse_args()
    for result in run(args):
        print(json.dumps(result))


if __name__ == "__main__":
    main()

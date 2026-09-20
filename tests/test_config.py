from argparse import Namespace
from unittest import TestCase

from hse_fgir.config import DATASETS, get_dataset_spec
from hse_fgir.train import resolve_hyperparameters


class ConfigTests(TestCase):
    def test_dataset_hierarchy_names_are_complete(self) -> None:
        self.assertEqual(
            DATASETS["cub"].level_names,
            ("order", "family", "genus", "species"),
        )
        self.assertEqual(
            DATASETS["butterfly"].level_names,
            ("family", "subfamily", "genus", "species"),
        )

    def test_unknown_dataset_has_helpful_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            get_dataset_spec("unknown")

    def test_dataset_defaults_are_resolved_at_runtime(self) -> None:
        args = Namespace(
            epochs=None,
            batch_size=None,
            learning_rate=None,
            image_size=None,
        )
        values = resolve_hyperparameters(args, DATASETS["cub"])
        self.assertEqual(
            values,
            {
                "epochs": 20,
                "batch_size": 32,
                "learning_rate": 0.005,
                "image_size": 448,
            },
        )

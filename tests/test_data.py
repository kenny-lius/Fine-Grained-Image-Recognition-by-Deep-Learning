import pickle
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from hse_fgir.data import load_hierarchy_metadata


def write_metadata(path: Path, matrices: list[np.ndarray]) -> None:
    values = [
        ["train.jpg"],
        [[0, 0, 0, 0]],
        ["test.jpg"],
        [[1, 1, 1, 1]],
        [2, 3, 4, 5],
        None,
        None,
        None,
        None,
        *matrices,
    ]
    with path.open("wb") as stream:
        for value in values:
            pickle.dump(value, stream)


def valid_matrices() -> list[np.ndarray]:
    return [
        np.ones((1, 2)),
        np.ones((2, 3)),
        np.ones((3, 4)),
        np.ones((4, 5)),
    ]


class DataTests(TestCase):
    def test_load_hierarchy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.pkl"
            write_metadata(path, valid_matrices())

            metadata = load_hierarchy_metadata(path)

        self.assertEqual(metadata.class_counts, (2, 3, 4, 5))
        self.assertEqual(metadata.level_2_to_3.shape, (3, 4))

    def test_rejects_invalid_transition_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.pkl"
            matrices = valid_matrices()
            matrices[-1] = np.ones((3, 5))
            write_metadata(path, matrices)

            with self.assertRaisesRegex(ValueError, "transition matrix shape"):
                load_hierarchy_metadata(path)

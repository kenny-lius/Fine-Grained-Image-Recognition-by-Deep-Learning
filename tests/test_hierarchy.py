from unittest import TestCase

import numpy as np
import torch

from hse_fgir.data import HierarchyMetadata
from hse_fgir.hierarchy import FinestLevelMappings, fuse_predictions


def metadata() -> HierarchyMetadata:
    return HierarchyMetadata(
        train_files=[],
        train_labels=[],
        test_files=[],
        test_labels=[],
        class_counts=(2, 3, 4, 5),
        auxiliary=(None, None, None, None),
        root_to_level_1=np.ones((1, 2)),
        level_1_to_2=np.ones((2, 3)),
        level_2_to_3=np.ones((3, 4)),
        level_3_to_4=np.ones((4, 5)),
    )


class HierarchyTests(TestCase):
    def test_mapping_composition_shapes(self) -> None:
        mappings = FinestLevelMappings.from_metadata(metadata())
        self.assertEqual(mappings.level_1.shape, (2, 5))
        self.assertEqual(mappings.level_2.shape, (3, 5))
        self.assertEqual(mappings.level_3.shape, (4, 5))

    def test_fused_predictions_use_finest_label_space(self) -> None:
        mappings = FinestLevelMappings.from_metadata(metadata())
        logits = (
            torch.zeros(2, 2),
            torch.zeros(2, 3),
            torch.zeros(2, 4),
            torch.zeros(2, 5),
        )
        fused = fuse_predictions(logits, mappings)
        self.assertEqual(fused.shape, (2, 5))
        self.assertTrue(torch.allclose(fused[:, 0], fused[:, 1]))

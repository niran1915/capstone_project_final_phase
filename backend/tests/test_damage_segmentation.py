from __future__ import annotations

import unittest
import numpy as np

from app.services.damage_segmentation import DamageSegmentationService


class DetectorGuidedMaskTests(unittest.TestCase):
    def test_non_crack_detector_labels_map_to_stable_mask_ids(self):
        self.assertEqual(DamageSegmentationService._label_id("concrete spalling"), 2)
        self.assertEqual(DamageSegmentationService._label_id("rebar"), 3)
        self.assertEqual(DamageSegmentationService._label_id("micro-cracks"), 4)
        self.assertIsNone(DamageSegmentationService._label_id("crack"))

    def test_proposals_are_not_reported_as_supervised_multiclass_model(self):
        info = DamageSegmentationService().model_info()
        self.assertFalse(info["supervised_multiclass_checkpoint"])
        self.assertIn("FastSAM", info["method"])

    def test_box_iou_is_zero_for_non_overlapping_regions(self):
        self.assertEqual(
            DamageSegmentationService._box_iou(
                np.array([0, 0, 10, 10]),
                np.array([20, 20, 30, 30]),
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

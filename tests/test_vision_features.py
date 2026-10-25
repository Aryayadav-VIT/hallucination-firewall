"""Automated tests for vision feature extraction and prompt generation."""

import unittest
from PIL import Image
import numpy as np

from src.vision_features import (
    object_prompts,
    calculate_8_features,
    load_clip_model
)
from src.config import EXPECTED_FEATURES

class TestVisionFeatures(unittest.TestCase):

    def test_object_prompts_generation(self):
        """Verify prompt templates are correctly generated for positive and negative cases."""
        pos, neg = object_prompts("bicycle")
        self.assertEqual(len(pos), 4)
        self.assertEqual(len(neg), 4)
        for p in pos:
            self.assertIn("bicycle", p)
        for p in neg:
            self.assertIn("bicycle", p)

    def test_8_features_calculation(self):
        """Verify 8 features are calculated with valid numeric values and correct keys."""
        # Create a small test image in memory
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        features = calculate_8_features(
            image=img,
            question="Is there a car in the image?",
            answer="Yes, there is a car.",
            target_object="car",
            claim_present=1
        )

        self.assertEqual(len(features), 8)
        for key in EXPECTED_FEATURES:
            self.assertIn(key, features)
            val = features[key]
            self.assertTrue(np.isfinite(val), f"Feature {key} is non-finite: {val}")

if __name__ == "__main__":
    unittest.main()

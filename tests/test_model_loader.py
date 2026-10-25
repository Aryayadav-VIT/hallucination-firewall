"""Automated tests for model loading and schema validation."""

import unittest
from pathlib import Path
from src.model_loader import find_model_path, load_bundle, validate_bundle
from src.config import EXPECTED_FEATURES, DEFAULT_THRESHOLD

class TestModelLoader(unittest.TestCase):

    def test_find_model_path(self):
        """Verify dynamic discovery locates the actual model file."""
        model_path = find_model_path()
        self.assertTrue(isinstance(model_path, Path))
        self.assertTrue(model_path.is_file(), f"Model path does not exist: {model_path}")

    def test_bundle_structure(self):
        """Verify the model bundle contains all expected keys, features, classes, and threshold."""
        bundle = load_bundle()
        self.assertIsInstance(bundle, dict)

        required_keys = [
            "model",
            "feature_columns",
            "threshold",
            "clip_model_name",
            "clip_pretrained",
            "target_definition",
        ]
        for key in required_keys:
            self.assertIn(key, bundle)

        # Feature columns and order
        self.assertEqual(list(bundle["feature_columns"]), EXPECTED_FEATURES)

        # Model attributes
        model = bundle["model"]
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertEqual(model.n_features_in_, 8)
        self.assertEqual(list(model.classes_), [0, 1])

        # Threshold
        self.assertEqual(bundle["threshold"], DEFAULT_THRESHOLD)

    def test_validate_bundle_invalid(self):
        """Verify schema validation raises errors on malformed bundles."""
        with self.assertRaises(ValueError):
            validate_bundle("not_a_dict")

        with self.assertRaises(ValueError):
            validate_bundle({"model": None})  # missing keys

if __name__ == "__main__":
    unittest.main()

"""Automated tests for demo laboratory integrity."""

import unittest
from pathlib import Path
from src.demo_cases import DEMO_CASES
from src.config import DEMO_IMAGES_DIR

class TestDemoCases(unittest.TestCase):

    def test_demo_cases_count(self):
        """Verify at least 12 demonstration cases exist."""
        self.assertGreaterEqual(len(DEMO_CASES), 12)

    def test_demo_case_structure_and_assets(self):
        """Verify each demo case has valid metadata and an existing image asset."""
        for case in DEMO_CASES:
            case_id = case.get("id")
            self.assertIsNotNone(case_id)
            self.assertIn("title", case)
            self.assertIn("category", case)
            self.assertIn("question", case)
            self.assertIn("answer", case)
            self.assertIn("expected_status", case)
            self.assertIn("method_available", case)

            # Check that the image file exists on disk
            img_file = case.get("image_file")
            self.assertIsNotNone(img_file)
            img_path = DEMO_IMAGES_DIR / img_file
            self.assertTrue(img_path.is_file(), f"Demo image missing: {img_path}")

if __name__ == "__main__":
    unittest.main()

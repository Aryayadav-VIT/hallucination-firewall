"""Automated tests for hallucination checks across categories."""

import unittest
from PIL import Image

from src.config import HallucinationCategory, VerificationStatus
from src.question_parser import parse_question_and_answer
from src.hallucination_checks import verify_claim
from src.model_loader import load_bundle

class TestHallucinationChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        # Synthetic test image
        cls.img = Image.new("RGB", (100, 100), color=(220, 30, 30))

    def test_object_existence_check(self):
        """Test object existence verification output."""
        claim = parse_question_and_answer(
            "Is there a red box in the image?",
            "Yes, there is a red box."
        )
        res = verify_claim(self.img, claim, model_bundle=self.bundle)
        self.assertEqual(res.category, HallucinationCategory.OBJECT_EXISTENCE)
        self.assertIn(res.status, [
            VerificationStatus.LIKELY_SUPPORTED,
            VerificationStatus.POTENTIAL_HALLUCINATION,
            VerificationStatus.INCONCLUSIVE
        ])
        self.assertIsNotNone(res.risk_score)
        self.assertTrue(0.0 <= res.risk_score <= 1.0)
        self.assertIn("image_question_similarity", res.detailed_metrics)

    def test_counting_limitation_flag(self):
        """Test that count claims are not given fabricated counts and limitation is stated."""
        claim = parse_question_and_answer(
            "How many people are in the image?",
            "There are two people."
        )
        res = verify_claim(self.img, claim, model_bundle=self.bundle)
        self.assertEqual(res.category, HallucinationCategory.COUNTING)
        self.assertEqual(res.status, VerificationStatus.UNSUPPORTED_CHECK_TYPE)
        self.assertTrue(any("instance" in l.lower() or "count" in l.lower() for l in res.limitations))

    def test_unsupported_speculation(self):
        """Test that non-visual speculation is cleanly refused with pixel-boundary explanation."""
        claim = parse_question_and_answer(
            "How much does this laptop cost?",
            "It costs Rs 80,000."
        )
        res = verify_claim(self.img, claim, model_bundle=self.bundle)
        self.assertEqual(res.category, HallucinationCategory.UNSUPPORTED_SPECULATION)
        self.assertEqual(res.status, VerificationStatus.UNSUPPORTED_CHECK_TYPE)
        self.assertIn("non-visual", res.evidence_summary.lower())

if __name__ == "__main__":
    unittest.main()

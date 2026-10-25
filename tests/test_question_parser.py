"""Automated tests for question and claim parser across all categories."""

import unittest
from src.question_parser import parse_question_and_answer, clean_object_name, singularize
from src.config import HallucinationCategory

class TestQuestionParser(unittest.TestCase):

    def test_phase1_audit_queries(self):
        """Verify the 5 queries specified in Phase 1 audit."""
        # 1. Phone
        c1 = parse_question_and_answer("Is there a phone in the image?", "Yes, there is a phone.")
        self.assertEqual(c1.category, HallucinationCategory.OBJECT_EXISTENCE)
        self.assertEqual(c1.target_object, "phone")
        self.assertEqual(c1.claim_polarity, 1)

        # 2. Dog
        c2 = parse_question_and_answer("Is there a dog in the image?", "No, there is no dog.")
        self.assertEqual(c2.category, HallucinationCategory.OBJECT_EXISTENCE)
        self.assertEqual(c2.target_object, "dog")
        self.assertEqual(c2.claim_polarity, 0)

        # 3. Cars (plural normalized to singular)
        c3 = parse_question_and_answer("Are there any cars in the image?", "Yes, there are cars.")
        self.assertEqual(c3.category, HallucinationCategory.OBJECT_EXISTENCE)
        self.assertEqual(c3.target_object, "car")
        self.assertEqual(c3.claim_polarity, 1)

        # 4. Color Attribute
        c4 = parse_question_and_answer("What color is the car?", "The car is red.")
        self.assertEqual(c4.category, HallucinationCategory.ATTRIBUTE)
        self.assertEqual(c4.target_object, "car")
        self.assertEqual(c4.target_attribute, "red")

        # 5. Counting
        c5 = parse_question_and_answer("How many people are in the image?", "There are three people.")
        self.assertEqual(c5.category, HallucinationCategory.COUNTING)
        self.assertEqual(c5.target_object, "person")
        self.assertEqual(c5.claimed_count, 3)

    def test_various_existence_formats(self):
        """Test variations of object presence questions."""
        c1 = parse_question_and_answer("Can you see a bicycle?", "Yes, I see a bicycle.")
        self.assertEqual(c1.target_object, "bicycle")
        self.assertEqual(c1.claim_polarity, 1)

        c2 = parse_question_and_answer("Does the image contain an apple?", "No, it does not.")
        self.assertEqual(c2.target_object, "apple")
        self.assertEqual(c2.claim_polarity, 0)

        c3 = parse_question_and_answer("Is there any cup visible?", "Yes, present.")
        self.assertEqual(c3.target_object, "cup")
        self.assertEqual(c3.claim_polarity, 1)

        c4 = parse_question_and_answer("There isn't a phone in the image.", "Correct, no phone.")
        self.assertEqual(c4.target_object, "phone")
        self.assertEqual(c4.claim_polarity, 0)

    def test_spatial_relations(self):
        """Test spatial relation detection between two objects."""
        c = parse_question_and_answer("Is the cup on the table?", "Yes, the cup is on the table.")
        self.assertEqual(c.category, HallucinationCategory.SPATIAL_RELATION)
        self.assertEqual(c.target_object, "cup")
        self.assertEqual(c.relation, "on")
        self.assertEqual(c.reference_object, "table")

    def test_unsupported_speculation(self):
        """Test non-visual speculation detection."""
        c = parse_question_and_answer("How much does the car cost?", "It costs 10 lakhs.")
        self.assertEqual(c.category, HallucinationCategory.UNSUPPORTED_SPECULATION)
        self.assertFalse(c.is_supported_category)

    def test_meta_word_avoidance(self):
        """Verify that meta words are never extracted as the object."""
        self.assertIsNone(clean_object_name("image"))
        self.assertIsNone(clean_object_name("picture"))
        self.assertIsNone(clean_object_name("photo"))
        self.assertIsNone(clean_object_name("there"))
        self.assertIsNone(clean_object_name("visible"))

    def test_singularization(self):
        """Verify plural to singular mapping."""
        self.assertEqual(singularize("cars"), "car")
        self.assertEqual(singularize("people"), "person")
        self.assertEqual(singularize("bicycles"), "bicycle")
        self.assertEqual(singularize("boxes"), "box")

if __name__ == "__main__":
    unittest.main()

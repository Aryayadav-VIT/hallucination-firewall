"""Configuration and constants for the Multimodal Hallucination Firewall."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DEMO_IMAGES_DIR = ASSETS_DIR / "demo_images"

# Model hyperparameters & CLIP setup
DEFAULT_THRESHOLD = 0.27
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Expected 8 features in exact order for the trained SVM bundle
EXPECTED_FEATURES = [
    "image_question_similarity",
    "image_answer_similarity",
    "multimodal_grounding_score",
    "object_positive_similarity",
    "object_negative_similarity",
    "object_presence_margin",
    "answer_claim_present",
    "claim_evidence_alignment",
]

# Hallucination Categories
class HallucinationCategory:
    OBJECT_EXISTENCE = "Object Existence"
    ATTRIBUTE = "Visual Attribute"
    COUNTING = "Object Count"
    SPATIAL_RELATION = "Spatial Relation"
    ACTION_ACTIVITY = "Action / Activity"
    SCENE_CONTEXT = "Scene / Context"
    TEXT_READING = "Text / OCR"
    UNSUPPORTED_SPECULATION = "External Speculation"
    COMPOUND = "Compound Claim"

# Verification Statuses
class VerificationStatus:
    LIKELY_SUPPORTED = "Likely supported"
    POTENTIAL_HALLUCINATION = "Potential hallucination"
    INCONCLUSIVE = "Inconclusive"
    UNSUPPORTED_CHECK_TYPE = "Unsupported check type"

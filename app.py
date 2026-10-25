import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import open_clip
from PIL import Image

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Hallucination Firewall",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "hallucination_firewall_corrected_bundle.joblib"
DEFAULT_THRESHOLD = 0.27
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

st.title("🛡️ Multimodal Hallucination Firewall")
st.markdown("### Evidence-Based AI Answer Verification")
st.caption(
    "A research prototype that checks object-presence claims against visual evidence. "
    "Its current classifier is trained on POPE-derived object-presence contradictions."
)
st.divider()


# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------
@st.cache_resource
def load_bundle(path_string):
    bundle = joblib.load(path_string)
    required = {
        "model", "feature_columns", "threshold",
        "clip_model_name", "clip_pretrained", "target_definition"
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")

    expected = [
        "image_question_similarity",
        "image_answer_similarity",
        "multimodal_grounding_score",
        "object_positive_similarity",
        "object_negative_similarity",
        "object_presence_margin",
        "answer_claim_present",
        "claim_evidence_alignment",
    ]
    if bundle["feature_columns"] != expected:
        raise ValueError(
            "Feature schema mismatch. The loaded model bundle does not contain "
            "the expected eight features in the training order."
        )
    model_count = getattr(bundle["model"], "n_features_in_", None)
    if model_count != len(expected):
        raise ValueError(
            f"Model expects {model_count} features, but {len(expected)} were declared."
        )
    return bundle


@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


# ------------------------------------------------------------
# QUESTION / ANSWER PARSING
# Keep conservative: this prototype supports object-presence claims.
# ------------------------------------------------------------
def extract_target_object(question):
    q = str(question).strip().rstrip("?.! ")
    patterns = [
        r"^(?:is there|are there any|are there|does (?:the|this) image contain|does (?:the|this) image have|is) (?:a|an|the|any)?\s*(.+?)\s+(?:in|inside|visible in) (?:the|this|an?) (?:image|picture|photo)$",
        r"^(?:is|are) (.+?) present (?:in|inside) (?:the|this) (?:image|picture|photo)$",
        r"^(?:does (?:the|this) image contain)\s+(.+)$",
        r"^(?:is there)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q, flags=re.I)

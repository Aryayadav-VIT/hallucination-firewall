import streamlit as st
import torch
import open_clip
import pandas as pd
import joblib
import re
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal Hallucination Firewall",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Multimodal Hallucination Firewall")
st.markdown("### Evidence-Based AI Answer Verification")
st.caption(
    "Analyze whether an AI-generated answer is supported by "
    "the visual evidence in an image."
)
st.divider()


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "hallucination_firewall_svm.pkl"


# ============================================================
# LOAD SVM MODEL
# ============================================================

@st.cache_resource
def load_firewall_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD CLIP
# ============================================================

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device


# ============================================================
# CALCULATE ORIGINAL CLIP FEATURES
# ============================================================

def calculate_features(image, question, answer):

    clip_model, preprocess, tokenizer, device = load_clip_model()

    image_input = preprocess(image).unsqueeze(0).to(device)
    question_tokens = tokenizer([question]).to(device)
    answer_tokens = tokenizer([answer]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        question_features = clip_model.encode_text(question_tokens)
        answer_features = clip_model.encode_text(answer_tokens)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        question_features = question_features / question_features.norm(
            dim=-1, keepdim=True
        )
        answer_features = answer_features / answer_features.norm(
            dim=-1, keepdim=True
        )

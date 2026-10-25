
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import open_clip
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Multimodal Hallucination Firewall",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "hallucination_firewall_corrected_bundle.joblib",
    BASE_DIR / "models" / "hallucination_firewall_corrected_bundle.joblib",
]

MODEL_PATH = next(
    (path for path in MODEL_CANDIDATES if path.is_file()),
    MODEL_CANDIDATES[0],
)

DEFAULT_THRESHOLD = 0.27
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

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


# ============================================================
# PAGE HEADER
# ============================================================
st.title("🛡️ Multimodal Hallucination Firewall")
st.subheader("Evidence-Based AI Answer Verification")

st.write(
    "Upload an image, enter an object-presence question and "
    "provide an AI-generated answer. The system combines CLIP "
    "visual-language similarities with an SVM classifier to "
    "estimate whether the answer may contradict the image."
)

st.info(
    "Research prototype: the classifier was trained on "
    "POPE-derived object-presence examples. Its predictions "
    "are not definitive object detection or proof of truth."
)

st.divider()


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================
@st.cache_resource
def load_bundle(model_path, modified_time):
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("The model bundle must be a dictionary.")

    required_keys = [
        "model",
        "feature_columns",
        "threshold",
        "clip_model_name",
        "clip_pretrained",
        "target_definition",
    ]

    missing = [
        key for key in required_keys if key not in bundle
    ]

    if missing:
        raise ValueError(
            f"Missing model bundle keys: {missing}"
        )

    feature_columns = list(bundle["feature_columns"])

    if feature_columns != EXPECTED_FEATURES:
        raise ValueError(
            "Feature schema mismatch.\n\n"
            f"Expected: {EXPECTED_FEATURES}\n\n"
            f"Found: {feature_columns}\n\n"
            "Do not deploy this model until the training "
            "and application feature schemas match."
        )

    model = bundle["model"]
    model_count = getattr(model, "n_features_in_", None)

    if model_count is not None and model_count != len(EXPECTED_FEATURES):
        raise ValueError(
            f"The classifier expects {model_count} features, "
            f"but the application prepares "
            f"{len(EXPECTED_FEATURES)}."
        )

    return bundle


# ============================================================
# LOAD CLIP
# ============================================================
@st.cache_resource
def load_clip(model_name, pretrained):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    return model, preprocess, tokenizer, device


# ============================================================
# QUESTION PARSING
# ============================================================
def extract_target_object(question):
    q = str(question).strip().rstrip("?.! ")

    patterns = [
        r"^(?:is there|are there any|are there)\s+"
        r"(?:a|an|the|any)?\s*(.+?)\s+"
        r"(?:in|inside|visible in)\s+"
        r"(?:the|this|an?)\s+(?:image|picture|photo)$",

        r"^(?:is|are)\s+(.+?)\s+present\s+"
        r"(?:in|inside)\s+(?:the|this)\s+"
        r"(?:image|picture|photo)$",

        r"^does\s+(?:the|this)\s+image\s+contain\s+(.+)$",

        r"^does\s+(?:the|this)\s+image\s+have\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, q, flags=re.IGNORECASE)

        if match:
            obj = match.group(1).strip(" .?!")
            obj = re.sub(
                r"^(?:a|an|the|any)\s+",
                "",
                obj,
                flags=re.IGNORECASE,
            ).strip()

            if obj:
                return obj

    return None


# ============================================================
# ANSWER CLAIM PARSING
# 1 = claims present, 0 = claims absent, None = ambiguous
# ============================================================
def infer_answer_claim(answer):
    a = re.sub(
        r"\s+",
        " ",
        str(answer).lower(),
    ).strip()

    negative_patterns = [
        r"\bno\b",
        r"\bnot present\b",
        r"\bnot visible\b",
        r"\bdoes not contain\b",
        r"\bdoesn't contain\b",
        r"\bthere (?:is|are) no\b",
        r"\bwithout\b",
        r"\bcannot see\b",
        r"\bcan't see\b",
        r"\bdo not see\b",
        r"\bdon't see\b",
        r"\babsent\b",
    ]

    if any(re.search(p, a) for p in negative_patterns):
        return 0

    positive_patterns = [
        r"^\s*yes\b",
        r"\bthere (?:is|are)\b",
        r"\bcontains?\b",
        r"\bshows?\b",
        r"\bvisible\b",
        r"\bi can see\b",
        r"\bpresent\b",
    ]

    if any(re.search(p, a) for p in positive_patterns):
        return 1

    return None


# ============================================================
# CLIP OBJECT PROMPTS
# ============================================================
def object_prompts(obj):
    positive = [
        f"an image containing {obj}",
        f"a picture of {obj}",
        f"a photo with {obj}",
        f"{obj} visible in the image",
    ]

    negative = [
        f"an image without {obj}",
        f"a picture without {obj}",
        f"a photo with no {obj}",
        f"no {obj} is visible in the image",
    ]

    return positive, negative


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def calculate_features(
    image,
    question,
    answer,
    target_object,
    claim_present,
):
    model, preprocess, tokenizer, device = load_clip(
        CLIP_MODEL_NAME,
        CLIP_PRETRAINED,
    )

    positive_prompts, negative_prompts = object_prompts(
        target_object
    )

    texts = (
        [question, answer]
        + positive_prompts
        + negative_prompts
    )

    image_tensor = (
        preprocess(image.convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)
        text_embeddings = model.encode_text(tokens)

        image_embedding = (
            image_embedding
            / image_embedding.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        text_embeddings = (
            text_embeddings
            / text_embeddings.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        similarities = (
            image_embedding @ text_embeddings.T
        )[0].float().cpu().numpy()

    image_question = float(similarities[0])
    image_answer = float(similarities[1])

    positive_similarity = float(
        np.mean(similarities[2:6])
    )

    negative_similarity = float(
        np.mean(similarities[6:10])
    )

    presence_margin = (
        positive_similarity - negative_similarity
    )

    claim_sign = 1.0 if claim_present == 1 else -1.0

    features = {
        "image_question_similarity": image_question,
        "image_answer_similarity": image_answer,
        "multimodal_grounding_score": (
            image_question + image_answer
        ) / 2.0,
        "object_positive_similarity": positive_similarity,
        "object_negative_similarity": negative_similarity,
        "object_presence_margin": presence_margin,
        "answer_claim_present": int(claim_present),
        "claim_evidence_alignment": (
            claim_sign * presence_margin
        ),
    }

    return features


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🛡️ Firewall")
    st.caption("CLIP + SVM object-presence verification")

    threshold = st.slider(
        "Hallucination threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help=(
            "Lower thresholds flag more candidates. "
            "This changes the decision cutoff, not training."
        ),
    )

    st.caption(
        "Lower threshold = stricter flagging"
    )

    st.divider()
    st.subheader("Supported scope")

    st.write("- Object-presence questions")
    st.write("- Clear positive or negative answers")
    st.write("- CLIP similarity features + SVM score")

    st.warning(
        "Counting, color, spatial relations and complex "
        "free-form answers have not been validated."
    )


# ============================================================
# INPUT INTERFACE
# ============================================================
left, right = st.columns(2)

with left:
    st.subheader("📷 Visual Evidence")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Could not read the image: {exc}")


with right:
    st.subheader("❓ Question")

    question = st.text_input(
        "Enter an object-presence question",
        placeholder="Is there a phone in the image?",
    )

    st.subheader("🤖 AI-Generated Answer")

    answer = st.text_area(
        "Enter the answer to verify",
        placeholder="No, there is no phone in the image.",
        height=150,
    )


run_analysis = st.button(
    "🔍 RUN FIREWALL ANALYSIS",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN ANALYSIS
# ============================================================
if run_analysis:
    if image is None:
        st.warning("Please upload a valid image.")
        st.stop()

    if not question.strip() or not answer.strip():
        st.warning(
            "Please enter both the question and the answer."
        )
        st.stop()

    target_object = extract_target_object(question)
    claim_present = infer_answer_claim(answer)

    if not target_object:
        st.error(
            "I couldn't identify the target object from your "
            "question. Try: 'Is there a phone in the image?'"
        )
        st.stop()

    if claim_present is None:
        st.error(
            "I couldn't determine whether the answer claims "
            "the object is present or absent. Use a clear "
            "yes/no presence statement."
        )
        st.stop()

    try:
        if not MODEL_PATH.is_file():
    st.error("Model bundle not found in the deployed app.")

    st.write("**App directory:**", str(BASE_DIR))
    st.write("**Expected filename:**", MODEL_NAME)

    st.write("**Files visible to Streamlit:**")
    visible_files = [
        str(p.relative_to(BASE_DIR))
        for p in BASE_DIR.rglob("*")
        if p.is_file()
    ]

    st.code(
        "\n".join(visible_files[:100])
        if visible_files
        else "No files found"
    )

    st.warning(
        "Check whether the model appears in this list. "
        "If it does not, the deployed repository/branch "
        "does not contain the model file."
    )
    st.stop()

        with st.spinner(
            "Loading model and analyzing visual evidence..."
        ):
            bundle = load_bundle(
                str(MODEL_PATH),
                MODEL_PATH.stat().st_mtime,
            )

            features = calculate_features(
                image,
                question.strip(),
                answer.strip(),
                target_object,
                claim_present,
            )

            feature_columns = bundle["feature_columns"]

            X = pd.DataFrame(
                [[features[name] for name in feature_columns]],
                columns=feature_columns,
            )

            classifier = bundle["model"]

            if not hasattr(classifier, "predict_proba"):
                raise ValueError(
                    "The saved classifier does not provide "
                    "probability estimates."
                )

            probabilities = classifier.predict_proba(X)[0]
            classes = list(classifier.classes_)

            # Training definition: class 1 = hallucinated candidate
            if 1 not in classes:
                raise ValueError(
                    "The classifier has no class 1 output. "
                    "Check the training label definition."
                )

            risk = float(
                probabilities[classes.index(1)]
            )

            decision = (
                "HALLUCINATED"
                if risk >= threshold
                else "LOWER RISK"
            )

    except Exception as exc:
        st.error(
            "Analysis failed. Check the model bundle, "
            "dependencies and training feature schema."
        )
        st.exception(exc)
        st.stop()

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------
    st.divider()
    st.header("🛡️ Firewall Verdict")

    if decision == "HALLUCINATED":
        st.error(
            "### 🔴 POTENTIAL HALLUCINATION\n\n"
            "The classifier flags this candidate as potentially "
            "inconsistent with the visual evidence."
        )
    else:
        st.success(
            "### 🟢 LOWER MODEL-ESTIMATED RISK\n\n"
            "The classifier did not flag this candidate at the "
            "selected threshold. This does not prove that the "
            "answer is factually correct."
        )

    col1, col2 = st.columns(2)

    col1.metric(
        "SVM hallucination score",
        f"{risk:.1%}",
    )

    col2.metric(
        "Decision threshold",
        f"{threshold:.2f}",
    )

    st.progress(
        min(max(risk, 0.0), 1.0)
    )

    # --------------------------------------------------------
    # OBJECT-LEVEL CLIP EVIDENCE
    # --------------------------------------------------------
    st.subheader("🔎 Object-Level CLIP Evidence")

    st.write(f"**Target object:** {target_object}")

    st.write(
        "**Answer claim:** "
        + ("Object present" if claim_present == 1
           else "Object absent")
    )

    margin = features["object_presence_margin"]

    evidence1, evidence2, evidence3 = st.columns(3)

    evidence1.metric(
        "Positive prompt similarity",
        f"{features['object_positive_similarity']:.4f}",
    )

    evidence2.metric(
        "Negative prompt similarity",
        f"{features['object_negative_similarity']:.4f}",
    )

    evidence3.metric(
        "Presence margin",
        f"{margin:+.4f}",
    )

    # A small margin is deliberately treated as uncertain.
    if margin > 0.01:
        clip_suggests_present = True
    elif margin < -0.01:
        clip_suggests_present = False
    else:
        clip_suggests_present = None

    if clip_suggests_present is None:
        st.warning(
            "CLIP's positive-vs-negative prompt margin is "
            "small. The visual evidence is ambiguous under "
            "this heuristic."
        )
    elif (
        claim_present == 0 and clip_suggests_present
    ) or (
        claim_present == 1 and not clip_suggests_present
    ):
        st.warning(
            "The answer's presence/absence claim disagrees "
            "with the direction of the CLIP prompt margin. "
            "This is a heuristic warning, not definitive "
            "object detection."
        )
    else:
        st.info(
            "The CLIP prompt margin does not indicate a direct "
            "presence/absence conflict. Similarity scores alone "
            "cannot confirm whether an object is present."
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------
    st.subheader("📊 Model Features")

    feature_table = pd.DataFrame(
        [
            {
                "Feature": name,
                "Value": float(features[name]),
            }
            for name in feature_columns
        ]
    )

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------
    st.subheader("🧠 Interpretation")

    st.write(
        f"The model's estimated hallucination score is "
        f"**{risk:.1%}** and the selected decision threshold "
        f"is **{threshold:.2f}**."
    )

    st.caption(
        "The model was trained on POPE-derived templated "
        "object-presence examples. Its performance on arbitrary "
        "free-form answers, counting, colors and spatial "
        "relations has not been established."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================
st.divider()
st.subheader("⚙️ Model Information")

st.write("**Vision-language encoder:** CLIP ViT-B/32")
st.write("**Classifier:** RBF SVM with probability estimates")
st.write("**Feature count:** 8")

st.caption(
    "Research prototype. Always verify predictions against "
    "the original image, particularly when the model is uncertain."
)

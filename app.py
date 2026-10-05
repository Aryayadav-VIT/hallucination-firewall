import streamlit as st
import torch
import open_clip
import pandas as pd
import joblib
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
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_firewall_model():
    return joblib.load(MODEL_PATH)


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
# CLIP FEATURE CALCULATION
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

        image_question_similarity = (
            image_features @ question_features.T
        ).item()

        image_answer_similarity = (
            image_features @ answer_features.T
        ).item()

        grounding_score = (
            image_question_similarity + image_answer_similarity
        ) / 2

    return (
        image_question_similarity,
        image_answer_similarity,
        grounding_score
    )


# ============================================================
# OBJECT EVIDENCE
# ============================================================

OBJECT_ALIASES = {
    "phone": ["phone", "cell phone", "mobile phone", "telephone"],
    "laptop": ["laptop", "computer"],
    "dog": ["dog"],
    "cat": ["cat"],
    "car": ["car", "automobile"],
    "person": ["person", "people", "man", "woman"],
    "chair": ["chair"],
    "table": ["table"],
    "book": ["book"],
    "bottle": ["bottle"],
    "bag": ["bag", "backpack", "handbag"],
    "camera": ["camera"],
    "watch": ["watch"],
    "shoe": ["shoe", "shoes"]
}


def find_queried_object(question):
    q = question.lower()

    for canonical, aliases in OBJECT_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return canonical

    return None


def answer_denies_object(answer):
    a = answer.lower()

    denial_patterns = [
        "there is no",
        "there are no",
        "there isn't",
        "there isnt",
        "there aren't",
        "there arent",
        "no ",
        "not present",
        "not visible",
        "does not contain",
        "doesn't contain",
        "without"
    ]

    return any(pattern in a for pattern in denial_patterns)


def check_object_evidence(image, question, answer):
    """
    Uses CLIP to compare:
      'a photo containing X'
    against:
      'a photo without X'

    This is an auxiliary evidence check, not a trained detector.
    """
    object_name = find_queried_object(question)

    if object_name is None:
        return {
            "object": None,
            "present_score": 0.0,
            "absent_score": 0.0,
            "present": None,
            "contradiction": False
        }

    clip_model, preprocess, tokenizer, device = load_clip_model()

    aliases = OBJECT_ALIASES[object_name]
    phrase = aliases[0]

    positive_prompt = f"a photo of a {phrase}"
    negative_prompt = f"a photo without a {phrase}"

    image_input = preprocess(image).unsqueeze(0).to(device)
    tokens = tokenizer(
        [positive_prompt, negative_prompt]
    ).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(tokens)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        scores = (image_features @ text_features.T)[0]

    present_score = float(scores[0].item())
    absent_score = float(scores[1].item())

    # Margin-based evidence. A small margin means uncertain evidence.
    margin = present_score - absent_score
    present = margin > 0.01

    contradiction = (
        present
        and answer_denies_object(answer)
        and margin > 0.01
    )

    return {
        "object": object_name,
        "present_score": present_score,
        "absent_score": absent_score,
        "present": present,
        "contradiction": contradiction
    }


# ============================================================
# HALLUCINATION TYPE
# ============================================================

def determine_hallucination_type(
    question,
    answer,
    image_answer_similarity,
    risk,
    object_evidence
):
    q = question.lower()

    if object_evidence["contradiction"]:
        return "Object Presence Hallucination"

    count_words = [
        "how many",
        "number of",
        "count",
        "quantity",
        "how much"
    ]

    if any(word in q for word in count_words):
        return "Count Hallucination"

    attribute_words = [
        "color", "colour", "size", "small", "large",
        "big", "red", "blue", "green", "black", "white"
    ]

    if any(word in q for word in attribute_words):
        return "Attribute Hallucination"

    spatial_words = [
        "where", "left", "right", "behind", "front",
        "next to", "near", "beside", "above", "below"
    ]

    if any(word in q for word in spatial_words):
        return "Spatial Hallucination"

    if risk >= 0.60:
        if image_answer_similarity < 0.20:
            return "Object / Content Hallucination"
        return "Unsupported Answer"

    return "General Hallucination"


# ============================================================
# DEMO CASES
# ============================================================

demo_cases = {
    "Select a demo": {
        "question": "",
        "answer": ""
    },
    "Phone — Hallucinated Negation": {
        "question": "Is there a phone in the image?",
        "answer": "No, there is no phone in the picture."
    },
    "Supported Object": {
        "question": "Is there a laptop in the image?",
        "answer": "Yes, there is a laptop in the image."
    },
    "Count Hallucination": {
        "question": "How many dogs are in the image?",
        "answer": "There are five dogs in the image."
    },
    "Attribute Hallucination": {
        "question": "What color is the dog?",
        "answer": "The dog is bright purple."
    }
}


def update_demo_fields():
    selected = st.session_state.demo_selection
    st.session_state.question_input = demo_cases[selected]["question"]
    st.session_state.answer_input = demo_cases[selected]["answer"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🛡️ Firewall")
    st.divider()

    st.subheader("🧪 Demo Test Cases")

    if "demo_selection" not in st.session_state:
        st.session_state.demo_selection = "Select a demo"

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    if "answer_input" not in st.session_state:
        st.session_state.answer_input = ""

    st.selectbox(
        "Choose a test case:",
        list(demo_cases.keys()),
        key="demo_selection",
        on_change=update_demo_fields
    )

    st.divider()

    st.subheader("🎚️ Firewall Sensitivity")

    threshold = st.slider(
        "Hallucination Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Lower values make the SVM firewall stricter."
    )

    st.caption("Lower threshold = stricter SVM verification")

    st.divider()

    st.subheader("Pipeline")
    st.write("📷 Image")
    st.write("❓ Question")
    st.write("🤖 AI Answer")
    st.write("🧠 CLIP Features")
    st.write("⚡ SVM Classifier")
    st.write("🔎 Object Evidence")
    st.write("🛡️ Firewall Verdict")


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🔍 Verify an AI-Generated Answer")

left, right = st.columns(2)

with left:
    st.subheader("📷 Visual Evidence")
    st.write("Upload the image that should support the AI answer.")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

with right:
    st.subheader("❓ Question")

    st.text_input(
        "Enter the question",
        placeholder="Is there a phone in the image?",
        key="question_input"
    )

    st.subheader("🤖 AI-Generated Answer")

    st.text_area(
        "Enter the AI answer",
        placeholder="Yes, there is a phone in the image.",
        height=150,
        key="answer_input"
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.write("")

run_analysis = st.button(
    "🔍 RUN FIREWALL ANALYSIS",
    use_container_width=True
)


# ============================================================
# FIREWALL ANALYSIS
# ============================================================

if run_analysis:

    question = st.session_state.question_input
    answer = st.session_state.answer_input

    if image is None:
        st.warning("📷 Please upload an image first.")
        st.stop()

    if not question.strip():
        st.warning("❓ Please enter a question.")
        st.stop()

    if not answer.strip():
        st.warning("🤖 Please enter an AI-generated answer.")
        st.stop()

    # --------------------------------------------------------
    # LOAD SVM ONLY WHEN NEEDED
    # --------------------------------------------------------

    try:
        with st.spinner("⚡ Loading firewall classifier..."):
            firewall_model = load_firewall_model()
    except Exception as e:
        st.error(f"❌ Unable to load the SVM model: {e}")
        st.stop()

    # --------------------------------------------------------
    # CLIP FEATURES
    # --------------------------------------------------------

    with st.spinner("🧠 CLIP is analyzing the visual evidence..."):
        (
            image_question_similarity,
            image_answer_similarity,
            grounding_score
        ) = calculate_features(
            image,
            question,
            answer
        )

    # --------------------------------------------------------
    # SVM PREDICTION
    # --------------------------------------------------------

    X_input = pd.DataFrame(
        [[
            image_question_similarity,
            image_answer_similarity,
            grounding_score
        ]],
        columns=[
            "image_question_similarity",
            "image_answer_similarity",
            "multimodal_grounding_score"
        ]
    )

    prediction = firewall_model.predict(X_input)[0]

    if hasattr(firewall_model, "predict_proba"):
        probabilities = firewall_model.predict_proba(X_input)[0]
        classes = list(firewall_model.classes_)

        if 0 in classes:
            hallucination_probability = float(
                probabilities[classes.index(0)]
            )
        elif 1 in classes:
            hallucination_probability = float(
                1 - probabilities[classes.index(1)]
            )
        else:
            hallucination_probability = 0.0
    else:
        hallucination_probability = 0.0

    # --------------------------------------------------------
    # OBJECT EVIDENCE
    # --------------------------------------------------------

    with st.spinner("🔎 Checking object-level visual evidence..."):
        object_evidence = check_object_evidence(
            image,
            question,
            answer
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    svm_decision = (
        hallucination_probability >= threshold
    )

    # Object contradiction is a safety override.
    object_override = object_evidence["contradiction"]

    final_hallucination = svm_decision or object_override

    if final_hallucination:
        decision = "HALLUCINATED"
        verdict_message = (
            "The AI answer is not sufficiently supported "
            "by the visual evidence."
        )
    else:
        decision = "SUPPORTED"
        verdict_message = (
            "The AI answer is consistent with "
            "the available visual evidence."
        )

    risk_percent = hallucination_probability * 100

    hallucination_type = determine_hallucination_type(
        question,
        answer,
        image_answer_similarity,
        hallucination_probability,
        object_evidence
    )

    # ========================================================
    # RESULT
    # ========================================================

    st.divider()
    st.header("🛡️ Firewall Verdict")

    if decision == "SUPPORTED":
        st.success(
            "## 🟢 SUPPORTED\n\n" + verdict_message
        )
    else:
        st.error(
            "## 🔴 HALLUCINATED\n\n" + verdict_message
        )

    if decision == "HALLUCINATED":
        st.warning(
            f"🔎 **Hallucination Type:** {hallucination_type}"
        )
    else:
        st.info(
            "🔎 **Verification Type:** Visually Supported Answer"
        )

    # ========================================================
    # OBJECT EVIDENCE RESULT
    # ========================================================

    st.subheader("🔎 Object-Level Evidence")

    if object_evidence["object"] is None:
        st.info(
            "No recognizable object was extracted from the question. "
            "The decision relies on CLIP multimodal grounding + SVM."
        )
    else:
        obj = object_evidence["object"]
        present_score = object_evidence["present_score"]
        absent_score = object_evidence["absent_score"]

        obj1, obj2, obj3 = st.columns(3)

        with obj1:
            st.metric(
                "Queried Object",
                obj.title()
            )

        with obj2:
            st.metric(
                "Object-Present Score",
                f"{present_score:.4f}"
            )

        with obj3:
            st.metric(
                "Object-Absent Score",
                f"{absent_score:.4f}"
            )

        if object_evidence["contradiction"]:
            st.error(
                f"⚠️ Visual evidence conflicts with the AI answer: "
                f"the image shows stronger CLIP evidence for a "
                f"**{obj}**, while the answer denies its presence."
            )
        elif object_evidence["present"]:
            st.info(
                f"Visual evidence is more consistent with the "
                f"presence of a **{obj}**."
            )
        else:
            st.info(
                f"Visual evidence does not provide strong evidence "
                f"for the presence of a **{obj}**."
            )

    # ========================================================
    # RISK
    # ========================================================

    st.subheader("📊 Hallucination Risk")

    risk_col1, risk_col2 = st.columns([1, 2])

    with risk_col1:
        st.metric(
            "SVM Hallucination Probability",
            f"{risk_percent:.1f}%"
        )

    with risk_col2:
        st.progress(
            min(max(float(hallucination_probability), 0.0), 1.0)
        )

        if risk_percent < 30:
            st.success(
                "🟢 LOW RISK — SVM considers the answer well grounded."
            )
        elif risk_percent < 60:
            st.warning(
                "🟡 MODERATE RISK — Evidence should be reviewed."
            )
        else:
            st.error(
                "🔴 HIGH RISK — SVM detects substantial hallucination risk."
            )

    if object_override:
        st.warning(
            "🛡️ Safety override activated: object-level evidence "
            "contradicts the AI answer."
        )

    # ========================================================
    # EVIDENCE SCORES
    # ========================================================

    st.subheader("🔬 Multimodal Evidence Analysis")

    score1, score2, score3 = st.columns(3)

    with score1:
        st.metric(
            "Image ↔ Question",
            f"{image_question_similarity:.4f}"
        )

    with score2:
        st.metric(
            "Image ↔ Answer",
            f"{image_answer_similarity:.4f}"
        )

    with score3:
        st.metric(
            "Grounding Score",
            f"{grounding_score:.4f}"
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader("🧠 Why did the Firewall decide this?")

    if object_override:
        st.error(
            f"""
### 🔴 Object-level contradiction detected

The queried object is **{object_evidence["object"]}**.

The CLIP object-evidence comparison found stronger evidence
for the object's presence than its absence.

However, the AI answer denies that object.

Therefore, the object-evidence safety layer overrode the
SVM's original decision.

**Final Status:** Hallucinated
"""
        )

    elif decision == "SUPPORTED":
        st.success(
            f"""
### 🟢 Visual evidence supports the answer

**Grounding Score:** {grounding_score:.4f}

**SVM Hallucination Risk:** {risk_percent:.1f}%

The SVM risk is below the selected firewall threshold
of **{threshold:.2f}**, and no object-level contradiction
was detected.
"""
        )

    else:
        st.error(
            f"""
### 🔴 Potential hallucination detected

**Detected Type:** {hallucination_type}

**Image–Answer Similarity:** {image_answer_similarity:.4f}

**Grounding Score:** {grounding_score:.4f}

**SVM Hallucination Risk:** {risk_percent:.1f}%

The SVM risk exceeds the selected firewall threshold
of **{threshold:.2f}**.
"""
        )

    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader("⚙️ Verification Pipeline")

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.info("📷 **IMAGE**\n\nVisual Evidence")

    with p2:
        st.info("🧠 **CLIP**\n\nMultimodal Features")

    with p3:
        st.info("⚡ **SVM**\n\nRisk Prediction")

    with p4:
        st.info("🔎 **OBJECT**\n\nEvidence Check")

    with p5:
        if decision == "SUPPORTED":
            st.success("🟢 **SUPPORTED**\n\nVerified")
        else:
            st.error("🔴 **FLAGGED**\n\nPotential Hallucination")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()
st.header("⚙️ Model Information")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric("Vision-Language Model", "CLIP ViT-B/32")

with model_col2:
    st.metric("Classifier", "SVM")

with model_col3:
    st.metric("Default Threshold", "0.30")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🛡️ Multimodal Hallucination Firewall | "
    "CLIP multimodal grounding + SVM detection + object evidence"
)

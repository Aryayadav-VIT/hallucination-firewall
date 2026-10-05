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
# CALCULATE CLIP FEATURES
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

        image_features = (
            image_features /
            image_features.norm(dim=-1, keepdim=True)
        )

        question_features = (
            question_features /
            question_features.norm(dim=-1, keepdim=True)
        )

        answer_features = (
            answer_features /
            answer_features.norm(dim=-1, keepdim=True)
        )

        image_question_similarity = (
            image_features @ question_features.T
        ).item()

        image_answer_similarity = (
            image_features @ answer_features.T
        ).item()

        multimodal_grounding_score = (
            image_question_similarity +
            image_answer_similarity
        ) / 2

    return (
        image_question_similarity,
        image_answer_similarity,
        multimodal_grounding_score
    )


# ============================================================
# HALLUCINATION TYPE
# ============================================================

def determine_hallucination_type(
    question,
    answer,
    image_question_similarity,
    image_answer_similarity,
    grounding_score,
    risk
):

    question_lower = question.lower()

    # Count-related questions
    count_words = [
        "how many",
        "number of",
        "count",
        "quantity",
        "how much"
    ]

    if any(word in question_lower for word in count_words):
        return "Count Hallucination"

    # Attribute questions
    attribute_words = [
        "color",
        "colour",
        "size",
        "small",
        "large",
        "big",
        "red",
        "blue",
        "green",
        "black",
        "white"
    ]

    if any(word in question_lower for word in attribute_words):
        return "Attribute Hallucination"

    # Spatial questions
    spatial_words = [
        "where",
        "left",
        "right",
        "behind",
        "front",
        "next to",
        "near",
        "beside",
        "above",
        "below"
    ]

    if any(word in question_lower for word in spatial_words):
        return "Spatial Hallucination"

    # Weak visual grounding
    if risk >= 0.60:

        if image_answer_similarity < 0.20:
            return "Object / Content Hallucination"

        return "Unsupported Answer"

    return "General Hallucination"


# ============================================================
# DEMO TEST CASES
# ============================================================

demo_cases = {

    "Select a demo": {
        "question": "",
        "answer": ""
    },

    "Demo 1 — Supported": {
        "question": "Is there a dog in the image?",
        "answer": "Yes, there is a dog in the image."
    },

    "Demo 2 — Count Hallucination": {
        "question": "How many dogs are in the image?",
        "answer": "There are five dogs in the image."
    },

    "Demo 3 — Attribute Hallucination": {
        "question": "What color is the dog?",
        "answer": "The dog is bright purple."
    },

    "Demo 4 — Unsupported Answer": {
        "question": "Is there a cat in the image?",
        "answer": "Yes, there is a cat wearing sunglasses."
    }
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ Firewall")

    st.divider()

    st.subheader("🧪 Demo Test Cases")

    selected_demo = st.selectbox(
        "Choose a test case:",
        list(demo_cases.keys())
    )

    st.divider()

    st.subheader("🎚️ Firewall Sensitivity")

    THRESHOLD = st.slider(
        "Hallucination Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Lower values make the firewall more strict."
    )

    st.caption(
        "Lower threshold = stricter verification"
    )

    st.divider()

    st.subheader("System Status")

    st.success("🟢 SVM Classifier Ready")
    st.success("🟢 CLIP Engine Ready")

    st.divider()

    st.subheader("Pipeline")

    st.write("📷 Image")
    st.write("❓ Question")
    st.write("🤖 AI Answer")
    st.write("🧠 CLIP Features")
    st.write("⚡ SVM Classifier")
    st.write("🛡️ Firewall Verdict")


# ============================================================
# LOAD MODEL
# ============================================================

try:

    firewall_model = load_firewall_model()

except Exception as e:

    st.error(f"❌ Unable to load the SVM model: {e}")
    st.stop()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🔍 Verify an AI-Generated Answer")

left, right = st.columns(2)


# ============================================================
# IMAGE
# ============================================================

with left:

    st.subheader("📷 Visual Evidence")

    st.write(
        "Upload the image that should support the AI answer."
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    image = None

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


# ============================================================
# QUESTION + ANSWER
# ============================================================

with right:

    st.subheader("❓ Question")

    demo_question = demo_cases[selected_demo]["question"]

    question = st.text_input(
        "Enter the question",
        value=demo_question,
        placeholder="Is there a dog in the image?"
    )

    st.subheader("🤖 AI-Generated Answer")

    demo_answer = demo_cases[selected_demo]["answer"]

    answer = st.text_area(
        "Enter the AI answer",
        value=demo_answer,
        placeholder="Yes, there is a dog in the image.",
        height=150
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

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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
    # CLIP ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "🧠 CLIP is analyzing the visual and textual evidence..."
    ):

        (
            image_question_similarity,
            image_answer_similarity,
            multimodal_grounding_score
        ) = calculate_features(
            image,
            question,
            answer
        )


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    X_input = pd.DataFrame(
        [[
            image_question_similarity,
            image_answer_similarity,
            multimodal_grounding_score
        ]],
        columns=[
            "image_question_similarity",
            "image_answer_similarity",
            "multimodal_grounding_score"
        ]
    )


    # --------------------------------------------------------
    # SVM PREDICTION
    # --------------------------------------------------------

    prediction = firewall_model.predict(
        X_input
    )[0]


    # --------------------------------------------------------
    # HALLUCINATION PROBABILITY
    # --------------------------------------------------------

    if hasattr(firewall_model, "predict_proba"):

        probabilities = firewall_model.predict_proba(
            X_input
        )[0]

        classes = list(
            firewall_model.classes_
        )

        # 0 = Hallucinated
        # 1 = Supported

        if 0 in classes:

            hallucination_probability = probabilities[
                classes.index(0)
            ]

        else:

            hallucination_probability = (
                1 -
                probabilities[
                    classes.index(1)
                ]
            )

    else:

        hallucination_probability = 0.0


    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if hallucination_probability >= THRESHOLD:

        decision = "HALLUCINATED"

        verdict_message = (
            "The AI answer is not sufficiently supported "
            "by the visual evidence."
        )

    else:

        decision = "SUPPORTED"

        verdict_message = (
            "The AI answer is consistent with "
            "the visual evidence."
        )


    risk_percent = hallucination_probability * 100


    # --------------------------------------------------------
    # HALLUCINATION TYPE
    # --------------------------------------------------------

    hallucination_type = determine_hallucination_type(
        question,
        answer,
        image_question_similarity,
        image_answer_similarity,
        multimodal_grounding_score,
        hallucination_probability
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("🛡️ Firewall Verdict")


    if decision == "SUPPORTED":

        st.success(
            "## 🟢 SUPPORTED\n\n"
            + verdict_message
        )

    else:

        st.error(
            "## 🔴 HALLUCINATED\n\n"
            + verdict_message
        )


    # ========================================================
    # HALLUCINATION TYPE
    # ========================================================

    if decision == "HALLUCINATED":

        st.warning(
            f"🔎 Hallucination Type: {hallucination_type}"
        )

    else:

        st.info(
            "🔎 Verification Type: Visually Supported Answer"
        )


    # ========================================================
    # RISK
    # ========================================================

    st.subheader("📊 Hallucination Risk")

    risk_col1, risk_col2 = st.columns([1, 2])

    with risk_col1:

        st.metric(
            "Hallucination Probability",
            f"{risk_percent:.1f}%"
        )

    with risk_col2:

        st.progress(
            min(
                max(
                    float(hallucination_probability),
                    0.0
                ),
                1.0
            )
        )

        if risk_percent < 30:

            st.success(
                "🟢 LOW RISK — Answer appears well grounded."
            )

        elif risk_percent < 60:

            st.warning(
                "🟡 MODERATE RISK — Evidence should be reviewed."
            )

        else:

            st.error(
                "🔴 HIGH RISK — Answer may contain unsupported information."
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
            f"{multimodal_grounding_score:.4f}"
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader(
        "🧠 Why did the Firewall decide this?"
    )

    if decision == "SUPPORTED":

        st.success(
            f"""
### 🟢 Visual evidence supports the answer

The AI answer shows sufficient alignment with
the uploaded image.

**Grounding Score:** {multimodal_grounding_score:.4f}

**Hallucination Risk:** {risk_percent:.1f}%

**Firewall Status:** Supported

The calculated risk is below the firewall threshold
of **{THRESHOLD:.2f}**.
"""
        )

    else:

        st.error(
            f"""
### 🔴 Potential hallucination detected

The AI answer is not sufficiently grounded in the
visual evidence.

**Detected Type:** {hallucination_type}

**Image–Answer Similarity:**
{image_answer_similarity:.4f}

**Grounding Score:**
{multimodal_grounding_score:.4f}

**Hallucination Risk:**
{risk_percent:.1f}%

The calculated risk exceeds the firewall threshold
of **{THRESHOLD:.2f}**.

**Recommendation:** Review the AI answer against
the uploaded image before accepting it.
"""
        )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader("⚙️ Verification Pipeline")

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.info(
            "📷 **IMAGE**\n\n"
            "Visual Evidence"
        )

    with p2:

        st.info(
            "🧠 **CLIP**\n\n"
            "Multimodal Features"
        )

    with p3:

        st.info(
            "⚡ **SVM**\n\n"
            "Risk Prediction"
        )

    with p4:

        if decision == "SUPPORTED":

            st.success(
                "🟢 **SUPPORTED**\n\n"
                "Verified"
            )

        else:

            st.error(
                "🔴 **FLAGGED**\n\n"
                "Potential Hallucination"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("⚙️ Model Information")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:

    st.metric(
        "Vision-Language Model",
        "CLIP ViT-B/32"
    )

with model_col2:

    st.metric(
        "Classifier",
        "SVM"
    )

with model_col3:

    st.metric(
        "Firewall Threshold",
        f"{THRESHOLD:.2f}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ Multimodal Hallucination Firewall | "
    "CLIP-based multimodal grounding + SVM hallucination detection"
)

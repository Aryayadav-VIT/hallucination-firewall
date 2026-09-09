import streamlit as st
import torch
import open_clip
import numpy as np
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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

body {
    background-color: #07111F;
}

.stApp {
    background-color: #07111F;
    color: #F8FAFC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main hero */

.hero {
    background: linear-gradient(135deg, #0F1B2D, #111827);
    border: 1px solid #23405F;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 17px;
    color: #94A3B8;
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    background-color: #172554;
    border: 1px solid #2563EB;
    color: #60A5FA;
    border-radius: 20px;
    padding: 6px 14px;
    margin-right: 8px;
    font-size: 13px;
    font-weight: 600;
}

.online {
    display: inline-block;
    background-color: #052E1A;
    border: 1px solid #16A34A;
    color: #4ADE80;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
}

/* Section titles */

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #F8FAFC;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Cards */

.card {
    background-color: #0F1B2D;
    border: 1px solid #23405F;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #F8FAFC;
}

.card-text {
    font-size: 14px;
    color: #94A3B8;
}

/* Inputs */

.stTextInput input,
.stTextArea textarea {
    background-color: #0B1728 !important;
    color: #F8FAFC !important;
    border: 1px solid #315579 !important;
    border-radius: 10px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #3B82F6 !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #0B1728;
    border-radius: 12px;
    border: 1px dashed #315579;
}

/* Button */

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    box-shadow: 0px 8px 25px rgba(59, 130, 246, 0.3);
}

/* Verdict */

.verdict-supported {
    background-color: #082818;
    border: 1px solid #16A34A;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-top: 20px;
}

.verdict-hallucinated {
    background-color: #2A0D12;
    border: 1px solid #DC2626;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-top: 20px;
}

.verdict-title {
    font-size: 34px;
    font-weight: 800;
    color: #F8FAFC;
    margin: 10px;
}

.verdict-description {
    font-size: 17px;
    color: #CBD5E1;
}

/* Risk */

.risk-card {
    background-color: #0F1B2D;
    border: 1px solid #23405F;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.risk-number {
    font-size: 38px;
    font-weight: 800;
    color: #F8FAFC;
}

.risk-label {
    color: #94A3B8;
}

/* Features */

.feature-card {
    background-color: #0B1728;
    border: 1px solid #23405F;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.feature-title {
    color: #94A3B8;
    font-size: 13px;
    margin-bottom: 8px;
}

.feature-value {
    color: #F8FAFC;
    font-size: 25px;
    font-weight: 750;
}

/* Pipeline */

.pipeline {
    background-color: #0F1B2D;
    border: 1px solid #23405F;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748B;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #1E293B;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ Firewall")

    st.markdown("---")

    st.markdown("### Navigation")

    st.markdown("🏠 **Verify Answer**")
    st.markdown("🔬 **Evidence Analysis**")
    st.markdown("🧠 **How It Works**")
    st.markdown("⚙️ **Model Information**")

    st.markdown("---")

    st.markdown("### System Status")

    st.success("● CLIP Engine Ready")
    st.success("● SVM Classifier Ready")

    st.markdown("---")

    st.caption("Multimodal Hallucination Firewall")
    st.caption("AI Answer Verification System")


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "hallucination_firewall_svm.pkl"


# ============================================================
# LOAD SVM MODEL
# ============================================================

@st.cache_resource
def load_firewall_model():

    model = joblib.load(MODEL_PATH)

    return model


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
# FEATURE EXTRACTION
# ============================================================

def calculate_features(image, question, answer):

    clip_model, preprocess, tokenizer, device = load_clip_model()

    image_input = preprocess(image).unsqueeze(0).to(device)

    question_tokens = tokenizer([question]).to(device)

    answer_tokens = tokenizer([answer]).to(device)

    with torch.no_grad():

        image_features = clip_model.encode_image(
            image_input
        )

        question_features = clip_model.encode_text(
            question_tokens
        )

        answer_features = clip_model.encode_text(
            answer_tokens
        )

        # Normalize embeddings

        image_features = (
            image_features /
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        question_features = (
            question_features /
            question_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        answer_features = (
            answer_features /
            answer_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        # Image-question similarity

        image_question_similarity = (
            image_features @ question_features.T
        ).item()

        # Image-answer similarity

        image_answer_similarity = (
            image_features @ answer_features.T
        ).item()

        # Overall grounding

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
# LOAD MODEL
# ============================================================

try:

    firewall_model = load_firewall_model()

except Exception as e:

    st.error(
        f"❌ Could not load firewall model: {e}"
    )

    st.stop()


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🛡️ Multimodal Hallucination Firewall
    </div>

    <div class="hero-subtitle">
        Evidence-Based AI Answer Verification & Grounding Engine
    </div>

    <span class="badge">
        CLIP + SVM
    </span>

    <span class="online">
        ● SYSTEM ONLINE
    </span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Verify an AI-Generated Answer</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)


# ============================================================
# IMAGE INPUT
# ============================================================

with left:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            📷 Visual Evidence
        </div>

        <div class="card-text">
            Upload the image that should support the AI answer.
        </div>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed"
    )

    image = None

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Visual Evidence",
            use_container_width=True
        )


# ============================================================
# QUESTION + ANSWER
# ============================================================

with right:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            ❓ Question
        </div>

    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Question",
        placeholder="Example: Is there a dog in the image?",
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🤖 AI-Generated Answer
        </div>

    </div>
    """, unsafe_allow_html=True)

    answer = st.text_area(
        "AI Answer",
        placeholder="Example: Yes, there is a dog in the image.",
        height=130,
        label_visibility="collapsed"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

run_analysis = st.button(
    "🔍  RUN FIREWALL ANALYSIS",
    use_container_width=True
)


# ============================================================
# FIREWALL ANALYSIS
# ============================================================

if run_analysis:

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if image is None:

        st.warning(
            "📷 Please upload an image."
        )

        st.stop()

    if not question.strip():

        st.warning(
            "❓ Please enter a question."
        )

        st.stop()

    if not answer.strip():

        st.warning(
            "🤖 Please enter the AI-generated answer."
        )

        st.stop()


    # --------------------------------------------------------
    # CLIP PROCESSING
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Analyzing image and textual evidence..."
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

    X_input = np.array([
        [
            image_question_similarity,
            image_answer_similarity,
            multimodal_grounding_score
        ]
    ])


    # --------------------------------------------------------
    # SVM PREDICTION
    # --------------------------------------------------------

    prediction = firewall_model.predict(
        X_input
    )[0]


    # --------------------------------------------------------
    # HALLUCINATION PROBABILITY
    # --------------------------------------------------------

    if hasattr(
        firewall_model,
        "predict_proba"
    ):

        probabilities = firewall_model.predict_proba(
            X_input
        )[0]

        classes = list(
            firewall_model.classes_
        )

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
    # FIREWALL THRESHOLD
    # --------------------------------------------------------

    THRESHOLD = 0.30


    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if hallucination_probability >= THRESHOLD:

        decision = "HALLUCINATED"

        verdict_icon = "🔴"

        verdict_description = (
            "The AI answer is not sufficiently "
            "supported by the visual evidence."
        )

    else:

        decision = "SUPPORTED"

        verdict_icon = "🟢"

        verdict_description = (
            "The AI answer is consistent "
            "with the visual evidence."
        )


    # ========================================================
    # FIREWALL VERDICT
    # ========================================================

    st.markdown(
        '<div class="section-title">🛡️ Firewall Verdict</div>',
        unsafe_allow_html=True
    )


    if decision == "SUPPORTED":

        st.markdown(
            f"""
            <div class="verdict-supported">

                <div style="color:#94A3B8;">
                    FIREWALL DECISION
                </div>

                <div class="verdict-title">
                    {verdict_icon} SUPPORTED
                </div>

                <div class="verdict-description">
                    {verdict_description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="verdict-hallucinated">

                <div style="color:#94A3B8;">
                    FIREWALL DECISION
                </div>

                <div class="verdict-title">
                    {verdict_icon} HALLUCINATED
                </div>

                <div class="verdict-description">
                    {verdict_description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # RISK
    # ========================================================

    risk_percent = hallucination_probability * 100


    st.markdown(
        '<div class="section-title">📊 Hallucination Risk</div>',
        unsafe_allow_html=True
    )


    risk_left, risk_right = st.columns([1, 2])


    with risk_left:

        st.markdown(
            f"""
            <div class="risk-card">

                <div class="risk-number">
                    {risk_percent:.1f}%
                </div>

                <div class="risk-label">
                    Hallucination Probability
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with risk_right:

        st.markdown("<br>", unsafe_allow_html=True)

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
    # EVIDENCE FEATURES
    # ========================================================

    st.markdown(
        '<div class="section-title">🔬 Evidence Analysis</div>',
        unsafe_allow_html=True
    )


    feature1, feature2, feature3 = st.columns(3)


    with feature1:

        st.markdown(
            f"""
            <div class="feature-card">

                <div class="feature-title">
                    IMAGE ↔ QUESTION
                </div>

                <div class="feature-value">
                    {image_question_similarity:.4f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with feature2:

        st.markdown(
            f"""
            <div class="feature-card">

                <div class="feature-title">
                    IMAGE ↔ ANSWER
                </div>

                <div class="feature-value">
                    {image_answer_similarity:.4f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with feature3:

        st.markdown(
            f"""
            <div class="feature-card">

                <div class="feature-title">
                    MULTIMODAL GROUNDING
                </div>

                <div class="feature-value">
                    {multimodal_grounding_score:.4f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 Why did the Firewall decide this?</div>',
        unsafe_allow_html=True
    )


    if decision == "SUPPORTED":

        explanation = (
            f"The AI answer demonstrates sufficient alignment "
            f"with the visual evidence. The multimodal grounding "
            f"score is <b>{multimodal_grounding_score:.4f}</b> "
            f"and the hallucination risk is below the firewall "
            f"threshold of <b>{THRESHOLD:.2f}</b>."
        )

    else:

        explanation = (
            f"The AI answer demonstrates insufficient alignment "
            f"with the visual evidence. The multimodal grounding "
            f"score is <b>{multimodal_grounding_score:.4f}</b> "
            f"and the hallucination risk exceeds the firewall "
            f"threshold of <b>{THRESHOLD:.2f}</b>."
        )


    st.info(
        explanation,
        icon="🧠"
    )


    # ========================================================
    # VERIFICATION PIPELINE
    # ========================================================

    st.markdown(
        '<div class="section-title">⚙️ Verification Pipeline</div>',
        unsafe_allow_html=True
    )


    pipe1, pipe2, pipe3, pipe4 = st.columns(4)


    with pipe1:

        st.markdown(
            """
            <div class="pipeline">

                <div style="font-size:30px;">
                    📷
                </div>

                <b>IMAGE</b>

                <br>

                <small>
                    Visual Evidence
                </small>

            </div>
            """,
            unsafe_allow_html=True
        )


    with pipe2:

        st.markdown(
            """
            <div class="pipeline">

                <div style="font-size:30px;">
                    🧠
                </div>

                <b>CLIP</b>

                <br>

                <small>
                    Feature Extraction
                </small>

            </div>
            """,
            unsafe_allow_html=True
        )


    with pipe3:

        st.markdown(
            """
            <div class="pipeline">

                <div style="font-size:30px;">
                    ⚡
                </div>

                <b>SVM</b>

                <br>

                <small>
                    Risk Prediction
                </small>

            </div>
            """,
            unsafe_allow_html=True
        )


    with pipe4:

        if decision == "SUPPORTED":

            st.markdown(
                """
                <div class="pipeline">

                    <div style="font-size:30px;">
                        🟢
                    </div>

                    <b>SUPPORTED</b>

                    <br>

                    <small>
                        Verified
                    </small>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="pipeline">

                    <div style="font-size:30px;">
                        🔴
                    </div>

                    <b>FLAGGED</b>

                    <br>

                    <small>
                        Potential Hallucination
                    </small>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Model Information</div>',
    unsafe_allow_html=True
)


model1, model2, model3 = st.columns(3)


with model1:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-title">
                VISION-LANGUAGE MODEL
            </div>

            <div class="feature-value">
                CLIP ViT-B/32
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with model2:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-title">
                CLASSIFIER
            </div>

            <div class="feature-value">
                SVM
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with model3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-title">
                DECISION THRESHOLD
            </div>

            <div class="feature-value">
                0.30
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🛡️ Multimodal Hallucination Firewall
        <br><br>
        CLIP-based multimodal grounding + SVM hallucination detection

    </div>
    """,
    unsafe_allow_html=True
)

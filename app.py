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
# TITLE
# ============================================================

st.title("🛡️ Multimodal Hallucination Firewall")

st.markdown(
    """
    ### Evidence-Based AI Answer Verification

    Upload an image, enter a question, and provide an AI-generated
    answer. The system automatically extracts multimodal evidence
    features using CLIP and uses a trained SVM classifier to detect
    potential hallucinations.
    """
)

st.divider()


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "hallucination_firewall_svm.pkl"


# ============================================================
# LOAD SVM
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

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_input = preprocess(image).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    question_tokens = tokenizer([question]).to(device)
    answer_tokens = tokenizer([answer]).to(device)

    # --------------------------------------------------------
    # CLIP EMBEDDINGS
    # --------------------------------------------------------

    with torch.no_grad():

        image_features = clip_model.encode_image(image_input)

        question_features = clip_model.encode_text(question_tokens)

        answer_features = clip_model.encode_text(answer_tokens)

        # Normalize embeddings
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        question_features = question_features / question_features.norm(
            dim=-1,
            keepdim=True
        )

        answer_features = answer_features / answer_features.norm(
            dim=-1,
            keepdim=True
        )

        # ----------------------------------------------------
        # IMAGE-QUESTION SIMILARITY
        # ----------------------------------------------------

        image_question_similarity = (
            image_features @ question_features.T
        ).item()

        # ----------------------------------------------------
        # IMAGE-ANSWER SIMILARITY
        # ----------------------------------------------------

        image_answer_similarity = (
            image_features @ answer_features.T
        ).item()

        # ----------------------------------------------------
        # MULTIMODAL GROUNDING SCORE
        # ----------------------------------------------------

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
# LOAD FIREWALL MODEL
# ============================================================

try:

    firewall_model = load_firewall_model()

    st.success("✅ Hallucination Firewall model loaded")

except Exception as e:

    st.error(f"❌ Could not load firewall model: {e}")

    st.stop()


# ============================================================
# USER INPUT
# ============================================================

st.subheader("📷 1. Upload Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )


st.subheader("❓ 2. Enter Question")

question = st.text_input(
    "Question",
    placeholder="Example: Is there a dog in the image?"
)


st.subheader("🤖 3. Enter AI-Generated Answer")

answer = st.text_area(
    "AI Answer",
    placeholder="Example: Yes, there is a dog in the image.",
    height=120
)


st.divider()


# ============================================================
# CHECK ANSWER
# ============================================================

if st.button(
    "🔍 CHECK ANSWER",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning("Please upload an image.")

        st.stop()

    if not question.strip():

        st.warning("Please enter a question.")

        st.stop()

    if not answer.strip():

        st.warning("Please enter the AI-generated answer.")

        st.stop()


    # --------------------------------------------------------
    # CALCULATE CLIP FEATURES
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing image and extracting CLIP evidence features..."
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
    # PREPARE FEATURES FOR SVM
    # --------------------------------------------------------

    X_input = np.array([[
        image_question_similarity,
        image_answer_similarity,
        multimodal_grounding_score
    ]])


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = firewall_model.predict(X_input)[0]


    # --------------------------------------------------------
    # HALLUCINATION RISK
    # --------------------------------------------------------

    if hasattr(firewall_model, "predict_proba"):

        probabilities = firewall_model.predict_proba(X_input)[0]

        classes = list(firewall_model.classes_)

        # Project convention:
        # class 1 = supported
        # class 0 = hallucinated

        if 0 in classes:

            hallucination_probability = probabilities[
                classes.index(0)
            ]

        else:

            hallucination_probability = 1 - probabilities[
                classes.index(1)
            ]

    else:

        hallucination_probability = 0.0


    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    THRESHOLD = 0.30

    if hallucination_probability >= THRESHOLD:

        decision = "HALLUCINATED"

    else:

        decision = "SUPPORTED"


    # ========================================================
    # DISPLAY FEATURES
    # ========================================================

    st.subheader("🧠 CLIP-Derived Evidence Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Image–Question Similarity",
            f"{image_question_similarity:.4f}"
        )

    with col2:

        st.metric(
            "Image–Answer Similarity",
            f"{image_answer_similarity:.4f}"
        )

    with col3:

        st.metric(
            "Multimodal Grounding Score",
            f"{multimodal_grounding_score:.4f}"
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🛡️ Firewall Decision")


    if decision == "HALLUCINATED":

        st.error(
            "🔴 HALLUCINATED / POTENTIALLY UNSUPPORTED"
        )

    else:

        st.success(
            "🟢 SUPPORTED / GROUNDED"
        )


    # ========================================================
    # RISK SCORE
    # ========================================================

    st.metric(
        "Hallucination Risk",
        f"{hallucination_probability * 100:.2f}%"
    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.info(
        f"""
        **How the decision was made**

        CLIP compared the uploaded image with the question and
        generated answer.

        The resulting evidence features were passed to the
        trained SVM hallucination classifier.

        **Decision threshold:** {THRESHOLD}

        **Firewall result:** {decision}
        """
    )

"""
Multimodal Hallucination Firewall: An Evidence-Aware System for Detecting Unsupported Claims in Image-Based AI Answers.
Streamlit Application & Research Prototype.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Setup import paths
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import (
    DEFAULT_THRESHOLD,
    HallucinationCategory,
    VerificationStatus,
    EXPECTED_FEATURES,
    DEMO_IMAGES_DIR,
)
from src.model_loader import find_model_path, load_bundle
from src.question_parser import parse_question_and_answer, ParsedClaim
from src.hallucination_checks import verify_claim, VerificationResult
from src.demo_cases import DEMO_CASES, get_demo_case_by_id

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Multimodal Hallucination Firewall",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark navy, teal, and slate palette
st.markdown(
    """
    <style>
    :root {
        --bg-main: #0F172A;
        --card-bg: #1E293B;
        --accent-teal: #0D9488;
        --accent-cyan: #0EA5E9;
        --text-light: #F8FAFC;
        --text-muted: #94A3B8;
        --border-color: #334155;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-supported {
        background-color: #064E3B;
        color: #6EE7B7;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-hallucinated {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-inconclusive {
        background-color: #78350F;
        color: #FCD34D;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-unsupported {
        background-color: #1E293B;
        color: #93C5FD;
        border: 1px solid #3B82F6;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "query_question" not in st.session_state:
    st.session_state["query_question"] = "Is there a phone in the image?"
if "query_answer" not in st.session_state:
    st.session_state["query_answer"] = "Yes, there is a smartphone visible."
if "query_image_path" not in st.session_state:
    st.session_state["query_image_path"] = None
if "active_demo_title" not in st.session_state:
    st.session_state["active_demo_title"] = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🛡️ Firewall Controls")
    st.caption("Evidence-Aware Multimodal Verification")

    threshold = st.slider(
        "Hallucination Decision Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Higher thresholds require stronger statistical risk before flagging as hallucination. Default is 0.27.",
    )

    st.markdown("---")
    st.subheader("Model Status")
    try:
        model_path = find_model_path()
        st.success(f"Bundle: `{model_path.name}`")
        bundle = load_bundle(str(model_path))
        st.caption(f"Classifier: SVM RBF (8 features)")
        st.caption(f"Calibrated Threshold: {bundle.get('threshold', 0.27):.2f}")
    except Exception as e:
        st.error(f"Model error: {e}")

    st.markdown("---")
    st.caption(
        "MSc AI & ML Research Prototype\n"
        "Evaluates Vision-Language claim alignment."
    )

# ============================================================
# MAIN NAVIGATION TABS
# ============================================================
tabs = st.tabs([
    "🔍 Live Analysis",
    "🧪 Demo Lab / Playground",
    "🛡️ Overview & Scope",
    "📊 Model Architecture",
    "📖 Methodology & Limits"
])

# ------------------------------------------------------------
# TAB 1: LIVE ANALYSIS
# ------------------------------------------------------------
with tabs[0]:
    st.header("🔍 Multimodal Hallucination Analysis")
    st.write(
        "Upload an image, specify an inquiry, and provide the candidate AI answer. "
        "The firewall classifies the query category, extracts vision-language evidence, and evaluates potential hallucinations."
    )

    col_img, col_query = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("1. Visual Evidence")
        uploaded_file = st.file_uploader(
            "Upload image (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"]
        )

        active_image = None
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
            st.image(active_image, caption="Uploaded Image", use_container_width=True)
        elif st.session_state["query_image_path"]:
            p = Path(st.session_state["query_image_path"])
            if p.is_file():
                active_image = Image.open(p).convert("RGB")
                st.image(
                    active_image,
                    caption=f"Loaded Demo Image: {p.name}",
                    use_container_width=True
                )

        if active_image is None:
            st.info("Please upload an image or load one from the Demo Lab.")

    with col_query:
        st.subheader("2. Query & Candidate Answer")

        if st.session_state.get("active_demo_title"):
            st.info(f"Loaded from Demo Lab: **{st.session_state['active_demo_title']}**")

        input_q = st.text_input(
            "Inquiry / Question",
            value=st.session_state["query_question"],
            placeholder="e.g. Is there a cat in the image? / What color is the car?",
        )

        input_a = st.text_area(
            "Candidate AI-Generated Answer",
            value=st.session_state["query_answer"],
            placeholder="e.g. Yes, there is a cat sitting on the couch.",
            height=120,
        )

        # Real-time category detection preview
        parsed_preview = parse_question_and_answer(input_q, input_a)
        cat_badge = f"`Category: {parsed_preview.category}`"
        target_disp = f"Target: **{parsed_preview.target_object or 'None'}**"
        pol_disp = "Claim: **Positive**" if parsed_preview.claim_polarity == 1 else ("Claim: **Negative**" if parsed_preview.claim_polarity == 0 else "Claim: **Ambiguous**")
        st.caption(f"{cat_badge} | {target_disp} | {pol_disp}")

        run_btn = st.button("🛡️ RUN FIREWALL ANALYSIS", type="primary", use_container_width=True)

    if run_btn:
        if active_image is None:
            st.warning("Please upload an image or select a demo case first.")
            st.stop()
        if not input_q.strip() or not input_a.strip():
            st.warning("Please enter both the question and the candidate answer.")
            st.stop()

        with st.spinner("Analyzing multimodal evidence and evaluating claims..."):
            try:
                bundle = load_bundle()
                claim = parse_question_and_answer(input_q, input_a)
                result = verify_claim(
                    image=active_image,
                    claim=claim,
                    threshold=threshold,
                    model_bundle=bundle,
                )

                st.markdown("---")
                st.subheader("🛡️ Firewall Verdict & Evidence Report")

                # Verdict Banner
                col_v1, col_v2 = st.columns([2, 1])
                with col_v1:
                    if result.status == VerificationStatus.LIKELY_SUPPORTED:
                        st.markdown(
                            f"<div class='status-badge-supported'>✅ {result.status.upper()}</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Visual Alignment:** {result.evidence_summary}")
                    elif result.status == VerificationStatus.POTENTIAL_HALLUCINATION:
                        st.markdown(
                            f"<div class='status-badge-hallucinated'>⚠️ {result.status.upper()}</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Contradiction Detected:** {result.evidence_summary}")
                    elif result.status == VerificationStatus.INCONCLUSIVE:
                        st.markdown(
                            f"<div class='status-badge-inconclusive'>❓ {result.status.upper()}</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Ambiguous Evidence:** {result.evidence_summary}")
                    else:
                        st.markdown(
                            f"<div class='status-badge-unsupported'>ℹ️ {result.status.upper()}</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Boundary Notice:** {result.evidence_summary}")

                with col_v2:
                    if result.risk_score is not None:
                        st.metric("Estimated Hallucination Risk", f"{result.risk_score:.1%}")
                        st.progress(min(max(result.risk_score, 0.0), 1.0))
                        st.caption(f"Decision Threshold: {threshold:.2f}")
                    else:
                        st.metric("Classifier Scope", "Heuristic / Rule")

                st.markdown("#### Evidence & Verification Breakdown")
                st.write(f"**Verification Method:** `{result.method_used}`")

                # Display 8 features table if object existence
                if result.category == HallucinationCategory.OBJECT_EXISTENCE and "object_presence_margin" in result.detailed_metrics:
                    m = result.detailed_metrics
                    f_col1, f_col2, f_col3 = st.columns(3)
                    f_col1.metric("Presence Margin", f"{m['object_presence_margin']:+.4f}")
                    f_col2.metric("Positive Prompt Sim", f"{m['object_positive_similarity']:.4f}")
                    f_col3.metric("Negative Prompt Sim", f"{m['object_negative_similarity']:.4f}")

                    with st.expander("📊 Inspect Full 8-Feature Vector"):
                        ft_data = [
                            {"Feature": name, "Value": f"{m.get(name, 0.0):.4f}"}
                            for name in EXPECTED_FEATURES
                        ]
                        st.dataframe(pd.DataFrame(ft_data), hide_index=True, use_container_width=True)

                # Limitations
                if result.limitations:
                    st.markdown("##### ⚠️ Research Limitations & Confidence Bounds")
                    for lim in result.limitations:
                        st.caption(f"• {lim}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

# ------------------------------------------------------------
# TAB 2: DEMO LAB / PLAYGROUND
# ------------------------------------------------------------
with tabs[1]:
    st.header("🧪 Hallucination Demonstration Lab")
    st.write(
        "Explore curated test cases across 12 distinct multimodal hallucination categories. "
        "Select any case to inspect its ground truth, expected behavior, and click **Load into Live Analysis** to run it through the live firewall pipeline."
    )

    selected_case_id = st.selectbox(
        "Choose a Curated Demonstration Case:",
        options=[case["id"] for case in DEMO_CASES],
        format_func=lambda cid: next((c["title"] for c in DEMO_CASES if c["id"] == cid), cid)
    )

    case = get_demo_case_by_id(selected_case_id)
    if case:
        d_col1, d_col2 = st.columns([1, 1], gap="large")

        with d_col1:
            img_path = DEMO_IMAGES_DIR / case["image_file"]
            if img_path.is_file():
                st.image(str(img_path), caption=f"Curated Case Asset: {case['image_file']}", use_container_width=True)
            else:
                st.warning(f"Image asset missing: {img_path}")

        with d_col2:
            st.subheader(case["title"])
            st.markdown(f"**Category:** `{case['category']}`")
            st.markdown(f"**Inquiry:** *\"{case['question']}\"*")
            st.markdown(f"**AI Candidate Answer:** *\"{case['answer']}\"*")

            st.markdown("---")
            st.markdown(f"**Ground Truth Context:** {case['intended_ground_truth']}")
            st.markdown(f"**Expected Status:** `{case['expected_status']}`")
            st.markdown(f"**Observation:** {case['expected_observation']}")
            st.markdown(f"**Method Actually Available:** `{case['method_available']}`")

            st.caption(f"**Limitation Note:** {case['limitation_note']}")

            if st.button("🚀 Load into Live Analysis", key=f"load_{case['id']}", type="primary"):
                st.session_state["query_question"] = case["question"]
                st.session_state["query_answer"] = case["answer"]
                st.session_state["query_image_path"] = str(img_path)
                st.session_state["active_demo_title"] = case["title"]
                st.success("Loaded! Switch to the '🔍 Live Analysis' tab to run the prediction.")

# ------------------------------------------------------------
# TAB 3: OVERVIEW & SCOPE
# ------------------------------------------------------------
with tabs[2]:
    st.header("🛡️ Multimodal Hallucination Firewall Overview")
    st.write(
        "Large Vision-Language Models (VLMs) frequently generate answers that are fluent and semantically plausible, "
        "yet unfaithful to the visual evidence present in the image. This research prototype provides an evidence-aware "
        "firewall to detect, quantify, and explain these hallucinations."
    )

    st.subheader("Capabilities Matrix")
    cap_data = [
        {"Category": "Object Existence", "Supported?": "✅ Fully Supported", "Method": "8-feature SVM on POPE benchmark (ViT-B/32)", "Confidence Level": "Statistical Probability"},
        {"Category": "Visual Attributes (Color)", "Supported?": "🟡 Supported (Heuristic)", "Method": "Contrastive Prompt Ranking", "Confidence Level": "Relative Prompt Margin"},
        {"Category": "Action / Activity", "Supported?": "🟡 Supported (Heuristic)", "Method": "Zero-Shot CLIP Action Matching", "Confidence Level": "Semantic Similarity"},
        {"Category": "Scene / Context", "Supported?": "🟡 Supported (Heuristic)", "Method": "Zero-Shot Scene Classification", "Confidence Level": "Cosine Similarity"},
        {"Category": "Spatial Relations", "Supported?": "⚠️ Experimental", "Method": "Relational Contrast Prompting", "Confidence Level": "Uncertain (Compositionality Limits)"},
        {"Category": "Object Counting", "Supported?": "❌ Unsupported / Flagged", "Method": "Limitation Detection Filter", "Confidence Level": "Requires Instance Segmenter"},
        {"Category": "Text Reading / OCR", "Supported?": "❌ Unsupported / Flagged", "Method": "Signage Concept Check", "Confidence Level": "Requires Dedicated OCR"},
        {"Category": "External Speculation", "Supported?": "❌ Refused (Out of Pixel Bounds)", "Method": "Pixel Boundary Domain Filter", "Confidence Level": "Strictly Unverifiable"}
    ]
    st.dataframe(pd.DataFrame(cap_data), hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# TAB 4: MODEL ARCHITECTURE
# ------------------------------------------------------------
with tabs[3]:
    st.header("📊 Model Architecture & Training Methodology")
    st.write(
        "The primary object-existence detector combines a pretrained vision-language encoder with a leakage-safe supervised SVM classifier."
    )

    st.subheader("The 8 Multimodal Feature Definitions")
    features_desc = [
        {"Feature": "image_question_similarity", "Formula": "cos(v_img, v_q)", "Description": "Cosine similarity between normalized image embedding and question text."},
        {"Feature": "image_answer_similarity", "Formula": "cos(v_img, v_a)", "Description": "Cosine similarity between normalized image embedding and candidate answer text."},
        {"Feature": "multimodal_grounding_score", "Formula": "(sim_q + sim_a) / 2", "Description": "Average visual-textual semantic grounding across question and answer."},
        {"Feature": "object_positive_similarity", "Formula": "mean(cos(v_img, v_pos_prompts))", "Description": "Average similarity against 4 affirmative presence prompt templates."},
        {"Feature": "object_negative_similarity", "Formula": "mean(cos(v_img, v_neg_prompts))", "Description": "Average similarity against 4 absence prompt templates."},
        {"Feature": "object_presence_margin", "Formula": "pos_sim - neg_sim", "Description": "Directional visual evidence margin. Positive indicates presence, negative indicates absence."},
        {"Feature": "answer_claim_present", "Formula": "1 (claim present) or 0 (claim absent)", "Description": "Parsed semantic assertion polarity of the candidate AI answer."},
        {"Feature": "claim_evidence_alignment", "Formula": "sign(claim) * presence_margin", "Description": "Alignment score. Positive values indicate agreement; negative values signal hallucination."},
    ]
    st.dataframe(pd.DataFrame(features_desc), hide_index=True, use_container_width=True)

    st.subheader("Classifier Pipeline")
    st.write(
        "- **Encoder:** OpenAI CLIP ViT-B/32 (L2 normalized 512-dim embeddings)\n"
        "- **Preprocessor:** StandardScaler (z-score normalization across all 8 features)\n"
        "- **Classifier:** Support Vector Machine (SVC) with Radial Basis Function (RBF) kernel\n"
        "- **Calibration:** Platt scaling (`probability=True`) producing calibrated risk scores\n"
        "- **Optimal Threshold:** `0.27` calibrated on the validation set for maximum F1-Score"
    )

# ------------------------------------------------------------
# TAB 5: METHODOLOGY & LIMITS
# ------------------------------------------------------------
with tabs[4]:
    st.header("📖 Research Methodology & Scientific Limitations")
    st.markdown(
        """
        ### Why Vision-Language Models Hallucinate
        Vision-Language Models generate answers autoregressively. When language prior probabilities overpower weak visual tokens, 
        the model generates objects that are typically associated with the scene rather than objects that are genuinely present 
        (e.g., claiming a telephone is on an office desk simply because telephones frequently co-occur with desks).

        ### Fundamental Limitations of CLIP-Based Heuristics
        1. **Compositionality Blindness:** CLIP treats images and prompts largely as bags-of-visual-words. It frequently struggles to distinguish *'a red car and a blue truck'* from *'a blue car and a red truck'*, or *'the cat on the box'* from *'the box on the cat'*.
        2. **Lack of Counting Heads:** Contrastive similarity does not count instances. An image with three apples will match the prompt *'two apples'* with nearly identical cosine similarity.
        3. **Resolution & Occlusion:** Distant, blurred, or occluded items produce margins near zero, necessitating explicit **Inconclusive** handling rather than arbitrary classification.

        ### Citation & Academic Attribution
        - **POPE Benchmark:** Li et al., *Evaluating Object Hallucination in Large Vision-Language Models*, EMNLP 2023.
        - **CLIP:** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
        """
    )

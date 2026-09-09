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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #07111F;
    color: #F8FAFC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label {
    color: #CBD5E1 !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #081321;
    border-right: 1px solid #1E3A5F;
}

/* Hero */

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 17px;
    color: #94A3B8;
    margin-bottom: 20px;
}

/* Cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0F1B2D;
    border-color: #23405F !important;
    border-radius: 16px;
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
    box-shadow: 0 0 0 1px #3B82F6 !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #0B1728;
    border: 1px dashed #315579;
    border-radius: 12px;
}

/* Main button */

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.30);
}

/* Metrics */

div[data-testid="stMetric"] {
    background-color: #0B1728;
    border: 1px solid #23405F;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

/* Divider */

hr {
    border-color: #1E3A5F;
}

/* Footer */

.footer-text {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ Firewall")

    st.divider()

    st.markdown("### Navigation")

    st.markdown("🏠 **Verify Answer**")
    st.markdown("🔬 **Evidence Analysis**")
    st.markdown("🧠 **How It Works**")
    st.markdown("⚙️ **Model Information**")

    st.divider()

    st.markdown("### System Status")

    st.success("🟢 CLIP Engine Ready")
    st.success("🟢 SVM Classifier Ready")

    st.divider()

    st.caption("Multimodal Hallucination Firewall")
    st.caption("AI Answer Verification System")
    st.title("TEST - Multimodal Hallucination Firewall")
    st.write("If you can see this, the main page is working.")

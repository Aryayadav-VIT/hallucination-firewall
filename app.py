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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(59,130,246,0.10), transparent 25%),
            radial-gradient(circle at 90% 20%, rgba(139,92,246,0.08), transparent 25%),
            #07111F;
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    p, label {
        color: #CBD5E1 !important;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid #1E3A5F;
    }

    section[data-testid="stSidebar"] h1 {
        color: #F8FAFC !important;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            rgba(15,27,45,0.95),
            rgba(17,24,39,0.92)
        );
        border: 1px solid #1E3A5F;
        border-radius: 22px;
        padding: 30px 35px;
        margin-bottom: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 8px;
        color: #F8FAFC;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 18px;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.35);
        color: #60A5FA;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }

    .online {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.30);
        color: #4ADE80;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: rgba(15,27,45,0.88);
        border: 1px solid #1E3A5F;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
    }

    .card-description {
        color: #94A3B8;
        font-size: 0.9rem;
    }

    /* ---------- INPUTS ---------- */

    .stTextInput input,
    .stTextArea textarea {
        background: #0B1728 !important;
        color: #F8FAFC !important;
        border: 1px solid #274568 !important;
        border-radius: 12px !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }

    /* ---------- UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #0B1728;
        border: 1px dashed #315579;
        border-radius: 14px;
        padding: 8px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 13px 20px;
        font-size: 1rem;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.30);
    }

    /* ---------- VERDICT ---------- */

    .verdict {
        border-radius: 22px;
        padding: 35px;
        text-align: center;
        margin: 25px 0;
    }

    .verdict-supported {
        background: linear-gradient(
            135deg,
            rgba(20,83,45,0.45),
            rgba(15,27,45,0.95)
        );
        border: 1px solid rgba(34,197,94,0.45);
        box-shadow: 0 15px 45px rgba(34,197,94,0.08);
    }

    .verdict-hallucinated {
        background: linear-gradient(
            135deg,
            rgba(127,29,29,0.45),
            rgba(15,27,45,0.95)
        );
        border: 1px solid rgba(239,68,68,0.45);
        box-shadow: 0 15px 45px rgba(239,68,68,0.08);
    }

    .verdict-uncertain {
        background: linear-gradient(
            135deg,
            rgba(120,53,15,0.45),
            rgba(15,27,45,0.95)
        );
        border: 1px solid rgba(245,158,11,0.45);
    }

    .verdict-label {
        font-size: 2.2rem;
        font-weight: 850;
        margin: 8px 0 15px

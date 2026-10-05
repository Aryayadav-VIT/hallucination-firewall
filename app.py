
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

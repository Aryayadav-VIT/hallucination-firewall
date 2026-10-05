        st.metric(
            "Grounding Score",
            f"{multimodal_grounding_score:.4f}"
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader("🧠 Why did the Firewall decide this?")

    if object_contradiction:

        st.error(
            f"""
### 🔴 Object-level contradiction detected

The queried object is **{object_name}**.

The visual evidence indicates that the object is **present**,
but the AI answer claims that it is **absent**.

**Detected Type:** Object Presence Hallucination

**Object Evidence Confidence:** {object_confidence * 100:.1f}%

**SVM Hallucination Probability:** {risk_percent:.1f}%

The firewall therefore overrides the SVM-only decision and flags
the answer as a potential hallucination.
"""
        )

    elif decision == "SUPPORTED":

        st.success(
            f"""
### 🟢 Visual evidence supports the answer

The AI answer shows sufficient alignment with the uploaded image.

**Grounding Score:** {multimodal_grounding_score:.4f}

**SVM Hallucination Risk:** {risk_percent:.1f}%

**Firewall Status:** Supported

The calculated SVM risk is below the firewall threshold of
**{THRESHOLD:.2f}**, and no strong object-level contradiction was found.
"""
        )

    else:

        st.error(
            f"""
### 🔴 Potential hallucination detected

The AI answer is not sufficiently grounded in the visual evidence.

**Detected Type:** {hallucination_type}

**Image–Answer Similarity:** {image_answer_similarity:.4f}

**Grounding Score:** {multimodal_grounding_score:.4f}

**SVM Hallucination Risk:** {risk_percent:.1f}%

The calculated SVM risk exceeds the firewall threshold of
**{THRESHOLD:.2f}**.

**Recommendation:** Review the AI answer against the uploaded image before accepting it.
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
        st.info("🔎 **OBJECT**\n\nVisual Evidence")

    with p4:
        st.info("⚡ **SVM**\n\nRisk Prediction")

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
    st.metric("Firewall Threshold", f"{THRESHOLD:.2f}")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🛡️ Multimodal Hallucination Firewall | "

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

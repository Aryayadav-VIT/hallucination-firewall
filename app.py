import streamlit as st

st.set_page_config(
    page_title="Multimodal Hallucination Firewall",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multimodal Hallucination Firewall")

st.subheader("AI Answer Verification")

st.write(
    "Upload an image, enter a question and an AI-generated answer "
    "to check whether the answer is supported by the visual evidence."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📷 Visual Evidence")
    st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

with col2:
    st.markdown("### ❓ Question")
    st.text_input(
        "Question",
        placeholder="Is there a dog in the image?"
    )

    st.markdown("### 🤖 AI Answer")
    st.text_area(
        "AI-generated answer",
        placeholder="Yes, there is a dog in the image."
    )

st.divider()

st.button(
    "🔍 RUN FIREWALL ANALYSIS",
    use_container_width=True
)

st.success("🟢 Main page is working correctly.")

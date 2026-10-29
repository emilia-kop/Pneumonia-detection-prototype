"""Streamlit app - Pneumonia Detection from Chest X-ray Images (educational prototype)."""
import streamlit as st

from pneumonia_model import (
    DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, SUPPORTED_WEIGHTS,
    InvalidImageError, ModelLoadError, PneumoniaClassifier, load_image,
)

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def get_classifier(weights: str) -> PneumoniaClassifier:
    return PneumoniaClassifier(weights=weights)  # loaded once, reused across reruns


st.title("🫁 Pneumonia Detection from Chest X-rays")
st.warning(
    "**Educational prototype - NOT a clinical diagnostic tool.** "
    "Do not use it for medical decisions. Always consult a qualified clinician."
)

with st.sidebar:
    st.header("Model settings")
    weights = st.selectbox("Pretrained weights", SUPPORTED_WEIGHTS,
                           index=SUPPORTED_WEIGHTS.index(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS in SUPPORTED_WEIGHTS else 0)
    threshold = st.slider("Decision threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05,
                          help="Score >= threshold is reported as PNEUMONIA.")
    st.caption(
        "DenseNet121 from TorchXRayVision (library code: Apache-2.0), trained on chest "
        "X-ray datasets. Multi-label model; we read its 'Pneumonia' output."
    )

uploaded = st.file_uploader("Upload a chest X-ray (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = load_image(uploaded)
    except InvalidImageError as e:
        st.error(f"Invalid image: {e}")
        st.stop()
    st.image(image, caption="Uploaded image")

    if st.button("Predict", type="primary"):
        try:
            clf = get_classifier(weights)
        except ModelLoadError as e:
            st.error(f"Model could not be loaded.\n\n{e}")
            st.stop()
        try:
            with st.spinner("Analysing..."):
                pred = clf.predict(image, threshold=threshold)
        except Exception as e:  # keep the app alive on any unexpected inference failure
            st.error(f"Prediction failed: {e}")
            st.stop()
        if pred.label == "PNEUMONIA":
            st.error(f"Predicted class: **{pred.label}**")
        else:
            st.success(f"Predicted class: **{pred.label}** (pneumonia not flagged)")
        st.metric("Pneumonia score (raw model output)", f"{pred.pneumonia_score:.3f}")
        st.caption(
            f"Threshold {pred.threshold:.2f}. This score is not a calibrated probability or "
            "confidence. Models trained on adult datasets can perform poorly on other "
            "populations (e.g. children) or image sources."
        )
        if pred.all_scores:
            with st.expander("All named model outputs (for transparency)"):
                st.caption("Real outputs of the loaded weights for this image. "
                           "Higher = more likely flagged by the model; not probabilities.")
                st.table({"Finding": list(pred.all_scores.keys()),
                          "Score": [f"{v:.3f}" for v in pred.all_scores.values()]})
else:
    st.info("Upload an image to begin.")

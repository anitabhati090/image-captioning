"""Streamlit front-end: upload an image, pick a style and language, see caption + heat-map."""

from __future__ import annotations

import streamlit as st

from captioning.config import AVAILABLE_MODELS, settings
from captioning.pipeline import CaptioningPipeline
from captioning.styles import STYLES
from captioning.translator import LANGUAGES
from captioning.utils import load_rgb

st.set_page_config(page_title="AI Image Captioning", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="Loading model (first run downloads the weights)...")
def get_model(name: str):
    from captioning.model import CaptionModel

    return CaptionModel(name)


st.title("AI-Powered Image Captioning")
st.caption("Multilingual & accessibility-focused captions with explainable attention maps")

with st.sidebar:
    st.header("Settings")
    default_idx = AVAILABLE_MODELS.index(settings.caption_model) if settings.caption_model in AVAILABLE_MODELS else 0
    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=default_idx)
    style = st.radio("Caption style", list(STYLES), format_func=lambda s: f"{s} - {STYLES[s]}")
    language = st.selectbox("Output language", list(LANGUAGES))
    show_xai = st.checkbox("Show explainability heat-map", value=True)
    xai_method = st.radio("XAI method", ["Grad-CAM", "Attention rollout"], disabled=not show_xai)
    target_word = st.text_input("Explain a specific word (optional)", disabled=not show_xai)

tab_caption, tab_eval = st.tabs(["Caption", "Evaluate"])

with tab_caption:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = load_rgb(uploaded)
        left, right = st.columns(2)
        left.image(image, caption="Input image", use_container_width=True)

        if st.button("Generate caption", type="primary"):
            model = get_model(model_name)
            with st.spinner("Generating..."):
                result = CaptioningPipeline(model).run(image, style, LANGUAGES[language])
            st.session_state["result"] = result
            st.session_state["xai"] = None
            if show_xai:
                try:
                    from captioning.xai import VisionExplainer

                    explainer = VisionExplainer(model)
                    if xai_method == "Grad-CAM":
                        _, overlay = explainer.grad_cam(image, result.base_caption, target_word or None)
                    else:
                        _, overlay = explainer.attention_rollout(image)
                    st.session_state["xai"] = overlay
                except Exception as exc:  # keep the app usable even if XAI fails
                    st.warning(f"Heat-map unavailable: {exc}")

        result = st.session_state.get("result")
        if result:
            with right:
                st.subheader("Caption")
                st.success(result.translated_caption)
                if language != "English":
                    st.caption(f"English: {result.styled_caption}")
                if result.translation_error:
                    st.warning(result.translation_error)
                st.caption(f"Base model output: {result.base_caption}  |  {result.latency_s}s")
                if st.session_state.get("xai") is not None:
                    st.image(st.session_state["xai"], caption=f"{xai_method} overlay", use_container_width=True)

with tab_eval:
    st.write("Score the generated caption against your own reference captions (one per line).")
    refs_text = st.text_area("Reference captions", height=120)
    if st.button("Compute metrics"):
        result = st.session_state.get("result")
        if not result or not uploaded:
            st.info("Generate a caption first (Caption tab).")
        elif not refs_text.strip():
            st.info("Enter at least one reference caption.")
        else:
            from captioning import evaluation

            refs = {"img": [r for r in refs_text.splitlines() if r.strip()]}
            preds = {"img": result.styled_caption}
            with st.spinner("Computing metrics..."):
                scores = evaluation.compute_all(preds, refs, images=[load_rgb(uploaded)], include_cider=False)
            st.table({k: [round(v, 4)] for k, v in scores.items()})
            st.caption("CIDEr needs a batch of images - use scripts/evaluate.py for the full benchmark.")

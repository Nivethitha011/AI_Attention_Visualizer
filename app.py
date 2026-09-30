import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text
from embedding import create_embeddings
from attention import calculate_attention


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="AI Attention Visualizer",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("AI Attention Visualizer")

st.write(
    "Upload a study-note image to extract text "
    "using Tesseract OCR and visualize word-level "
    "attention scores."
)


# -----------------------------------------
# IMAGE UPLOAD
# -----------------------------------------

file = st.file_uploader(
    "Upload a study-note image",
    type=["jpg", "jpeg", "png"]
)


if file is not None:

    # -------------------------------------
    # OPEN IMAGE
    # -------------------------------------

    image = Image.open(file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    # -------------------------------------
    # OCR
    # -------------------------------------

    st.subheader("Extracted Text")

    with st.spinner("Extracting text using Tesseract OCR..."):

        text = extract_text(image)


    if not text.strip():

        st.error(
            "No text found in the image."
        )

        st.stop()


    st.text_area(
        "OCR Result",
        text,
        height=200
    )


    # -------------------------------------
    # EXTRACT WORDS
    # -------------------------------------

    words = text.split()

    words = [
        word.strip(
            ".,!?;:()[]{}"
        )
        for word in words
    ]


    # Remove empty words
    words = [
        word
        for word in words
        if word
    ]


    # Remove very short words
    words = [
        word
        for word in words
        if len(word) > 2
    ]


    # Maximum 20 words
    words = words[:20]


    if not words:

        st.error(
            "No suitable words found."
        )

        st.stop()


    # -------------------------------------
    # DISPLAY WORDS
    # -------------------------------------

    st.subheader("Words Selected for Analysis")

    st.write(
        ", ".join(words)
    )


    # -------------------------------------
    # CREATE EMBEDDINGS
    # -------------------------------------

    with st.spinner(
        "Generating word embeddings..."
    ):

        embeddings = create_embeddings(
            words
        )


    st.success(
        "Word embeddings generated successfully."
    )


    # -------------------------------------
    # EMBEDDING INFORMATION
    # -------------------------------------

    st.subheader("Embedding Information")

    st.write(
        f"Number of words: **{len(words)}**"
    )

    st.write(
        f"Embedding dimension: **{embeddings.shape[1]}**"
    )


    # -------------------------------------
    # ATTENTION
    # -------------------------------------

    with st.spinner(
        "Calculating attention scores..."
    ):

        scores = calculate_attention(
            embeddings
        )


    # -------------------------------------
    # ATTENTION VISUALIZATION
    # -------------------------------------

    st.subheader("Word Attention")

    # Normalize scores
    if scores.max() > 0:

        display_scores = (
            scores / scores.max()
        )

    else:

        display_scores = scores


    # Display each word
    for word, score in zip(
        words,
        display_scores
    ):

        st.write(
            f"**{word}**"
        )

        st.progress(
            float(score)
        )

        st.caption(
            f"Attention Score: {score:.4f}"
        )


    # -------------------------------------
    # TOP ATTENTION WORD
    # -------------------------------------

    top_index = np.argmax(scores)

    top_word = words[top_index]

    top_score = scores[top_index]


    st.divider()

    st.subheader(
        "Highest Attention Word"
    )

    st.success(
        f"Word: **{top_word}**"
    )

    st.write(
        f"Calculated Attention Score: "
        f"**{top_score:.4f}**"
    )


    # -------------------------------------
    # ATTENTION TABLE
    # -------------------------------------

    st.subheader(
        "Attention Score Summary"
    )

    for word, score in zip(
        words,
        scores
    ):

        st.write(
            f"{word} : {score:.6f}"
        )


else:

    st.info(
        "Upload an image to start the AI Attention analysis."
    )

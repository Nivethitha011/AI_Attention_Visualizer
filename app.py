import streamlit as st
from PIL import Image

from ocr import load_ocr, extract_text
from embedding import load_model, create_embeddings
from attention import (
    get_attention_matrix,
    attention_dataframe
)


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI OCR Attention Visualizer",
    page_icon="🤖",
    layout="wide"
)


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🤖 AI OCR Attention Visualizer")

st.write(
    "Upload an image → OCR extracts the text → "
    "Transformer analyzes the text → "
    "Attention is visualized."
)


# ------------------------------------------------
# LOAD OCR
# ------------------------------------------------

@st.cache_resource
def get_ocr():

    return load_ocr()


# ------------------------------------------------
# LOAD TRANSFORMER
# ------------------------------------------------

@st.cache_resource
def get_transformer():

    return load_model()


# ------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload an image containing text",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    # --------------------------------------------
    # READ IMAGE
    # --------------------------------------------

    image = Image.open(
        uploaded_file
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------
    # OCR
    # --------------------------------------------

    with st.spinner(
        "🔎 Extracting text using OCR..."
    ):

        reader = get_ocr()

        extracted_text, results = extract_text(
            reader,
            image
        )


    with col2:

        st.subheader(
            "📝 OCR Result"
        )

        if extracted_text.strip():

            st.text_area(
                "Extracted Text",
                extracted_text,
                height=200
            )

            st.success(
                "OCR completed successfully!"
            )

        else:

            st.error(
                "No text detected in the image."
            )


    # --------------------------------------------
    # OCR DETAILS
    # --------------------------------------------

    if results:

        st.subheader(
            "🔍 OCR Detection Details"
        )

        for i, result in enumerate(results):

            text = result[1]
            confidence = result[2]

            st.write(
                f"**{i + 1}. {text}**  "
                f"Confidence: {confidence:.2%}"
            )


    # --------------------------------------------
    # TRANSFORMER ATTENTION
    # --------------------------------------------

    if extracted_text.strip():

        st.divider()

        st.header(
            "🧠 Transformer Attention"
        )

        with st.spinner(
            "🤖 Calculating attention..."
        ):

            tokenizer, model = get_transformer()

            inputs, outputs = create_embeddings(
                extracted_text,
                tokenizer,
                model
            )

            tokens, attention = get_attention_matrix(
                inputs,
                outputs,
                tokenizer
            )


        st.success(
            "Attention calculated successfully!"
        )


        # ----------------------------------------
        # ATTENTION MATRIX
        # ----------------------------------------

        st.subheader(
            "🔥 Attention Heatmap"
        )

        attention_df = attention_dataframe(
            tokens,
            attention
        )

        st.dataframe(
            attention_df.style
            .background_gradient(
                cmap="Blues"
            )
            .format("{:.3f}"),
            use_container_width=True
        )


        # ----------------------------------------
        # SELECT TOKEN
        # ----------------------------------------

        st.subheader(
            "🔍 Explore Token Attention"
        )

        selected_token = st.selectbox(
            "Select a token",
            tokens
        )

        selected_index = tokens.index(
            selected_token
        )


        # ----------------------------------------
        # ATTENTION SCORES
        # ----------------------------------------

        scores = attention[
            selected_index
        ].detach().numpy()


        score_data = []

        for token, score in zip(
            tokens,
            scores
        ):

            score_data.append(
                {
                    "Token": token,
                    "Attention": float(score)
                }
            )


        score_data.sort(
            key=lambda x: x["Attention"],
            reverse=True
        )


        # ----------------------------------------
        # BAR CHART
        # ----------------------------------------

        st.subheader(
            f'📊 Attention from "{selected_token}"'
        )

        chart_data = {
            item["Token"]:
            item["Attention"]
            for item in score_data
        }

        st.bar_chart(
            chart_data
        )


        # ----------------------------------------
        # TOP TOKENS
        # ----------------------------------------

        st.subheader(
            "⭐ Highest Attention Tokens"
        )

        for item in score_data[:10]:

            st.write(
                f"**{item['Token']}** → "
                f"{item['Attention']:.4f}"
            )


else:

    st.info(
        "👆 Upload an image to start OCR."
    )

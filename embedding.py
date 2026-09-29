import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "distilbert-base-uncased"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_attentions=True
    )

    model.eval()

    return tokenizer, model


def create_embeddings(
    text,
    tokenizer,
    model
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    return inputs, outputs

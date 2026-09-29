import torch
import pandas as pd


def get_attention_matrix(
    inputs,
    outputs,
    tokenizer
):

    # Last Transformer layer
    last_layer_attention = outputs.attentions[-1]

    # Average all attention heads
    attention = last_layer_attention.mean(
        dim=1
    )[0]

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    # Remove special tokens
    valid_indices = [
        i
        for i, token in enumerate(tokens)
        if token not in [
            "[CLS]",
            "[SEP]",
            "[PAD]"
        ]
    ]

    tokens = [
        tokens[i]
        for i in valid_indices
    ]

    attention = attention[
        valid_indices
    ][:, valid_indices]

    return tokens, attention


def attention_dataframe(
    tokens,
    attention
):

    values = attention.detach().numpy()

    return pd.DataFrame(
        values,
        index=tokens,
        columns=tokens
    )

from oai.utils.wrappers import retry_openai_api, metered
import numpy as np
from typing import Any, Union
import openai

Embedding = Union[list[np.float32], np.ndarray[Any, np.dtype[np.float32]]]
"""Embedding vector"""
TText = list[int]
"""Token array representing text"""


@metered
@retry_openai_api()
def get_embedding(
    embed: Union[str, TText, list[str], list[TText]],
    openai_api_key: Union[str, None] = None,
    model: str = "text-embedding-ada-002",
) -> Union[Embedding, list[Embedding]]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.

    Returns:
        List[float]: The embedding.
    """
    multiple = isinstance(embed, list) and all(not isinstance(i, int) for i in input)

    if isinstance(embed, str):
        embed = embed.replace("\n", " ")
    elif multiple and isinstance(embed[0], str):
        embed = [text.replace("\n", " ") for text in embed]

    embeddings = openai.Embedding.create(
        input=embed,
        api_key=openai_api_key,
        model=model,
    ).data

    if not multiple:
        return embeddings[0]["embedding"]

    embeddings = sorted(embeddings, key=lambda x: x["index"])
    return [d["embedding"] for d in embeddings]

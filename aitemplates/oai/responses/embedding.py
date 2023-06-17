import numpy as np
from typing import Any, Union
import os
from dotenv import load_dotenv

import openai

from aitemplates.oai.utils.wrappers import retry_openai_api
from aitemplates.oai.ApiManager import ApiManager

dotenv_path = os.path.join(os.getcwd(), '.env')  # get the path to .env file in current working directory
load_dotenv(dotenv_path)  # load environment variables from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY

Embedding = Union[list[np.float32], np.ndarray[Any, np.dtype[np.float32]]]
"""Embedding vector"""
TText = list[int]
"""Token array representing text"""

@retry_openai_api()
def get_embedding(
    embed: Union[str, TText, list[str], list[TText]],
    model: str = "text-embedding-ada-002",
) -> Union[Embedding, list[Embedding]]:
    """Get an embedding from the ada model.

    Args:
        embed: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.
        openai_api_key: OpenAI API key.
        model: The OpenAI embedding model to use. Defaults to "text-embedding-ada-002".

    Returns:
        List[float]: The embedding.
    """
    api_manager = ApiManager()
    multiple = isinstance(embed, list) and all(not isinstance(i, int) for i in input)

    # clean the input string
    if isinstance(embed, str):
        embed = embed.replace("\n", " ")
    elif multiple and isinstance(embed[0], str):
        embed = [text.replace("\n", " ") for text in embed]

    embeddings = openai.Embedding.create(
        input=embed,
        model=model,
    )
    
    api_manager.update_cost(
        embeddings.usage.prompt_tokens,
        0,
        embeddings.model
    )

    if not multiple:
        return embeddings.data[0]["embedding"]

    # sort the multiple return embeddings in their correct order
    embeddings = sorted(embeddings.data, key=lambda x: x["index"])
    return [d["embedding"] for d in embeddings.data]

from typing import Optional, Dict
import os
from dotenv import load_dotenv

import openai

from aitemplates.oai.utils.wrappers import retry_openai_api, metered
from aitemplates.oai.types.base import ChatSequence

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@metered
@retry_openai_api()
def create_chat_completion(
    messages: ChatSequence,
    model: str = "gpt-3.5-turbo",
    temperature: Optional[float] = 0,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (list[MessageDict]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The temperature to use. Defaults to 0.
        max_tokens (int, optional): The maximum tokens to use. Defaults to None.
        top_p (float, optional): The nucleus sampling probability. Defaults to 1.
        n (int, optional): The number of messages to generate. Defaults to 1.
        stop (str, optional): The sequence at which the generation will stop. Defaults to None.
        presence_penalty (float, optional): The presence penalty to use. Defaults to 0.
        frequency_penalty (float, optional): The frequency penalty to use. Defaults to 0.

    Returns:
        Any: The response from the chat completion.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages.raw(),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    if n > 1:
        return response.choices
    return response.choices[0].message["content"]

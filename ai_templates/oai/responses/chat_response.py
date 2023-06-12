from typing import Optional
import os
from dotenv import load_dotenv

import openai

from ai_templates.oai.utils.wrappers import retry_openai_api, metered
from ai_templates.oai.types.base import ChatSequence

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@metered
@retry_openai_api()
def create_chat_completion(
    prompt: ChatSequence,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt.raw(),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    resp = response.choices[0].message["content"]
    return resp

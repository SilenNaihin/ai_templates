from ai_templates.oai.utils.wrappers import retry_openai_api, metered
import asyncio
import os
from dotenv import load_dotenv

from typing import Any, Optional, List, Union

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from ai_templates.oai.types.base import MessageDict, ChatMessages
from ai_templates.oai.ApiManager import ApiManager

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@retry_openai_api()
async def _throttled_openai_chat_completion_acreate(
    messages: list[MessageDict],
    limiter: aiolimiter.AsyncLimiter,
    temperature: Optional[float] = 0,
    model: Optional[str] = "gpt-3.5-turbo",
    max_tokens: Optional[int] = None,
) -> Any:
    async with limiter:
        return await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


async def generate_from_openai_chat_completion(
    parallel_calls: ChatMessages,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: Union[int, None] = None,
    return_every: int = False,
    requests_per_minute: int = 300,
    response_list: Optional[List[str]] = None,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        parallel_calls: List of full prompts to generate from.
        model: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    api_manager = ApiManager()
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=chat_sequence.raw(),
            temperature=temperature,
            max_tokens=max_tokens,
            limiter=limiter,
        )
        for chat_sequence in parallel_calls.chat_sequences
    ]

    responses = []

    # return_every wont preserve order
    for future in tqdm_asyncio.as_completed(async_responses):
        response = await future

        api_manager.update_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.model,
        )

        response_text = response.choices[0].message.content
        responses.append(response_text)
        if return_every:
            print(response_text + "\n")

        # Check if a list was passed and add the response_text to it
        if response_list is not None:
            response_list.append(response_text)

    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return responses

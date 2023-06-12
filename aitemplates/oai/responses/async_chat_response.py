from aitemplates.oai.utils.wrappers import retry_openai_api, metered
import logging
import os
from dotenv import load_dotenv

from typing import Any, Optional, List, Union

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from aitemplates.oai.types.base import MessageDict, ChatMessages
from aitemplates.oai.ApiManager import ApiManager

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@retry_openai_api()
async def _throttled_acreate_chat_completion(
    messages: list[MessageDict],
    limiter: aiolimiter.AsyncLimiter,
    temperature: Optional[float] = 0,
    model: Optional[str] = "gpt-3.5-turbo",
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
) -> Any:
    """Create a throttled chat completion using the OpenAI API.

    This function ensures the number of requests made to the OpenAI API does not exceed a specified limit.

    Args:
        messages (list[MessageDict]): The messages to send to the chat completion.
        limiter (aiolimiter.AsyncLimiter): The limiter that manages throttling of requests.
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
    async with limiter:
        return await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )


async def async_create_chat_completion(
    parallel_calls: ChatMessages,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: Union[int, None] = None,
    print_every: int = False,
    keep_order: bool = False,
    requests_per_minute: int = 300,
    response_list: Optional[List[str]] = None,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
) -> list[str]:
    """Generate from OpenAI Chat Completion API asynchronously.

    Args:
        parallel_calls (ChatMessages): List of full prompts to generate from.
        model (str, optional): Model configuration. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Temperature to use. Defaults to 0.
        max_tokens (Union[int, None], optional): Maximum number of tokens to generate. Defaults to None.
        print_every (int, optional): Print the response after every this many calls. It does nothing if keep_order is True. Defaults to False.
        keep_order (bool, optional): If True, keeps the order of responses same as the input prompts. Defaults to False.
        requests_per_minute (int, optional): Number of requests per minute to allow. Defaults to 300.
        response_list (List[str], optional): If provided, responses are added to this list.
        top_p (float, optional): The nucleus sampling probability. Defaults to 1.
        n (int, optional): The number of messages to generate. Defaults to 1.
        stop (str, optional): The sequence at which the generation will stop. Defaults to None.
        presence_penalty (float, optional): The presence penalty to use. Defaults to 0.
        frequency_penalty (float, optional): The frequency penalty to use. Defaults to 0.

    Returns:
        list[str]: List of generated responses.
    """
    if keep_order and print_every:
        logging.warning("print_every do nothing since keep_order is True")
    api_manager = ApiManager()
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_acreate_chat_completion(
            model=model,
            messages=chat_sequence.raw(),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            limiter=limiter,
        )
        for chat_sequence in parallel_calls.chat_sequences
    ]

    responses = []
    prompt_tokens = 0
    completion_tokens = 0

    if keep_order:
        # Keep the order of responses same as input prompts
        responses = await tqdm_asyncio.gather(*async_responses)

        # Update the cost associated with this response in the API manager
        for response in responses:
            responses.append(response.choices[0].message.content)
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

        api_manager.update_cost(
            prompt_tokens,
            completion_tokens,
            responses[0].model,
        )
    else:
        # Without order
        for future in tqdm_asyncio.as_completed(async_responses):
            response = await future

            # Update the cost associated with this response in the API manager
            api_manager.update_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.model,
            )

            response_text = response.choices[0].message.content
            responses.append(response_text)
            if print_every:
                logging.info(response_text + "\n")

            # Check if a list was passed and add the response_text to it
            if response_list is not None:
                response_list.append(response_text)

    # Close the session
    await openai.aiosession.get().close()  # type: ignore
    return responses

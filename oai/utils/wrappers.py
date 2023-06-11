from __future__ import annotations

from unittest.mock import patch
import functools
import time

from openai.error import APIError, RateLimitError
import openai
import openai.api_resources.abstract.engine_api_resource as engine_api_resource
import openai.util
from openai.openai_object import OpenAIObject

from oai.ApiManager import ApiManager
from oai.types.base import ChatSequence, Message

from oai.responses.chat_response import create_chat_completion


def metered(func):
    """Adds ApiManager metering to functions which make OpenAI API calls"""
    api_manager = ApiManager()

    openai_obj_processor = openai.util.convert_to_openai_object

    def update_usage_with_response(response: OpenAIObject):
        try:
            usage = response.usage
            print(f"Reported usage from call to model {response.model}: {usage}")
            api_manager.update_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens if "completion_tokens" in usage else 0,
                response.model,
            )
        except Exception as err:
            print(f"Failed to update API costs: {err.__class__.__name__}: {err}")

    def metering_wrapper(*args, **kwargs):
        openai_obj = openai_obj_processor(*args, **kwargs)
        if isinstance(openai_obj, OpenAIObject) and "usage" in openai_obj:
            update_usage_with_response(openai_obj)
        return openai_obj

    def metered_func(*args, **kwargs):
        with patch.object(
            engine_api_resource.util,
            "convert_to_openai_object",
            side_effect=metering_wrapper,
        ):
            return func(*args, **kwargs)

    return metered_func


def retry_openai_api(
    num_retries: int = 10,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            user_warned = not warn_user
            num_attempts = num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise

                    print("Error: Reached rate limit, passing...")
                    if not user_warned:
                        print(
                            """Please double check that you have setup a paid OpenAI API 
                              Account. You can read more here: https://docs.agpt.co/setup/#getting-an-api-key"""
                        )
                        user_warned = True

                except APIError as e:
                    if (e.http_status not in [502, 429]) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                print(f"Error: API Bad gateway. Waiting {backoff} seconds...")
                time.sleep(backoff)

        return _wrapped

    return _wrapper


def call_ai_function(
    function: str,
    args: list,
    description: str,
    model: str = "gpt-3.5-turbo",
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    arg_str: str = ", ".join(args)

    prompt = ChatSequence.for_model(
        model,
        [
            Message(
                "system",
                f"You are now the following python function: ```# {description}"
                f"\n{function}```\n\nOnly respond with your `return` value.",
            ),
            Message("user", arg_str),
        ],
    )
    return create_chat_completion(prompt=prompt, temperature=0)

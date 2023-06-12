from __future__ import annotations

from unittest.mock import patch
import functools
import time
import os

from openai.error import (
    APIError,
    RateLimitError,
    InvalidRequestError,
    APIConnectionError,
    Timeout,
)

import openai
import openai.api_resources.abstract.engine_api_resource as engine_api_resource
import openai.util
from openai.openai_object import OpenAIObject

from ai_templates.oai.ApiManager import ApiManager


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

    @functools.wraps(func)
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
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set when using OpenAI API."
                )
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
                except InvalidRequestError:
                    print("OpenAI API Invalid Request: Prompt was filtered")
                    user_warned = True
                except APIConnectionError:
                    print(
                        "OpenAI API Connection Error: Error Communicating with OpenAI"
                    )
                    user_warned = True
                except Timeout:
                    print("OpenAI APITimeout Error: OpenAI Timeout")
                    user_warned = True
                except APIError as e:
                    if (e.http_status not in [502, 429]) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                print(f"Error: API Bad gateway. Waiting {backoff} seconds...")
                time.sleep(backoff)

        return _wrapped

    return _wrapper

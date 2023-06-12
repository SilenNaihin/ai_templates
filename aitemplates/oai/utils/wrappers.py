from __future__ import annotations

from unittest.mock import patch
import functools
import time
import os
import logging


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

from aitemplates.oai.ApiManager import ApiManager


def metered(func):
    """Adds ApiManager metering to functions which make OpenAI API calls.

    Args:
        func (Callable): The function to meter.

    Returns:
        Callable: The metered function.
    """
    api_manager = ApiManager()

    openai_obj_processor = openai.util.convert_to_openai_object

    def update_usage_with_response(response: OpenAIObject):
        """Updates the usage statistics based on the response."""
        try:
            usage = response.usage
            api_manager.update_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens if "completion_tokens" in usage else 0,
                response.model,
            )
        except Exception as err:
            logging.error(
                f"Failed to update API costs: {err.__class__.__name__}: {err}"
            )

    def metering_wrapper(*args, **kwargs):
        """Wraps the function in a metering logic."""
        openai_obj = openai_obj_processor(*args, **kwargs)
        if isinstance(openai_obj, OpenAIObject) and "usage" in openai_obj:
            update_usage_with_response(openai_obj)
        return openai_obj

    @functools.wraps(func)
    def metered_func(*args, **kwargs):
        """A function that runs the original function with metering applied."""
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
    """Decorate a function to retry OpenAI API call when it fails.

    This function uses exponential backoff strategy for retries.

    Args:
        num_retries (int, optional): Number of retries. Defaults to 10.
        backoff_base (float, optional): Base for exponential backoff. Defaults to 2.0.
        warn_user (bool, optional): Whether to warn the user. Defaults to True.

    Returns:
        Callable: The decorated function.
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

                    logging.error("Error: Reached rate limit, passing...")
                    if not user_warned:
                        logging.warning(
                            """Please double check that you have setup a paid OpenAI API 
                              Account. You can read more here: https://docs.agpt.co/setup/#getting-an-api-key"""
                        )
                        user_warned = True
                except InvalidRequestError:
                    logging.error("OpenAI API Invalid Request: Prompt was filtered")
                    user_warned = True
                except APIConnectionError:
                    logging.error(
                        "OpenAI API Connection Error: Error Communicating with OpenAI"
                    )
                    user_warned = True
                except Timeout:
                    logging.error("OpenAI APITimeout Error: OpenAI Timeout")
                    user_warned = True
                except APIError as e:
                    if (e.http_status not in [502, 429]) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logging.error(f"Error: API Bad gateway. Waiting {backoff} seconds...")
                time.sleep(backoff)

        return _wrapped

    return _wrapper

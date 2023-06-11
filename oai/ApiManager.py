from __future__ import annotations
import os

from typing import List, Optional, Literal

import openai
from openai import Model

from oai.types.base import MessageDict
from oai.types.models import OPEN_AI_MODELS
from oai.types.Singleton import Singleton

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[Model]] = None

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None

    def create_chat_completion(
        self,
        messages: list[MessageDict],
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
        stream: bool = False,
        api_key: Optional[str] = None,
    ) -> str:
        self.check_model(model)

        response: = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            api_key=api_key
        )

        if stream:
            # create variables to collect the stream of chunks
            collected_chunks = []
            collected_messages = []
            # iterate through the stream of events
            for chunk in response:
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk["choices"][0]["delta"]  # extract the message
                collected_messages.append(chunk_message)  # save the message

            full_reply_content = "".join(
                [m.get("content", "") for m in collected_messages]
            )
            print(f"Full conversation received: {full_reply_content}")
            return full_reply_content

        if not hasattr(response, "error"):
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            self.update_cost(prompt_tokens, completion_tokens, model)

        return response

    # def create_streaming_completion(

    # ):

    # def creat_async_completion(

    # ):

    def update_cost(self, prompt_tokens, completion_tokens, model: str):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # the .model property in API responses can contain version suffixes like -v2
        model = model[:-3] if model.endswith("-v2") else model

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (
            prompt_tokens * OPEN_AI_MODELS[model].prompt_token_cost
            + completion_tokens * OPEN_AI_MODELS[model].completion_token_cost
        ) / 1000
        print(f"Total running cost: ${self.total_cost:.3f}")

    def check_model(
        self, model: str, model_type: Literal["smart_llm_model", "fast_llm_model"]
    ) -> str:
        """Check if model specified is available for use. If not, return gpt-3.5-turbo."""
        api_manager = ApiManager()
        models = api_manager.get_models()

        if any(model in m["id"] for m in models):
            return model

        print("You do not have access to {model}.")
        return "gpt-3.5-turbo"

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget

    def get_models(self) -> List[Model]:
        """
        Get list of available GPT models.

        Returns:
        list: List of available GPT models.

        """
        if self.models is None:
            all_models = openai.Model.list()["data"]
            self.models = [model for model in all_models if "gpt" in model["id"]]

        return self.models

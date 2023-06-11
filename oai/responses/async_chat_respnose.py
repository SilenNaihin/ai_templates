from oai.utils.wrappers import retry_openai_api, metered
from oai.types.base import ChatSequence
from oai.ApiManager import ApiManager
from typing import Optional

@metered
@retry_openai_api()
def create_async_chat_completion(
    prompt: ChatSequence,
    model: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    api_key: Optional[str] = None
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
    if model is None:
        model = prompt.model.name

    api_manager = ApiManager()
    response = None

    response = api_manager.create_chat_completion(
        model=model,
        messages=prompt.raw(),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream
        api_key=api_key
    )

    resp = response.choices[0].message["content"]
    return resp

from typing import Optional, Any, Union
import os
from dotenv import load_dotenv

import openai
from aitemplates.oai.ApiManager import ApiManager

from aitemplates.oai.utils.wrappers import retry_openai_api
from aitemplates.oai.types.chat import ChatSequence, FunctionsAvailable, ChatConversation, Message

dotenv_path = os.path.join(os.getcwd(), '.env')  # get the path to .env file in current working directory
load_dotenv(dotenv_path)  # load environment variables from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@retry_openai_api()
def create_chat_completion(
    messages: Union[ChatSequence, ChatConversation, Message],
    model: str = "gpt-3.5-turbo-0613",
    temperature: Optional[float] = 0,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
    functions: Optional[FunctionsAvailable] = None,
    function_call: Optional[object] = None,
    send_object: bool = False,
) -> Any:
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
        functions (Optional[FunctionsAvailable], optional): The functions to use. Defaults to None.
        send_object (bool, optional): Whether to return the response object. Defaults to False.

    Returns:
        Any: The response from the chat completion.
    """
    kwarg_messages = None
    
    # we set it to the last sequence which the response is None
    if isinstance(messages, ChatConversation):
        kwarg_messages = messages.conversation_history[-1].prompt.raw()
    elif isinstance(messages, Message):
        messages=ChatSequence([messages])
    else: 
        kwarg_messages = messages.raw()
        
    api_manager = ApiManager()
    kwargs = {
        "model": model,
        "messages": kwarg_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    
    function_pairs = None
    
    # if it's a ChatSequence being passed in
    if messages.function_pairs:
        kwargs["functions"] = messages.function_pairs.get_function_defs()
        function_pairs = messages.function_pairs
    elif functions: 
        function_pairs = functions
        # if you're passing in global functions from a ChatConversation sequential call
        if "functions" in kwargs:
            # add functions not in the existing functions array 
            existing_function_names = {func["name"] for func in kwargs["functions"]}
            new_functions = [func for func in functions.get_function_defs() if func["name"] not in existing_function_names]
            kwargs["functions"].extend(new_functions)
        else:
            kwargs["functions"] = functions.get_function_defs()
    
    if function_call:
        kwargs["function_call"] = function_call
    
    response = openai.ChatCompletion.create(
        **kwargs
    )
    
    function_result = None
    
    if response.choices[0].message.get("function_call"):
        function_result = FunctionsAvailable.execute_function_call(response.choices[0].message.function_call, function_pairs)
    
    api_manager.update_cost(
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.model
    )
    
    if isinstance(messages, ChatConversation):
        if function_result:
            # we want to set the content to the function_result
            messages.conversation_history[-1].update_response(response, function_result)
        else:
            messages.conversation_history[-1].update_response(response)
    
    if send_object:
        return response, function_result if function_result else response
    elif n and n > 1:
        return response.choices
    elif function_result:
        return function_result
    else:
        return response.choices[0].message["content"]

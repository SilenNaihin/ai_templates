from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Literal, TypedDict, Optional, Union, Any

# Embedding vector
Embedding = Union[list[np.float32], np.ndarray[Any, np.dtype[np.float32]]]

# Token array representing text
TText = list[int]

# Different roles the chat API takes
MessageRole = Literal["system", "user", "assistant", "function"]


class MessageDict(TypedDict):
    role: MessageRole
    content: str

@dataclass
class ResponseDict:
    index: int
    message: Message
    finish_reason: str
    model: str
    created: str
    
    def __init__(self, response: dict, function_response: Any = None):
        choice = response["choices"][0]  # Getting the first choice
        self.index = choice["index"]
        if function_response:
            self.message = Message("function", function_response)
        else:
            self.message = Message(**choice["message"])
        self.finish_reason = choice["finish_reason"]
        self.model = response["model"]
        self.created = response["created"]
        
    def __getitem__(self, i: int):
        return self.message
    
    @staticmethod
    def convert_to_response_dict(response: Union[dict, tuple]) -> Union['ResponseDict', tuple[ResponseDict, Any]]:
        if response is not None and isinstance(response, tuple):
            # if it's a tuple that means a function was called
            return (ResponseDict(response[0]), response[1])
        else: 
            return ResponseDict(response)
    
    @staticmethod
    def convert_to_raw(response_dict: ResponseDict) -> MessageDict| tuple[MessageDict, Any]:
        if response_dict is not None and isinstance(response_dict, tuple):
            # if it's a tuple that means a function was called
            return (response_dict.message.raw(), response_dict[1])
        else: 
            return response_dict.message.raw()
        
        


class UsageDict(TypedDict):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: int


@dataclass
class FunctionCall:
    name: str
    arguments: str

@dataclass
class Message:
    """OpenAI Message object containing a role and the message content"""

    role: MessageRole
    content: str
    function_call: Optional[FunctionCall] = None

    def raw(self) -> MessageDict:
        return {"role": self.role, "content": self.content}


@dataclass
class ModelInfo:
    """Struct for model information."""

    name: str
    prompt_token_cost: float
    completion_token_cost: float
    max_tokens: int


@dataclass
class ChatModelInfo(ModelInfo):
    "Struct for chat model information."


@dataclass
class EmbeddingModelInfo(ModelInfo):
    "Struct for embedding model information."

    embedding_dimensions: int
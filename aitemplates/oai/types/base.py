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
    
    def __init__(self, response: dict):
        choice = response["choices"][0]  # Getting the first choice
        self.index = choice["index"]
        self.message = Message(**choice["message"])
        self.finish_reason = choice["finish_reason"]
        self.model = response["model"]
        self.created = response["created"]
        


class UsageDict(TypedDict):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: int


@dataclass
class Message:
    """OpenAI Message object containing a role and the message content"""

    role: MessageRole
    content: str
    function_call: Optional[object] = None

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
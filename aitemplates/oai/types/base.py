from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from math import ceil, floor
from typing import Literal, TypedDict, Optional, Union, Any

# Embedding vector
Embedding = Union[list[np.float32], np.ndarray[Any, np.dtype[np.float32]]]

# Token array representing text
TText = list[int]

# Different roles the chat API takes
MessageRole = Literal["system", "user", "assistant"]


class MessageDict(TypedDict):
    role: MessageRole
    content: str


class ResponseDict(TypedDict):
    index: int
    message: MessageDict
    finish_reason: str


class UsageDict(TypedDict):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: int


@dataclass
class Message:
    """OpenAI Message object containing a role and the message content"""

    role: MessageRole
    content: str

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


@dataclass
class ChatSequence:
    "Utility container for a chat sequence"

    messages: list[Message] = field(default_factory=list)

    def __getitem__(self, i: int):
        return self.messages[i]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __add__(self, other):
        if isinstance(other, ChatSequence):
            # Concatenate the messages lists
            return ChatSequence(self.messages + other.messages)
        else:
            raise TypeError(f"Unsupported operand type for +: {type(other)}")

    def append(self, message: Message):
        return self.messages.append(message)

    def extend(self, messages: list[Message] | ChatSequence):
        return self.messages.extend(messages)

    def insert(self, index: int, *messages: Message):
        for message in reversed(messages):
            self.messages.insert(index, message)

    def add(self, message_role: MessageRole, content: str):
        self.messages.append(Message(message_role, content))

    @property
    def token_length(self):
        from aitemplates.oai.utils.count_tokens import num_tokens_from_messages

        return num_tokens_from_messages(self.messages, self.model.name)

    def raw(self) -> list[MessageDict]:
        return [m.raw() for m in self.messages]

    def expand(self) -> list[Message]:
        return [m for m in self.messages]

    def dump(self) -> str:
        "Return all information stored in the dataclass as a string"
        SEPARATOR_LENGTH = 42

        def separator(text: str):
            half_sep_len = (SEPARATOR_LENGTH - 2 - len(text)) / 2
            return f"{floor(half_sep_len)*'-'} {text.upper()} {ceil(half_sep_len)*'-'}"

        formatted_messages = "\n".join(
            [f"{separator(m.role)}\n{m.content}" for m in self.messages]
        )
        return f"""
        ============== ChatSequence ==============
        Length: {self.token_length} tokens; {len(self.messages)} messages
        {formatted_messages}
        ==========================================
        """

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ChatSequence:
        "Create a chat messages object from a dictionary."
        return ChatSequence(
            [Message(role=x["role"], content=x["content"]) for x in data["messages"]]
        )

    def to_dict(self) -> dict[str, Any]:
        "Convert a chat messages object to a dictionary."
        return {
            "messages": [
                {
                    "role": x.role,
                    "content": x.content,
                }
                for x in self.messages
            ]
        }


@dataclass(frozen=True)
class ChatMessages:
    """A set of prompts to ask.

    Attributes:
        chat_sequences: A set of message sequences, each represents a prompt
    """

    chat_sequences: list[ChatSequence]
    sequential: bool = False

    def __getitem__(self, i: int):
        return self.chat_sequences[i]

    def __iter__(self):
        return iter(self.chat_sequences)

    def __len__(self):
        return len(self.chat_sequences)

    def append(self, message: ChatSequence):
        return self.chat_sequences.append(message)

    def extend(self, messages: list[ChatSequence] | ChatMessages):
        return self.chat_sequences.extend(messages)

from __future__ import annotations

from termcolor import colored

from dataclasses import dataclass, field
from math import ceil, floor
from typing import List, Tuple, Optional, Union, Any

from aitemplates.oai.types.functions import FunctionsAvailable, FunctionPair
from aitemplates.oai.types.base import Message, MessageRole, MessageDict, ResponseDict

class ChatSequence:
    "Utility container for a chat sequence"

    messages: list[Message]
    function_pairs: FunctionsAvailable
    
    def __init__(self, messages_list: list[Message], function_pairs: Optional[Union[FunctionsAvailable, list[FunctionPair], FunctionPair]] = None):
       self.messages = messages_list
       
       if function_pairs:
           self.function_pairs.set_function_pairs(function_pairs)
       else:
           self.function_pairs = FunctionsAvailable([])

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

    def add_message(self, message_role: MessageRole, content: str):
        self.messages.append(Message(message_role, content))

    def token_length(self, model: str):
        from aitemplates.oai.utils.count_tokens import num_tokens_from_messages

        return num_tokens_from_messages(self.messages, model)

    def raw(self) -> list[MessageDict]:
        return [m.raw() for m in self.messages]
    
    def prompt_string(self) -> str:
        return ''.join([m.raw().get("content") for m in self.messages])

    def expand(self) -> list[Message]:
        return [m for m in self.messages]

    def dump(self, model: str = 'gpt-3.5-turbo') -> str:
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
        Length: {self.token_length(model)} tokens; {len(self.messages)} messages
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

class ChatPair:
    chat_pair: Tuple[ChatSequence, Optional[ResponseDict]]
    
    def __init__(self, chat_sequence: ChatSequence, response: Optional[ResponseDict]) -> None:
        self.chat_pair = (chat_sequence, response)
    
    def __getitem__(self, i: int) -> Union[ChatSequence, Optional[ResponseDict]]:
        return self.chat_pair[i]
    
    @property
    def prompt(self) -> ChatSequence:
        return self.chat_pair[0]
    
    @property
    def prompt_raw(self) -> list[MessageDict]:
        return self.chat_pair[0].raw()

    @property
    def response(self) -> Optional[ResponseDict]:
        return self.chat_pair[1]
    
    def update_prompt(self, new_prompt: ChatSequence) -> 'ChatPair':
        self.chat_pair = (new_prompt, self.chat_pair[1])
        return self

    def update_response(self, new_response: ResponseDict) -> 'ChatPair':
        self.chat_pair = (self.chat_pair[0], new_response)
        return self
    

class ChatConversation:
    """A set of prompts to either ask the chat API in parallel or store the conversation history.

    Attributes:
        conversation_history: A list of tuples that represent the history of the conversation
        function_pairs: The available functions
    """
    
    conversation_history: List[ChatPair]
    # functions available to all prompts
    function_pairs: FunctionsAvailable
            
    def __init__(self, first_prompt: Union[ChatSequence, List[ChatSequence], List[ChatPair]], function_pairs: Optional[Union[FunctionsAvailable, list[FunctionPair], FunctionPair]] = None):
        if isinstance(first_prompt, list):
            if isinstance(first_prompt[0], ChatSequence): # case if list[ChatSequence]
                self.conversation_history = [ChatPair(chat_sequence, None) for chat_sequence in first_prompt]
            else:
                self.conversation_history = first_prompt
        else:
            self.conversation_history = [ChatPair(first_prompt, None)] # case of just a single ChatSequence

        if function_pairs:
            self.function_pairs.set_function_pairs(function_pairs)
        else:
            self.function_pairs = FunctionsAvailable([])
    
    def __getitem__(self, i: int) -> ChatPair:
        """Allows indexing into the conversation history.

        Args:
            i (int): The index to access.

        Returns:
            Tuple[ChatSequence, Optional[ResponseDict]]: The message-response pair at the given index.
        """
        return self.conversation_history[i]

    # ... the rest of your code

    def __add__(self, other: 'ChatConversation') -> 'ChatConversation':
        """Defines addition for ChatConversation instances as the concatenation of their conversation histories.

        Args:
            other (ChatConversation): Another ChatConversation instance.

        Returns:
            ChatConversation: A new ChatConversation instance containing conversation history from both operands.
        """
        return ChatConversation(self.conversation_history + other.conversation_history)

    def add_pair(self, message: ChatSequence, response: ResponseDict):
        self.conversation_history.append(ChatPair(message, response))
    
    def fill_conversation(self, updated_history: List[ChatPair]) -> 'ChatConversation':
        self.conversation_history = updated_history   
        return self  
    
    def sequential_complete(self, **kwargs) -> None:
        
        # this will override existing values for the responses
        from aitemplates.oai.responses.chat_response import create_chat_completion
        
        for i in range(len(self.conversation_history)):
            chat_pair = self.conversation_history[i]

            # Generate a response for the current ChatSequence
            openai_response = create_chat_completion(chat_pair.prompt, send_object=True, **kwargs)

            # Update the response in the current ChatPair
            chat_pair.update_response(ResponseDict(openai_response))

            print(chat_pair.response)
            # If this is not the last ChatPair, append the response message to the next ChatSequence
            if i + 1 < len(self.conversation_history):
                next_chat_sequence = self.conversation_history[i + 1].prompt
                next_chat_sequence.append(chat_pair.response.message)

    def display_conversation(self):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message_response_pair in self.conversation_history:
            prompt, response = message_response_pair
            
            messages = prompt.raw()
            
            for message in messages:
                formatted_message = ""
                if message["role"] == "system":
                    formatted_message = f"system: {message['content']}\n"
                elif message["role"] == "user":
                    formatted_message = f"user: {message['content']}\n"
                elif message["role"] == "assistant" and message.get("function_call"):
                    formatted_message = f"assistant: {message['function_call']}\n"
                elif message["role"] == "assistant" and not message.get("function_call"):
                    formatted_message = f"assistant: {message['content']}\n"
                elif message["role"] == "function":
                    formatted_message = f"function ({message['name']}): {message['content']}\n"
                
                print(
                    colored(
                        formatted_message,
                        role_to_color[message["role"]],
                    )
                )
                
            

            response_message = response.message.content
            
            print(response_message, role_to_color[response_message["role"]])
            print(colored(f"Response: {response_message} \n\n", role_to_color[response_message["role"]]))

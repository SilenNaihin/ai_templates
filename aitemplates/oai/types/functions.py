from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union, Tuple, Any, Callable, cast

from jsonschema import validate
    
class FunctionPair:
    function_pairs: Tuple[object, Callable[..., Any]]
    
    def __init__(self, function_def: object, function: Callable[..., Any]) -> None:
        self.function_pairs = (function_def, function)
    
    def __iter__(self):
        return iter(self.function_pairs)
    
    @property
    def definition(self) -> object:
        return self.function_pairs[0]
    
    @property
    def function(self) -> Callable[..., Any]:
        return self.function_pairs[1]
    
    def call(self, *args, **kwargs) -> Any:
        return self.function.call(*args, **kwargs)

    def update__def(self, new_def: object) -> 'FunctionPair':
        self.function_pairs = (new_def, self.function_pairs[1])
        return self

    def update_function(self, new_function: Callable[..., Any]) -> 'FunctionPair':
        self.function_pairs = (self.function_pairs[0], new_function)
        return self
   
@dataclass
class FunctionsAvailable:
    function_pairs: List[FunctionPair]

    def __getitem__(self, i: int) -> FunctionPair:
        return self.function_pairs[i]

    def __iter__(self):
        return iter(self.function_pairs)

    def __len__(self) -> int:
        return len(self.function_pairs)

    def __add__(self, other: 'FunctionsAvailable') -> 'FunctionsAvailable':
        return FunctionsAvailable(self.function_pairs + other.function_pairs)
    
    def get_function_defs(self) -> List[object]:
        return [definition for definition, _ in self.function_pairs]
    
    def get_functions(self) -> List[Callable[..., Any]]:
        return [cast(Callable[..., Any], function) for _, function in self.function_pairs]

    def add_function_pairs(self, functions: Union[FunctionPair, List[FunctionPair], 'FunctionsAvailable']):
        if isinstance(functions, FunctionsAvailable):
            self.function_pairs.extend(functions.function_pairs)
        elif isinstance(functions, list):
            self.function_pairs.extend(functions)
        else:
            self.function_pairs.append(functions)

    def set_function_pairs(self, functions: Union[FunctionPair, List[FunctionPair], 'FunctionsAvailable']):
        if isinstance(functions, FunctionsAvailable):
            self.function_pairs = functions.function_pairs
        elif isinstance(functions, list):
            self.function_pairs = functions
        else:
            self.function_pairs = [functions]
    
    def append(self, function_pair: FunctionPair):
        return self.function_pairs.append(function_pair)

    def extend(self, function_pairs: List[FunctionPair] | FunctionsAvailable):
        if isinstance(function_pairs, FunctionsAvailable):
            self.function_pairs.extend(function_pairs.function_pairs)
        else:
            self.function_pairs.extend(function_pairs)
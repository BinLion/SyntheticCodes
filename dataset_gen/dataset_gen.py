

from typing import Protocol
from pydantic import BaseModel


class Exercise(BaseModel):
    problem: str
    solution: str


class Result(BaseModel):
    prompt: str
    output: str
    
class Generator(Protocol):
    def generate(self, prompt: str) -> Result:
        ...
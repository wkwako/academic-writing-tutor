from typing import TypedDict, Annotated

from operator import add

class TutorState(TypedDict):
    passage: str
    prompt: str
    history: str
    critiques: Annotated[list, add]
    feedback: str


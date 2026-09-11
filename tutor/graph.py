from typing import TypedDict, Annotated

from operator import add

class TutorState(TypedDict):
    passage: str
    prompt: str
    history: str
    critiques: Annotated[list, add]
    feedback: str

def grammar_node(state):
    passage = state["passage"]

    # --- the LLM call goes here ---
    # send `passage` to a model with a grammar-focused prompt,
    # get back a critique string
    #critique = call_model_for_grammar(passage)   # placeholder for now

    critique = "Grammar recs"

    return {"critiques": [f"GRAMMAR: {critique}"]}

def vocab_node(state):
    passage = state["passage"]

    #LLM call goes here

    critique = "Vocab recs"

    return {"critiques": [f"VOCAB: {critique}"]}

def topic_node(state):
    passage = state["passage"]

    #LLM call goes here
    
    critique = "Topic recs"

    return {"critiques": [f"TOPIC: {critique}"]}

def structure_node(state):
    passage = state["passage"]

    #LLM call goes here
    
    critique = "Structure recs"

    return {"critiques": [f"STRUCTURE: {critique}"]}

def para_anatomy_node(state):
    passage = state["passage"]

    #LLM call goes here
    
    critique = "Paragraph anatomy recs"

    return {"critiques": [f"PARAGRAPH ANATOMY: {critique}"]}
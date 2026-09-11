from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

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

def synthesis_node(state):
    all_critiques = state["critiques"]
    return {"feedback": f"[synthesis of {len(all_critiques)} critiques]"}

def build_tutor_graph():

    #create the builder
    builder = StateGraph(TutorState)

    #register nodes
    builder.add_node("grammar", grammar_node)
    builder.add_node("vocabulary", vocab_node)
    builder.add_node("topic", topic_node)
    builder.add_node("structure", structure_node)
    builder.add_node("para_anatomy", para_anatomy_node)
    builder.add_node("synthesis", synthesis_node)

    #fan-out: connect START to each analyzer
    builder.add_edge(START, "grammar")
    builder.add_edge(START, "vocabulary")
    builder.add_edge(START, "topic")
    builder.add_edge(START, "structure")
    builder.add_edge(START, "para_anatomy")

    #fan-in: connect each analyzer to synthesis
    builder.add_edge("grammar", "synthesis")
    builder.add_edge("vocabulary", "synthesis")
    builder.add_edge("topic", "synthesis")
    builder.add_edge("structure", "synthesis")
    builder.add_edge("para_anatomy", "synthesis")

    #connect synthesis to END
    builder.add_edge("synthesis", END)

    return builder.compile()

#manual testing without django
if __name__ == "__main__":
    graph = build_tutor_graph()
    initial_state = {
        "passage": "Some test paragraph to analyze.",
        "prompt": "",
        "history": "",
        "critiques": [],      # starts empty — the reducer fills it
        "feedback": "",       # synthesis fills it
    }

    result = graph.invoke(initial_state)

    print("=== CRITIQUES ===")
    for c in result["critiques"]:
        print(c)
    print("=== FEEDBACK ===")
    print(result["feedback"])
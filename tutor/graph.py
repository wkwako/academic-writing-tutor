from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
load_dotenv()

import anthropic

class TutorState(TypedDict):
    passage: str
    prompt: str
    history: str
    critiques: Annotated[list, add]
    feedback: str

class TutorGraph:
    def __init__(self, max_tokens=100):
        self.cheap_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.strong_model = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)
        self.graph = self._build()

    def grammar_node(self, state):
        passage = state["passage"]
        system_prompt = """You are a grammar analyzer for academic writing.
                           Examine the passage for grammatical issues: verb tense,
                           punctuation, capitalization, agreement, and sentence-level
                           errors. Provide specific, constructive feedback. Do not
                           rewrite the passage; describe what to fix and why."""

        response = self.cheap_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage}
        ])

        return {"critiques": [f"GRAMMAR: {response.content}"]}

    def vocab_node(self, state):
        passage = state["passage"]

        #LLM call goes here

        critique = "Vocab recs"

        return {"critiques": [f"VOCAB: {critique}"]}

    def topic_node(self, state):
        passage = state["passage"]

        #LLM call goes here
        
        critique = "Topic recs"

        return {"critiques": [f"TOPIC: {critique}"]}

    def structure_node(self, state):
        passage = state["passage"]

        #LLM call goes here
        
        critique = "Structure recs"

        return {"critiques": [f"STRUCTURE: {critique}"]}

    def para_anatomy_node(self, state):
        passage = state["passage"]

        #LLM call goes here
        
        critique = "Paragraph anatomy recs"

        return {"critiques": [f"PARAGRAPH ANATOMY: {critique}"]}

    def synthesis_node(self, state):
        all_critiques = state["critiques"]
        return {"feedback": f"[synthesis of {len(all_critiques)} critiques]"}

    def _build(self):
        #create the builder
        builder = StateGraph(TutorState)

        #register nodes
        builder.add_node("grammar", self.grammar_node)
        builder.add_node("vocabulary", self.vocab_node)
        builder.add_node("topic", self.topic_node)
        builder.add_node("structure", self.structure_node)
        builder.add_node("para_anatomy", self.para_anatomy_node)
        builder.add_node("synthesis", self.synthesis_node)

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
# if __name__ == "__main__":
#     graph = build_tutor_graph()
#     initial_state = {
#         "passage": "Some test paragraph to analyze.",
#         "prompt": "",
#         "history": "",
#         "critiques": [],      # starts empty — the reducer fills it
#         "feedback": "",       # synthesis fills it
#     }

#     result = graph.invoke(initial_state)

#     print("=== CRITIQUES ===")
#     for c in result["critiques"]:
#         print(c)
#     print("=== FEEDBACK ===")
#     print(result["feedback"])
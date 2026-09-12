from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import static_variables
load_dotenv()

class TutorState(TypedDict):
    passage: str #what the user has written
    prompt: str #the prompt we send to the llm
    history: str #previous passages and their feedback
    critiques: Annotated[list, add]
    topic: str #what it's about
    purpose: str #what it's for

class TutorGraph:
    def __init__(self, max_tokens=1000):
        self.cheap_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.strong_model = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)
        self.graph = self._build()

    def grammar_node(self, state):
        passage = state["passage"]

        response = self.cheap_model.invoke([
            {"role": "system", "content": static_variables.grammar_prompt()},
            {"role": "user", "content": passage}
        ])

        return {"critiques": [f"GRAMMAR: {response.content}"]}

    def vocab_node(self, state):
        passage = state["passage"]
        history = state["history"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": static_variables.vocab_prompt()},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"VOCAB: {response.content}"]}

    def topic_node(self, state):
        passage = state["passage"]
        history = state["history"]
        topic = state["topic"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": static_variables.topic_prompt(topic)},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"TOPIC: {response.content}"]}

    def structure_node(self, state):
        passage = state["passage"]
        history = state["history"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": static_variables.structure_prompt()},
            {"role": "user", "content": user_content}
        ])
        
        return {"critiques": [f"STRUCTURE: {response.content}"]}

    def para_anatomy_node(self, state):
        passage = state["passage"]
        history = state["history"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": static_variables.para_anatomy_prompt()},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"PARAGRAPH ANATOMY: {response.content}"]}

    def purpose_node(self, state):
        passage = state["passage"]
        history = state["history"]
        purpose = state["purpose"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": static_variables.purpose_prompt(purpose)},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"PURPOSE: {response.content}"]}

    def synthesis_node(self, state):
        passage = state["passage"]
        purpose = state["purpose"] or "general academic writing"
        all_critiques = state["critiques"]

        critiques_block = "\n\n".join(all_critiques)
        user_content = (
            f"Current passage: \n{passage}\n\n"
            f"Critiques of the current passage: \n{critiques_block}"
        )

        response = self.strong_model.invoke([
                    {"role": "system", "content": static_variables.synthesis_prompt(purpose)},
                    {"role": "user", "content": user_content}
                ])

        return {"feedback": response.content}

    def _build(self):
        #create the builder
        builder = StateGraph(TutorState)

        #register nodes
        builder.add_node("grammar", self.grammar_node)
        builder.add_node("vocabulary", self.vocab_node)
        builder.add_node("topic", self.topic_node)
        builder.add_node("structure", self.structure_node)
        builder.add_node("para_anatomy", self.para_anatomy_node)
        builder.add_node("purpose", self.purpose_node)
        builder.add_node("synthesis", self.synthesis_node)

        #fan-out: connect START to each analyzer
        builder.add_edge(START, "grammar")
        builder.add_edge(START, "vocabulary")
        builder.add_edge(START, "topic")
        builder.add_edge(START, "structure")
        builder.add_edge(START, "para_anatomy")
        builder.add_edge(START, "purpose")

        #fan-in: connect each analyzer to synthesis
        builder.add_edge("grammar", "synthesis")
        builder.add_edge("vocabulary", "synthesis")
        builder.add_edge("topic", "synthesis")
        builder.add_edge("structure", "synthesis")
        builder.add_edge("para_anatomy", "synthesis")
        builder.add_edge("purpose", "synthesis")

        #connect synthesis to END
        builder.add_edge("synthesis", END)

        return builder.compile()

    def run(self, passage, prompt, history=""):
        initial_state = {
            "passage": passage, "prompt": prompt, "history": history, "critiques": [], "feedback": ""
        }

        return self.graph.invoke(initial_state)

if __name__ == "__main__":
    tutor = TutorGraph()
    state: TutorState = {
        "passage": "She doesn't know where they're going to put its bags.",
        "prompt": "", "history": "Last passage: She dont know where there going to put they're bags.", "critiques": [],
        "topic": "", "purpose": "",
    }
    print(tutor.run(state))
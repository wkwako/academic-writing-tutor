from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from . import static_variables
load_dotenv()

class TutorState(TypedDict):
    passage: str #what the user has written
    history: str #previous passages and their feedback
    critiques: Annotated[list, add] #critiques from each analyzer
    topic: str #what it's about
    purpose: str #what it's for
    feedback: str #what the student sees (built by synthesizer)

class TutorGraph:
    def __init__(self, max_tokens=20000):
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

    def run(self, passage, topic, purpose, history=""):
        initial_state = {
            "passage": passage, "topic": topic, "purpose": purpose,"history": history, "critiques": [], "feedback": ""
        }

        return self.graph.invoke(initial_state)

if __name__ == "__main__":
    tutor = TutorGraph()
    passage = "The French Revolution were caused by many things. There was economic problems, social inequality, and enlightenment ideas that spreaded through france. The common people, who was called the Third Estate, they paid most of the taxes while the nobility payed almost nothing. This made people very angry and upset and mad. Bread prices was also very high, and many people could not afford to eat, which is similar to how modern inflation affects grocery costs today. The king, Louis XVI, he was not a strong leader and he failed to fix the countrys financial crisis. Enlightenment thinkers like Rousseau and Voltaire, they wrote about liberty and equality. These ideas made people question the monarchy. In conclusion there was many causes of the french revolution and it changed history forever."
    topic = "Discuss the causes of the French Revolution"
    purpose = "a first-year undergraduate history essay"

    print(tutor.run(passage, topic, purpose))
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from operator import add
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from . import static_variables
import random
load_dotenv()

class TutorState(TypedDict):
    passage: str #what the user has written
    history: str #previous passages and their feedback
    critiques: Annotated[list, add] #critiques from each analyzer
    topic: str #what it's about
    purpose: str #what it's for
    feedback: str #what the student sees (built by synthesizer)
    criteria: list
    exemplars: list

class TutorGraph:
    def __init__(self, max_tokens=20000):
        self.cheap_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.strong_model = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.store = Chroma(collection_name="corpus", persist_directory="chroma_db", embedding_function=embeddings)

        self.analyzers = {
            "grammar": self.grammar_node,
            "vocabulary": self.vocab_node,
            "topic": self.topic_node,
            "structure": self.structure_node,
            "para_anatomy": self.para_anatomy_node,
            "purpose": self.purpose_node,
        }

        self.category_map = {
            "Undergraduate essay": "undergrad_essay",
            "Graduate admissions statement": "grad_admissions_statement",
        }

    def retrieve_exemplars(self, category, n=2):
        internal_category = self.category_map.get(category, "")
        if not internal_category:
            return []

        results = self.store.get(where={"$and": [
            {"category": internal_category},
            {"kind": "exemplars"},
        ]})

        # Group chunks back into whole essays, keyed by source filename.
        essays = {}
        for text, meta in zip(results["documents"], results["metadatas"]):
            essays.setdefault(meta["source"], []).append(text)

        if not essays:
            return []

        # Sample 1–n whole essays at random (genre-based, never similarity).
        chosen = random.sample(list(essays), min(n, len(essays)))
        return ["\n".join(essays[source]) for source in chosen]
    
    def retrieve_criteria(self, category):
        internal_category = self.category_map.get(category, "")
        if not internal_category:
            return []

        results = self.store.get(where={"$and": [
            {"category": internal_category},
            {"kind": "guide"},
        ]})

        return results["documents"]

    def grammar_node(self, state):
        passage = state["passage"]

        response = self.strong_model.invoke([
            {"role": "system", "content": static_variables.grammar_prompt()},
            {"role": "user", "content": passage}
        ])

        return {"critiques": [f"GRAMMAR: {self._extract_text(response)}"]}

    def vocab_node(self, state):
        passage = state["passage"]
        history = state["history"]
        exemplars = state["exemplars"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.strong_model.invoke([
            {"role": "system", "content": static_variables.vocab_prompt(exemplars)},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"VOCAB: {self._extract_text(response)}"]}

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

        response = self.strong_model.invoke([
            {"role": "system", "content": static_variables.topic_prompt(topic)},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"TOPIC: {self._extract_text(response)}"]}

    def structure_node(self, state):
        passage = state["passage"]
        history = state["history"]
        criteria = state["criteria"]
        exemplars = state["exemplars"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.strong_model.invoke([
            {"role": "system", "content": static_variables.structure_prompt(criteria, exemplars)},
            {"role": "user", "content": user_content}
        ])
        
        return {"critiques": [f"STRUCTURE: {self._extract_text(response)}"]}

    def para_anatomy_node(self, state):
        passage = state["passage"]
        history = state["history"]
        exemplars = state["exemplars"]

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.strong_model.invoke([
            {"role": "system", "content": static_variables.para_anatomy_prompt(exemplars)},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"PARAGRAPH ANATOMY: {self._extract_text(response)}"]}

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

        response = self.strong_model.invoke([
            {"role": "system", "content": static_variables.purpose_prompt(purpose)},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"PURPOSE: {self._extract_text(response)}"]}

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

        return {"feedback": self._extract_text(response)}

    def _build(self, enabled):
        builder = StateGraph(TutorState)
        builder.add_node("synthesis", self.synthesis_node)

        for name in enabled:
            builder.add_node(name, self.analyzers[name])
            builder.add_edge(START, name)
            builder.add_edge(name, "synthesis")

        builder.add_edge("synthesis", END)
        return builder.compile()

    def _extract_text(self, response):
        content = response.content
        if isinstance(content, str):
            return content
        # content is a list of blocks; concatenate the text blocks
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "".join(parts)

    def run(self, passage, topic, purpose, category, history="", enabled=None):
        if enabled is None:
            enabled = list(self.analyzers)

        enabled = [name for name in enabled if name in self.analyzers]

        if not enabled:
            return {"feedback": "Please select at least one type of feedback."}

        graph = self._build(enabled)

        initial_state = {
            "passage": passage, "topic": topic, "purpose": purpose,
            "history": history, "critiques": [], "feedback": "",
            "criteria": self.retrieve_criteria(category),
            "exemplars": self.retrieve_exemplars(category),
        }

        return graph.invoke(initial_state)

if __name__ == "__main__":
    tutor = TutorGraph()
    # passage = "The French Revolution were caused by many things. There was economic problems, social inequality, and enlightenment ideas that spreaded through france. The common people, who was called the Third Estate, they paid most of the taxes while the nobility payed almost nothing. This made people very angry and upset and mad. Bread prices was also very high, and many people could not afford to eat, which is similar to how modern inflation affects grocery costs today. The king, Louis XVI, he was not a strong leader and he failed to fix the countrys financial crisis. Enlightenment thinkers like Rousseau and Voltaire, they wrote about liberty and equality. These ideas made people question the monarchy. In conclusion there was many causes of the french revolution and it changed history forever."
    # topic = "Discuss the causes of the French Revolution"
    # purpose = "a first-year undergraduate history essay"
    # print(tutor.run(passage, topic, purpose, category="Undergraduate essay"))

    print (tutor.retrieve_criteria("Undergraduate essay"))
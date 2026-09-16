from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
    criteria: list

class TutorGraph:
    def __init__(self, max_tokens=20000):
        self.cheap_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.strong_model = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.store = Chroma(collection_name="criteria", persist_directory="chroma_db", embedding_function=embeddings)

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
        
    def retrieve_criteria(self, category, k=3):
        criteria = []
        query = f"What makes a strong {category}"
        internal_category = self.category_map.get(category, "")
        if not internal_category:
            return []
        for doc, score in self.store.similarity_search_with_score(query, k=k, filter={"category": internal_category}):
            if score < 1.2:
                criteria.append(doc.page_content)

        return criteria

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

    def _build(self, enabled):
        builder = StateGraph(TutorState)
        builder.add_node("synthesis", self.synthesis_node)

        for name in enabled:
            builder.add_node(name, self.analyzers[name])
            builder.add_edge(START, name)
            builder.add_edge(name, "synthesis")

        builder.add_edge("synthesis", END)
        return builder.compile()

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
        }

        return graph.invoke(initial_state)

if __name__ == "__main__":
    tutor = TutorGraph()
    passage = "The French Revolution were caused by many things. There was economic problems, social inequality, and enlightenment ideas that spreaded through france. The common people, who was called the Third Estate, they paid most of the taxes while the nobility payed almost nothing. This made people very angry and upset and mad. Bread prices was also very high, and many people could not afford to eat, which is similar to how modern inflation affects grocery costs today. The king, Louis XVI, he was not a strong leader and he failed to fix the countrys financial crisis. Enlightenment thinkers like Rousseau and Voltaire, they wrote about liberty and equality. These ideas made people question the monarchy. In conclusion there was many causes of the french revolution and it changed history forever."
    topic = "Discuss the causes of the French Revolution"
    purpose = "a first-year undergraduate history essay"

    print(tutor.run(passage, topic, purpose))
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
load_dotenv()

import anthropic

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
        history = state["history"]

        system_prompt = """You are a vocabulary analyzer for academic writing.
            Assess word choice, precision, register, and variety.
            If a previous draft is provided, comment on whether the
            vocabulary has improved relative to it. Give specific,
            constructive feedback; do not rewrite the passage."""

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"VOCAB: {response.content}"]}

    def topic_node(self, state):
        passage = state["passage"]
        history = state["history"]
        topic = state["topic"]

        system_prompt = f"""You are a topic analyzer for academic
            writing. The topic/prompt the user was given is: {topic}.
            Your single job is to assess whether the passage stays relevant
            to and consistent with that topic. Look at this on two levels:
            whether the passage as a whole and its individual paragraphs
            stay on topic, and whether specific sentences drift, digress,
            or introduce material unrelated to the topic. Focus only on
            relevance and consistency — whether what's present belongs.
            Do not evaluate the quality of the writing, the strength of
            the argument, grammar, or style; other analyzers handle those.
            Do not penalize the writer for omitting subtopics they did not
            set out to cover, and do not invent requirements the topic does
            not explicitly state. If a previous draft is provided, comment
            on whether topic adherence has improved relative to it. Give specific,
            constructive feedback pointing to where the passage does or does
            not stay on topic. Do not rewrite the passage."""

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"TOPIC: {response.content}"]}

    def structure_node(self, state):
        passage = state["passage"]
        history = state["history"]

        system_prompt = f"""You are a structure analyzer for academic writing.
            Your job is to assess the overall organization of the passage as a whole —
            not individual sentences or word choice. Evaluate: whether there is a clear
            thesis or central point; whether an introduction and conclusion are present
            and do their jobs; whether the paragraphs are ordered logically and flow
            from one to the next; and whether the overall arc of the passage holds together.
            Work only at the level of the whole document and how its parts fit together.
            Do not evaluate grammar, word choice, or the internal construction of individual
            paragraphs — other analyzers handle those.If a previous draft is provided, comment
            on whether the overall structure has improved relative to it. Give specific,
            constructive feedback about the passage's organization. Do not rewrite the passage."""

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])
        
        return {"critiques": [f"STRUCTURE: {response.content}"]}

    def para_anatomy_node(self, state):
        passage = state["passage"]
        history = state["history"]

        system_prompt = f"""You are a paragraph analyzer for academic writing.
            Your job is to assess the internal construction of individual paragraphs — 
            not the overall document structure or word-level grammar. Evaluate each
            paragraph for: a clear topic sentence; a concluding or transitional sentence
            where appropriate; coherence between the sentences within the paragraph;
            and sentence variety — whether sentence lengths and structures vary, or
            whether they are monotonous or overloaded (for example, overuse of semicolons
            or em dashes). Work only at the level of individual paragraphs and the
            sentences inside them. Do not evaluate the overall document organization
            or grammar correctness — other analyzers handle those. Give specific,
            constructive feedback, referring to particular paragraphs. Do not rewrite
            the passage."""

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"PARAGRAPH ANATOMY: {response.content}"]}

    def purpose_node(self, state):
        passage = state["passage"]
        history = state["history"]
        purpose = state["purpose"]

        system_prompt = f"""You are a purpose analyzer for academic writing.
            The stated purpose of this writing is: {purpose}. Your job is to assess,
            holistically, how well the passage achieves that purpose — whether it
            is an effective, strong piece of writing for what it is meant to do.
            Different purposes have different standards: what makes a strong blog
            post differs from what makes a strong cover letter, lab report, or
            fellowship essay. Judge the passage against the standards appropriate
            to its stated purpose. Where the passage would more strongly achieve
            what it is evidently trying to do, say so — for example, if a claim
            would be more convincing with specific evidence the writer seems
            positioned to provide. Frame such feedback as ways to better accomplish
            the writer's own goal, not as new topics to add, and do not invent
            requirements the purpose does not imply. Work at the level of the whole
            passage and its overall effectiveness. Do not give sentence-level grammar,
            word-choice, or paragraph-construction feedback — other analyzers handle those.
            Give specific, constructive feedback about how well the passage serves its purpose.
            Do not rewrite the passage."""

        if history:
            user_content = (
                f"Previous draft and prior feedback: \n{history}\n\n"
                f"Current passage: \n{passage}"
            )
        else:
            user_content = f"Current passage: \n{passage}"

        response = self.cheap_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        return {"critiques": [f"PURPOSE: {response.content}"]}

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
    test_state: TutorState = {
        "passage": "She doesn't know where they're going to put its bags.",
        "prompt": "", "history": "Last passage: She dont know where there going to put they're bags.", "critiques": [],
        "topic": "", "purpose": "",
    }
    print(tutor.vocab_node(test_state))
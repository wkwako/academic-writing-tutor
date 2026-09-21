import json
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tutor.graph import TutorGraph
import os
load_dotenv()

class Evaluation():
    def __init__(self, max_tokens=50000):
        #define passage generator model
        self.model_a = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)

        #feedback generator models
        self.tutor = TutorGraph(max_tokens=max_tokens)
        self.cheap_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.strong_model = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)
        self.tested_models = {"webapp": None, "cheap": self.cheap_model, "strong": self.strong_model}
    
        #define passage rewriter model
        self.model_b = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)
    
        #define evaluator model
        self.model_c = ChatOpenAI(model="gpt-5.6-terra", max_tokens=max_tokens)

        self.topic = "A personal statement"
        self.purpose = "Graduate school application"

        self.results = {}

    def _extract_text(self, response):
        content = response.content
        if isinstance(content, str):
            return content
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "".join(parts)

    def generate_passage(self, model, field="an unspecified field", flaw_focus="a mix of the flaw types below"):
        prompt = f"""Write a two-paragraph personal statement for a graduate school application in {field}.

        The statement should read like a real but flawed first draft from an applicant who is not a strong writer. The applicant HAS concrete, specific content — real experiences, named activities, particular details — but expresses it poorly. Introduce genuine, realistic weaknesses of these kinds ONLY:
        - grammatical errors (subject-verb agreement, verb tense, run-ons, comma splices, apostrophes)
        - awkward or imprecise word choice, and wrong-register or repetitive vocabulary
        - tangled or monotonous sentence structure
        - disorganized paragraph structure — ideas in an illogical order, a buried or missing thesis, weak transitions, a conclusion that doesn't land

        Do NOT make the flaws be missing content, vagueness, or generic unsupported claims. The applicant should already include specific examples and concrete details; the problems should be in how that material is written and organized, not in whether it exists. Every weakness you introduce should be fixable by correcting or rearranging what is already on the page, without needing to invent new facts.

        Make the flaws realistic and uneven — a real draft has some sentences that work and some that don't. Weight this particular draft toward these flaws in particular: {flaw_focus}. Do not make it a parody. Do not include any commentary, labels, or notes about the flaws — output only the statement itself."""
        response = model.invoke(prompt)
        return self._extract_text(response)

    def generate_feedback(self, name, model, passage):
        if name == "webapp":
            result = self.tutor.run(
                passage,
                topic=self.topic,
                purpose=self.purpose,
                category="Graduate admissions statement",
                history="",
                enabled=None,   # all analyzers
            )
            return result["feedback"]

        prompt = f"""You are an academic writing tutor. A student has submitted the following passage for feedback.

        Topic: {self.topic}
        Purpose: {self.purpose}

        Passage:
        {passage}

        Give the student thorough, specific, constructive feedback on how to improve this passage. Consider all relevant dimensions: grammar and mechanics, vocabulary and word choice, how well it stays on topic, overall structure and organization, the internal construction of individual paragraphs, and how well it achieves its stated purpose. Prioritize what matters most.

        Write your feedback as flowing prose addressed to the student. Do not use headers, bold text, or bullet points; write in plain paragraphs. Do not rewrite the passage for them; describe what to improve and why."""
        response = model.invoke(prompt)
        return self._extract_text(response)

    def rewrite_passage(self, model, passage, feedback):
        prompt = f"""Below is a passage and a set of feedback on that passage. Revise the passage by applying the feedback.

        Apply only what the feedback actually says. Do not make improvements of your own that the feedback does not call for, and do not ignore feedback you disagree with. Your job is to faithfully execute the feedback, not to exercise your own editorial judgment. If the feedback is vague or unhelpful on some point, do the best you can with what it says rather than substituting your own ideas.

        Output only the revised passage, with no commentary.

        Passage:
        {passage}

        Feedback:
        {feedback}"""
        response = model.invoke(prompt)
        return self._extract_text(response)

    def generate_evaluation(self, model, rewritten_passage1, rewritten_passage2):
        prompt = f"""Two revised versions of a graduate school personal statement are shown below. Judge which is the stronger piece of writing for its stated context.

        Topic: {self.topic}
        Purpose: {self.purpose}

        Evaluate holistically: which version is more effective, well-written, and appropriate for a graduate school application? Consider clarity, specificity, organization, and how convincingly it serves its purpose.

        Your response must begin with exactly "passage1" or "passage2" on the first line — the identifier of the stronger version, and nothing else on that line. Then leave a blank line, then explain your reasoning.

        Passage 1:
        {rewritten_passage1}

        Passage 2:
        {rewritten_passage2}"""
        response = model.invoke(prompt)
        return self._extract_text(response)

    def write_result(self, num_result, passage, model1_name, feedback1, rewritten_passage1, model2_name, feedback2, rewritten_passage2, evaluation):
        first_line = evaluation.strip().split("\n")[0].lower()
        if "passage1" in first_line:
            winner = "passage1"
        elif "passage2" in first_line:
            winner = "passage2"
        else:
            winner = "unparseable"
        reason = evaluation

        info = {
            "passage": passage,
            "passage1_name": model1_name,
            "feedback1": feedback1,
            "rewritten_passage1": rewritten_passage1,
            "passage2_name": model2_name,
            "feedback2": feedback2,
            "rewritten_passage2": rewritten_passage2,
            "winner": winner,
            "winner_model": model1_name if winner == "passage1" else (model2_name if winner == "passage2" else "none"),
            "reason": reason,
        }

        self.results[num_result] = info

        # crash-safe: append this one record immediately
        with open("results.jsonl", "a") as f:
            f.write(json.dumps({"num_result": num_result, **info}) + "\n")

    def run_pairing(self, num_result, passage, field, flaw_focus, name1, name2):
        model1 = self.tested_models[name1]
        model2 = self.tested_models[name2]

        feedback1 = self.generate_feedback(name1, model1, passage)
        rewritten1 = self.rewrite_passage(self.model_b, passage, feedback1)
        feedback2 = self.generate_feedback(name2, model2, passage)
        rewritten2 = self.rewrite_passage(self.model_b, passage, feedback2)

        #both orderings to control position bias
        eval_ab = self.generate_evaluation(self.model_c, rewritten1, rewritten2)
        eval_ba = self.generate_evaluation(self.model_c, rewritten2, rewritten1)

        winner_ab = self._winner_name(eval_ab, name1, name2)   # slot1=name1
        winner_ba = self._winner_name(eval_ba, name2, name1)   # slot1=name2 (swapped)

        if winner_ab == winner_ba and winner_ab is not None:
            final_winner = winner_ab
        else:
            final_winner = "tie"   # order-dependent verdict, or an unparseable

        info = {
            "passage": passage,
            "field": field,
            "flaw_focus": flaw_focus,
            "model1_name": name1,
            "feedback1": feedback1,
            "rewritten1": rewritten1,
            "model2_name": name2,
            "feedback2": feedback2,
            "rewritten2": rewritten2,
            "winner_ab": winner_ab,
            "winner_ba": winner_ba,
            "final_winner": final_winner,
            "reason_ab": eval_ab,
            "reason_ba": eval_ba,
        }
        self.results[num_result] = info
        with open("results.jsonl", "a") as f:
            f.write(json.dumps({"num_result": num_result, **info}) + "\n")

    def _winner_name(self, evaluation, slot1_name, slot2_name):
        first_line = evaluation.strip().split("\n")[0].lower()
        if "passage1" in first_line:
            return slot1_name
        elif "passage2" in first_line:
            return slot2_name
        return None

    def evaluate(self, passage_set):
        open("results.jsonl", "w").close()   # clear file at start of run

        pairings = [("webapp", "cheap"), ("webapp", "strong"), ("cheap", "strong")]

        num_result = 0
        for passage, field, flaw_focus in passage_set:
            for name1, name2 in pairings:
                self.run_pairing(num_result, passage, field, flaw_focus, name1, name2)
                num_result += 1

    def generate_passage_set(self):
        specs = [
            #("public policy", "disorganized paragraph order and a buried thesis"),
            #("molecular biology", "subject-verb agreement errors and tense inconsistency"),
            #("comparative literature", "run-on sentences and comma splices"),
            #("mechanical engineering", "monotonous sentence structure and repetitive phrasing"),
            #("clinical psychology", "imprecise word choice and wrong register"),
            ("economics", "weak transitions and an ending that doesn't land"),
            ("environmental science", "a mix of grammar errors and tangled syntax"),
            ("art history", "awkward vocabulary and misplaced modifiers"),
            ("computer science", "illogical idea order and missing topic sentences"),
            ("public health", "apostrophe errors and inconsistent verb tense"),
        ]
        passage_set = []
        for field, flaw_focus in specs:
            passage = self.generate_passage(self.model_a, field=field, flaw_focus=flaw_focus)
            passage_set.append((passage, field, flaw_focus))
        return passage_set

if __name__ == "__main__":
    evaluator = Evaluation()
    passages = evaluator.generate_passage_set()
    evaluator.evaluate(passages)

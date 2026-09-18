import json
from langchain_anthropic import ChatAnthropic
import numpy as np
import random

class Evaluation():
    def __init__(self, max_tokens=40000):
        #define passage generator model
        self.model_a = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        
        #define feedback generator model
        self.model_b = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
    
        #define passage rewriter model
        self.model_c = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
    
        #define evaluator model
        self.model_d = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)

        self.model_webapp = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.cheap_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=max_tokens)
        self.strong_model = ChatAnthropic(model="claude-sonnet-5", max_tokens=max_tokens)
        self.tested_models = {"webapp": self.model_webapp, "cheap": self.cheap_model, "strong": self.strong_model}

        self.topic = "A personal statement"
        self.purpose = "Graduate school application"

        self.results = {}

    def generate_passage(self, model):
        #uses topic and purpose
        prompt = """prompt goes here. include topic, purpose, length, etc."""
        response = model.invoke(prompt)
        return response.content

    def generate_feedback(self, model, passage):
        #uses topic and purpose
        prompt = """prompt goes here. include topic, purpose, passage, etc."""
        response = model.invoke(prompt)
        return response.content

    def rewrite_passage(self, model, passage, feedback):
        #rewrite should not use topic or purpose at all; only the feedback
        prompt = """prompt goes here. do not use topic or purpose, only feedback."""
        response = model.invoke(prompt)
        return response.content

    def generate_evaluation(self, model, rewritten_passage1, rewritten_passage2):
        #uses topic and purpose
        prompt = """prompt goes here. use topic, purpose, and new passages. have LLM output "passage1" or "passage2" at top, then two newlines, then reason."""
        response = model.invoke(model)
        return response.content

    def write_result(self, num_result, passage, model1_name, feedback1, rewritten_passage1, model2_name, feedback2, rewritten_passage2, evaluation):
        #extract winner and reason here
        winner = evaluation[:8]
        reason = evaluation[8:]

        info = {
            "passage": passage,
            "passage1_name": model1_name,
            "feedback1": feedback1,
            "rewritten_passage1": rewritten_passage1,
            "passage2_name": model2_name,
            "feedback2": feedback2,
            "rewritten_passage2": rewritten_passage2,
            "winner": winner,
            "reason": reason,

        }
        #winner model name in info[info["winner"]+"_name"]
        self.results[num_result] = info

    def one_pass(self, num_result):

        selected = range(len(self.tested_models))
        selected_idxs = random.shuffle(selected)[:2]
        model1_name = self.tested_models.keys()[selected_idxs[0]]
        model2_name = self.tested_models.keys()[selected_idxs[1]]
        model1 = self.tested_models.values()[selected_idxs[0]]
        model2 = self.tested_models.values()[selected_idxs[1]]

        passage = self.generate_passage(self.model_a)
        feedback1 = self.generate_feedback(model1, passage)
        rewritten_passage1 = self.rewrite_passage(self.model_b, passage, feedback1)
        feedback2 = self.generate_feedback(model2, passage)
        rewritten_passage2 = self.rewrite_passage(self.model_b, passage, feedback2)

        evaluation = self.generate_evaluation(self.model_c, rewritten_passage1, rewritten_passage2)

        self.write_result(num_result, passage, model1_name, feedback1, rewritten_passage1, model2_name, feedback2, rewritten_passage2, evaluation)

    def evaluate(self, num_iterations):
        for i in num_iterations:
            self.one_pass(self, i)

        with open("results.txt") as f:
            json.dump(self.results, f, indent=2)
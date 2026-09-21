import json

class Analysis:
    def __init__(self, results_file):
        self.data = {}

        with open(results_file, "r") as file:
            for i, line in enumerate(file):
                self.data[i] = json.loads(line)

    def tally_results(self):
        tally = {}
        for key, value in self.data.items():
            winner = value["final_winner"]
            tally[winner] = tally.get(winner, 0) + 1

        print (tally)

    def tally_results2(self):
        #win list for each model
        #{"weak": [topics], "strong": [topics], "app": [topics]}
        tally = {}
        for key, value in self.data.items():
            winner = value["final_winner"]
            if winner == "tie":
                tally.setdefault(winner, []).append((value["model1_name"], value["model2_name"]))
            else:
                opponent = value["model1_name"] if winner == value["model2_name"] else value["model2_name"]
                tally.setdefault(winner, []).append(opponent)

        return tally

if __name__ == "__main__":
    analysis = Analysis("strong_analyzers_strong_synthesis.jsonl")
    print (analysis.tally_results2())

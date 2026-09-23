# Repo Organization

This project uses a typical Django hierarchical structure. The core files include:
- `graph.py` - The LangGraph graph structure, including node communication and LLM query logic.
- `views.py` - Passes variables between frontend and backend; renders the page.
- `static_variables.py` - Contains text for LLM queries.
- `index_corpus.py` - Handles Chroma db generation for RAG, including loading, chunking, embedding, and storing.
- `Evaluation.py` - Evaluates the efficacy of the tool as compared to LLM models of varying strengths. 
- `Analysis.py` - Performs analysis on files generated from Evaluation.py 

# Key Findings Summary:
1. Analyzer nodes are the limiting factor in performance: Changing analyzer nodes from using weak models to strong models improves performance, but changing the synthesizer node from a weak model to a strong model sees only minor improvement.
2. RAG shifts outcomes from ties toward decisive results: RAG made judge decisions more decisive but split evenly between wins and losses.
3. The LangGraph structure has a meaningful impact on performance: The tool outperforms a weak model, even when solely weak models are used in the graph structure. 

# Project Description
I created a web app that aims to help users improve their writing ability by generating targeted feedback on submitted passages. The passage is evaluated on several attributes, including grammar, vocabulary, topic adherence, structure, paragraph anatomy, and overall strength for its intended purpose. I use LangGraph to create nodes with a separate query for each of these attributes (fan-out), and a synthesis node is used to combine the feedback from each node into a coherent structure for the user (fan-in). Retrieval-augmented generation (RAG) is used to provide both exemplars and criteria to the LLM to support feedback generation. The web app currently supports feedback for undergraduate essays and graduate statements of purpose.

# Model Choice
I used several different LLM models as part of this project:
1. Anthropic's Claude Haiku 4-5-20251001: Used as the weak model in each graph node, and used to generate passages and feedback during the evaluation step.
2. Anthropic's Claude Sonnet 5: Used as the strong model in each graph node, and used to generate feedback and rewrite passages during the evaluation step.
3. OpenAI's GPT 5.6-Terra: Used during the evaluation step to evaluate and compare rewritten passages. I purposefully used a different model family for the comparison to minimize in-group bias.

## RAG
We use RAG to modify feedback generation by supplying several nodes with external information relevant to developing a strong essay. For the vocabulary, and paragraph anatomy nodes, we add exemplar essays to the prompt; and for the structure node, we add both exemplars and a list of criteria for what makes a strong essay to the prompt. Exemplars were obtained by searching for strong essays online, and criteria were obtained by retrieving guides that describe how to write strong essays. Chroma was used to load, chunk, embed, and store the information. A list of sources that we used for RAG can be viewed in the corpus/sources.md file.

# Evaluation

We evaluate the web app by comparing the quality of its feedback to output generated from a cheap model and a stronger model. This is a multi-step process:
1. Generate a passage with various errors.
2. Send the passage through three models to generate feedback for it: the LangGraph structure, a weak model, and a strong model.
3. Send those three passages separately through a model to rewrite the original passage based on the feedback.
4. Select each pairwise combination of rewritten passages, and send them to a model to select which passage is superior for its stated purpose and topic. During this step, I eliminate order bias by comparing both passages across two rounds, where passage is swapped between rounds. A passage only "wins" if it selected during both comparisons (wins when passage is first and wins when passage is second).
5. Repeat steps 1 through 4 ten times, and analyze the data to determine which model produces the strongest feedback. 

# Results

We vary model strength and perform several iterations of this evaluation: with weak analyzers, strong analyzers, and including/removing RAG. I display the results below in tables, where each cell represents wins-losses-ties in 10 games, and the column and row determine which models are being compared. Ties occur when two models are compared twice (with their order swapped in the second comparison), and the evaluator selects each one once. The results are displayed below:

1. Weak analyzers and a weak synthesis node, with RAG:

||webapp|weak|strong|
|---|---|---|---|
|webapp|---| 6-2-2 | 3-6-1 |
|weak| 2-6-2 |---| 1-7-2|
|strong| 6-3-1 | 7-1-2 |---|

The passage written using the feedback from the webapp is chosen more often over the passage written using the feedback from the weak model. However, the webapp cleanly loses to the strong model. The strong model dominates the weak model, as expected. 

2. Weak analyzers and a strong synthesis node, with RAG:

||webapp|weak|strong|
|---|---|---|---|
|webapp|---| 8-2-0 | 2-6-2 |
|weak| 2-8-0 |---| 2-6-2 |
|strong| 6-2-2 | 6-2-2 |---|

The webapp performs slightly better with a strong synthesis node, but the outcomes overall are the same as the weak synthesis node. 

3. Strong analyzers and a strong synthesis node, with RAG:

||webapp|weak|strong|
|---|---|---|---|
|webapp|---| 9-0-1 | 6-3-1 |
|weak| 0-9-1 |---| 0-9-1 |
|strong| 3-6-1 | 9-0-1 |---|

The web app performs better than both the weak and strong models. 

4. Strong analyzers and a strong synthesis node, no RAG:

The webapp performs well against the weak model but ties in half the rounds against the strong model. 

||webapp|weak|strong|
|---|---|---|---|
|webapp|---| 9-0-1 | 4-1-5 |
|weak| 0-9-1 |---| 0-8-2 |
|strong| 1-4-5 | 8-0-2 |---|

# Discussion

In experiment 1 versus experiment 2, the web app performs slightly better against the weak and strong models when the synthesis node uses the stronger model as compared to the weaker model. Although it still performs worse than the strong model (1 call to a strong model is better than 6 calls to a weak one), the LangGraph architecture appears to be doing real work, as it significantly outperforms the weak model. In experiment 3, we swap out the weak models performing the analysis for strong models, and the web app performs substantially better against both the weak and strong models. And for the first time, it achieves more wins than losses against the strong model. This indicates that calls to each analyzer node -- not the synthesis node -- is the limiting factor in performance.

Last, we use the same models in the analyzers and synthesis node as the previous experiment, but remove RAG functionality. The web app performs well against the weak model again, but less well against the strong model, as half the rounds end in ties. This suggests that RAG contributes enough to shift ties toward decisive results. LLMs already have access to the information we injected into each prompt, however, our results suggest that directly reminding the LLM of relevant material before asking for feedback may improve results. Therefore, RAG earns its keep in this project, and justifies its inclusion.

# Limitations
There are two limitations:
1. While the tool itself uses a low amount of tokens (6 calls per generated feedback), evaluation is expensive, with over 100 calls required per experiment. Our sample size for evaluation was n=10. With more resources at my disposal, I would have liked to randomize model selection during several of the evaluation steps to help eliminate bias and noise.
2. The evaluation step is not directly measuring feedback. Instead, it evaluates the end result of that feedback based on a rewritten passage by incorporating that feedback. This does provide useful information about the performance of the tool, but lacks the ability to demonstrate the effectiveness of the tool for humans. The chosen method cannot evaluate if the feedback is clear, actionable, and, most importantly, improves writing ability. However, our control conditions hold (strong beats weak in every experiment), which indicates the evaluation reliably discriminates feedback quality. The tool's standing relative to a single strong call depends on analyzer strength, as experiments 1–3 show.

# Design Considerations
Design considerations are choices that were made throughout the project that had multiple defensible paths, of which I chose only one. There are several considerations:
1. **Node choice.** We chose six nodes: vocabulary, grammar, topic adherence, structure, paragraph anatomy, and overall strength for the passage's intended purpose. There are other selections that likely would have produced similar results, but I chose these nodes because I felt that they were specific enough that an LLM could easily provide targeted feedback for each of them, and general enough that users would be roughly familiar with each category (and thus would understand feedback direction and purpose).
2. **LangGraph.** LangGraph was a reasonable choice for implementing this project due to its non-sequential structure. We use both fan-out and fan-in methods alongside a synthesis node to combine feedback. While this could have been implemented using LangChain, other tools, or manually calling LLMs sequentially, LangGraph allows for parallelization and a large boost in efficiency.
3. **Why RAG was used.** Outside of my desire to gain experience with RAG, I was really just curious about whether RAG could actually improve LLM performance when generating feedback for a written passage. 
4. **Evaluation method.** Evaluation is a vital step in project development and in research, and personally, it's one of my favorite parts. I decided to evaluate this tool using the several steps outlined above. Those steps could easily be added, subtracted, or models interchanged to evaluate this tool. 
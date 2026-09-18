from django.shortcuts import render
from .graph import TutorGraph

tutor = TutorGraph(max_tokens=20000)

ANALYZER_LABELS = [
    ("grammar", "Grammar"),
    ("vocabulary", "Vocabulary"),
    ("topic", "Topic adherence"),
    ("structure", "Structure"),
    ("para_anatomy", "Paragraph construction"),
    ("purpose", "Fitness for purpose"),
]

def tutor_page(request):
    categories = list(tutor.category_map)

    submitted_text = ""
    topic = ""
    purpose = ""
    category = categories[0]
    result = {"feedback": ""}
    enabled = [name for name, _ in ANALYZER_LABELS]

    if request.method == "POST":
        submitted_text = request.POST.get("passage", "")
        topic = request.POST.get("topic", "")
        purpose = request.POST.get("purpose", "")
        category = request.POST.get("category", categories[0])
        enabled = request.POST.getlist("analyzers")

        history = request.session.get("history", "")
        result = tutor.run(submitted_text, topic, purpose, category, history=history, enabled=enabled)

        request.session["history"] = (
            f"Previous draft:\n{submitted_text}\n\n"
            f"Feedback given on that draft:\n{result['feedback']}"
        )

    return render(request, "tutor/index.html", {
        "submitted_text": submitted_text,
        "feedback": result["feedback"],
        "topic_text": topic,
        "purpose_text": purpose,
        "categories": categories,
        "selected_category": category,
        "analyzers": [
            {"name": name, "label": label, "checked": name in enabled}
            for name, label in ANALYZER_LABELS
        ],
    })
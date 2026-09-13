from django.shortcuts import render
from .graph import TutorGraph

tutor = TutorGraph(max_tokens=20000)

def tutor_page(request):
    submitted_text = ""
    topic = ""
    purpose = ""
    result = {"feedback": ""}
    if request.method == "POST":
        submitted_text = request.POST.get("passage", "")
        print ("=== SUBMITTED PASSAGE ===")
        print (submitted_text)

        
        topic = request.POST.get("topic", "")
        purpose = request.POST.get("purpose", "")
        #topic = "Discuss the causes of the French Revolution"
        #purpose = "a first-year undergraduate history essay"
        result = tutor.run(submitted_text, topic, purpose)

        print ("=== FEEDBACK ===")
        print (result["feedback"])

    return render(request, "tutor/index.html", {
        "submitted_text": submitted_text,
        "feedback": result["feedback"],
        "topic_text": topic,
        "purpose_text": purpose})
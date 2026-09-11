from django.shortcuts import render

def tutor_page(request):
    submitted_text = ""
    if request.method == "POST":
        submitted_text = request.POST.get("passage", "")
        print("=== SUBMITTED PASSAGE ===")
        print(submitted_text)
        print("=========================")
    return render(request, "tutor/index.html", {"submitted_text": submitted_text})
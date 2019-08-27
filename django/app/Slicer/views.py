from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from .slicer import handle_research
from .models import Research
from Account.models import User, ExtendedUser
from django.shortcuts import render
import json

def research_list(request):
    researches = Research.objects.all()
    return render(request, "Slicer/research_list.html", {"researches": researches})

def upload_research(request):
    if request.method == "POST" and "file" in request.FILES: 
        research = request.FILES["file"]
        resp = handle_research(research)
        return HttpResponse(json.dumps(resp), content_type="application/json")
    return HttpResponse("{'ok': false, 'error': 'invalid request'}", content_type="application/json")

def view_research(request, id):
    if "id" not in request.session:
        return HttpResponseRedirect("/")
    user_id = request.session["id"]
    user = User.objects.get(id=user_id)
    ext_user = ExtendedUser.objects.get(userID=user_id)

    res = Research.objects.filter(id=id)
    if res.count() != 1:
        return HttpResponse("404")
    return render(request, "Slicer/view_research.html", {"research": res[0], "extUser": ext_user, "user": user})
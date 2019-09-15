from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from .slicer import handle_research
from .models import Research
from Account.models import User, ExtendedUser
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json, zipfile
import os

def zipfolder(path, ziph):
    for root, dirs, files in os.walk(path):
        for f in files:
            print(os.path.join(root, f))
            ziph.write(os.path.join(root, f))

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
    
    res = res[0]
    nods = json.loads(res.predictions_nods)
    return render(request, "Slicer/view_research.html", {
                "research": res,
                "extUser": ext_user,
                "user": user,
                "nods": nods,
            }
        )

@csrf_exempt
def kafka_processed(request):
    if request.method == "POST" and "data" in request.POST:
        msg = json.loads(request.POST["data"])
       
        print("TEST")
        print(msg)
        print(msg["nods"])

        if msg["code"] == "success":
            path = msg["path"]
            research_id = int(msg["id"])

            research = Research.objects.filter(id=research_id)

            if research.count() != 1:
                print("Invalid research id recieved from kafka!")
                return HttpResponse("Invalid research id recieved from kafka!")

            dir_path = os.path.join("static/research_storage/results/experiments", path, "_")
            zipf = zipfile.ZipFile(f"static/research_storage/results/zips/{path}.zip", "w", zipfile.ZIP_DEFLATED)
            zipfolder(dir_path, zipf)
            zipf.close()

            research = research[0]
            research.predictions_dir = path
            research.predictions_nods = json.dumps(msg["nods"])
            research.save()
        else:
            print("An error occured during the prediction!!!")

        return HttpResponse("OK")
    return HttpResponse("Invalid request")

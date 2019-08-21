from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.conf import settings
from django.shortcuts import render

from .models import ImageSeries, SeriesInfo, PredictionMask
from Account.models import User, ExtendedUser
from Slicer.slicer import GetCurrentPredictFromCSV
from django.utils import timezone
import os, json, random, string

def buildJSONResponse(responseData):
    return HttpResponse(json.dumps(responseData), content_type="application/json")

def image_series_view(request):
    if 'id' not in request.session:
        return HttpResponseRedirect('/')
    id = request.session['id']
    if not User.objects.filter(id=id):
        return HttpResponseRedirect('/')
    user = User.objects.get(id=id)
    extUser = ExtendedUser.objects.get(userID=id)

    if 'id' in request.GET:
        seriesInfo = SeriesInfo.objects.get(seriesID=request.GET['id'])
        masks = PredictionMask.objects.filter(seriesID=request.GET['id'])

        return render(request, 'Slicer/image_series_view.html', {
            'user': user,
            'extUser': extUser,
            'seriesInfo': seriesInfo,
            'masks': masks,
        })
    return HttpResponse('Invalid request')

def changeDocComment(request):
    if 'id' not in request.session:
        return HttpResponseRedirect('/')
    id = request.session['id']
    if not User.objects.filter(id=id):
        return HttpResponseRedirect('/')
    user = User.objects.get(id=id)
    extUser = ExtendedUser.objects.get(userID=id)

    if 'id' in request.POST and 'comment' in request.POST:
        comment = request.POST["comment"]
        seriesID = request.POST["id"]
        if comment == "":
            return buildJSONResponse({"ok": False, "msg": "Комментарий не должен быть пустым"})
        if len(comment) > 250:
            return buildJSONResponse({"ok": False, "msg": "Комментарий не должен быть длиннее 350 символов"})
        
        series = SeriesInfo.objects.get(seriesID=seriesID)
        series.doctorComment = comment
        series.doctorCommentDate = timezone.now()
        series.save()
        return buildJSONResponse({"ok": True})
    return buildJSONResponse({"ok": False, "msg": "Invalid request"})

def setPreview(request):
    if 'id' in request.POST and 'fileName' in request.POST:
        seriesID = request.POST["id"]
        fileName = request.POST["fileName"]

        series = SeriesInfo.objects.get(seriesID=seriesID)
        series.previewSlice = fileName
        series.save()
        return buildJSONResponse({"ok": True})
    return buildJSONResponse({"ok": False, "msg": "Invalid request"})

def uploadPredictionMask(request):
    if "id" not in request.session:
        return HttpResponseRedirect("/")

    requiredPostFields = ("mask_name", "mask_description", "series_id")
    if "file" not in request.FILES or not all(key in request.POST for key in requiredPostFields):
        return buildJSONResponse({"ok": False, "message": "Заполните все поля!"})
    
    file = request.FILES["file"]
    if (file.name.split('.')[-1] != "csv"):
        return buildJSONResponse({"ok": False, "message": "Файл должен иметь расширение .csv"})

    seriesID = request.POST["series_id"]
    researchInfo = SeriesInfo.objects.get(seriesID=seriesID)

    predict = GetCurrentPredictFromCSV(file.read(), researchInfo.source_id)
    if not predict["ok"]:
        return buildJSONResponse({"ok": False, "message": predict["err"]})

    fileName = default_storage.save("masks/"+file.name, file).split("/")[-1]

    maskFolder = ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    slicesDirPath = "{}/images/{}/".format(settings.MEDIA_ROOT, researchInfo.slices_dir)
    while os.path.exists(slicesDirPath + maskFolder):
        maskFoler += random.choice(string.ascii_lowercase)

    os.makedirs(slicesDirPath + maskFolder)

    mask = PredictionMask.objects.create(seriesID=request.POST["series_id"],
        maskName=request.POST["mask_name"], maskDescription=request.POST["mask_description"],
        maskFolder=maskFolder, fileName=fileName)
    mask.save()

    return buildJSONResponse({"ok": True, "maskID": mask.id})
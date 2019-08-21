from django.shortcuts import render
from django.core import serializers
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from .models import HashPassword, User, Feedback, ExtendedUser
from Frontend import settings
from Frontend.TelegramBot import send as telegram_send
from Slicer.models import ImageSeries, SeriesInfo
import os, json, time, subprocess, datetime, urllib

def runCommand(commands):
    subprocess.run(commands)

def runShell(path, sh_input = ''):
    shellscript = subprocess.Popen([settings.BASE_DIR + '/' + path], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = shellscript.communicate(sh_input)
    returncode = shellscript.returncode
    return returncode, stdout, stderr

def buildJSONResponse(responseData):
	return HttpResponse(json.dumps(responseData), content_type="application/json")

def getResearchView(request):
    if 'id' not in request.GET:
        return buildJSONResponse({"ok": False, "error": "Invalid request"})
    if not ImageSeries.objects.filter(id=request.GET['id']):
        return buildJSONResponse({"ok": False, "error": "This research doesn`t exists"})
    series = ImageSeries.objects.get(id=request.GET['id'])
    return buildJSONResponse({"ok": True, "dirname": series.media_path})

def account(request):
    if 'id' not in request.session:
        return HttpResponseRedirect('/')
    id = request.session['id']
    user = User.objects.get(id=id)
    extUser = ExtendedUser.objects.get(userID=id)
    return render(request, "Account/index.html", {"user": user, "extUser": extUser})

def view(request):
    if 'id' not in request.session:
        return HttpResponseRedirect('/')
    id = request.session['id']
    user = User.objects.filter(id=id)[0]
    extUser = ExtendedUser.objects.get(userID=id)

    data = list()

    series = SeriesInfo.objects.all()
    for i, s in enumerate(series):
        data.append({"id": i + 1, "series": s})

    return render(request, "Account/view.html", {"user": user, "data": data, "extUser": extUser})

def upload(request):
    if 'id' not in request.session:
        return HttpResponseRedirect('/')
    id = request.session['id']
    user = User.objects.filter(id=id)[0]
    extUser = ExtendedUser.objects.get(userID=id)

    return render(request, "Account/upload.html", {"user": user, "extUser": extUser})

def predict_view(request):
    if 'id' not in request.session:
        return HttpResponseRedirect('/')
    id = request.session['id']
    user = User.objects.filter(id=id)[0]
    extUser = ExtendedUser.objects.get(userID=id)

    return render(request, "Account/predict.html", {"user": user, "extUser": extUser})

def predict(request):
    runCommand(["docker", "exec", "-i", "-t", "root_jupyter_1_826396f9a729", "/bin/bash", "/radio/temp/for_npcmr/run.sh"])
    #runCommand(["/radio/temp/for_npcmr/run.sh"])
    return buildJSONResponse({"message": "", "success": True})

def encPasswd(request):
    if 'passwd' not in request.GET:
        return HttpResponse("Incorrect GET request")
    return HttpResponse(HashPassword(request.GET['passwd']))

def feedback(request):
    if 'id' not in request.session:
        return buildJSONResponse({"message": 'Error: you are not authorized', "success": True})
    if 'title' not in request.POST or 'text' not in request.POST:
        return buildJSONResponse({"message": 'Error: invalid request data', "success": True})
    
    feedback = Feedback.objects.create(user_id=request.session['id'], title=request.POST['title'],
                text=request.POST['text'], time=datetime.datetime.now())
    feedback.save()

    if settings.TELEGRAM_FEEDBACK:
        client_ip = str(request.META['REMOTE_ADDR'])
        geo_ip_key = '7b9395a73758350d433f400a27280e69'

        geo_url = 'http://api.ipstack.com/{}?access_key={}'.format(client_ip, geo_ip_key)

        with urllib.request.urlopen(geo_url) as url:
            geo_info = json.loads(str(url.read(), 'utf-8'))

        user = User.objects.filter(id=request.session['id'])[0]
        telegram_msg = 'Title: ' + feedback.title + '\nText: ' + feedback.text +\
            '\nMail: ' + user.mail  + '\nFrom: ' + str(user) + '\nIP: ' + client_ip

        if geo_info["country_name"] != None and geo_info["city"] != None:     
            telegram_msg += '\nCity: ' + geo_info['country_name'] + ' ' + geo_info['city']

        telegram_send(settings.FeedbackTelegramChannelToken, settings.FeedbackTelegramChatId, telegram_msg)

    return buildJSONResponse({"message": "", "success": True})

def changeAccount(request):
    if "id" not in request.session:
        return HttpResponseRedirect("/")
    print(request.POST)
    return HttpResponse("OK")

def uploadAva(request):
    if "id" not in request.session:
        return HttpResponseRedirect("/")
    if "file" not in request.FILES:
        return buildJSONResponse({"ok": False, "message": "Неправильный запрос"})
    
    id = request.session['id']
    extUser = ExtendedUser.objects.get(userID=id)

    extUser.ava = request.FILES["file"]
    extUser.save()

    return buildJSONResponse({"ok": True})

def statistics(request):
    if "id" not in request.session:
        return HttpResponseRedirect("/")
    return render(request, "Account/statistics.html")

def searchResearch(request):
    if "id" not in request.session:
        return HttpResponseRedirect("/")
    if "word" not in request.POST:
        return HttpResponse("invalid request")
    word = "".join(request.POST["word"].split())
    if word == "":
        res = SeriesInfo.objects.all()
    else:
        res = SeriesInfo.objects.filter(SeriesInstanceUID__icontains=word)
    res = serializers.serialize('json', res)
    return buildJSONResponse({"ok": True, "res": res})

def uploadResearch(request):
    if "id" not in request.session:
        return HttpResponseRedirect("/")
    if "file" not in request.FILES:
        return buildJSONResponse({"ok": False, "message": "Неправильный запрос"})
    
    file = request.FILES["file"]
    if (file.name.split('.')[-1] != "zip"):
        return buildJSONResponse({"ok": False, "message": "Файл должен иметь расширение .zip"})

    fileName = default_storage.save("zips/"+file.name, file).split("/")[-1]

    return buildJSONResponse({"ok": True, "filename": fileName})
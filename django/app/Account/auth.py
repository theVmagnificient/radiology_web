from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import User, HashPassword, News
from Frontend import settings
from Frontend.TelegramBot import send as telegram_send
import json, urllib

def buildJSONResponse(responseData):
	return HttpResponse(json.dumps(responseData), content_type="application/json")

def auth(request):
	if 'id' in request.session:
		user = Users.objects.get(id=request.session["id"])
		return HttpResponseRedirect('/home/')
	return render(request, 'Authorize/auth.html')

def landpage(request):
	news = News.objects.all()
	return render(request, "Authorize/landpage.html", {"news": news})

def feedback(request):
	if 'name' not in request.POST or 'text' not in request.POST or 'mail' not in request.POST :
		return buildJSONResponse({"message": 'Error: invalid request data', "ok": False})
	if settings.TELEGRAM_FEEDBACK:
		client_ip = str(request.META['REMOTE_ADDR'])
		geo_ip_key = '7b9395a73758350d433f400a27280e69'

		geo_url = 'http://api.ipstack.com/{}?access_key={}'.format(client_ip, geo_ip_key)

		with urllib.request.urlopen(geo_url) as url:
			geo_info = json.loads(str(url.read(), 'utf-8'))

		telegram_msg = '\nText: ' + request.POST['text'] +\
			'\nMail: ' + request.POST['mail']  + '\nFrom: ' + request.POST['name'] + '\nIP: ' + client_ip

		if geo_info["country_name"] != None and geo_info["city"] != None:     
			telegram_msg += '\nCity: ' + geo_info['country_name'] + ' ' + geo_info['city']

		# telegram_send(settings.FeedbackTelegramChannelToken, settings.FeedbackTelegramChatId, telegram_msg)
	return buildJSONResponse({"message": "", "success": True})

def login(request):
	response = {"success": False, }
	if 'login' not in request.POST or 'passwd' not in request.POST:
		response["message"] = "Неправильный POST запрос"
		return buildJSONResponse(response)
	login, passwd = request.POST['login'], request.POST['passwd']
	if User.objects.filter(login=login):
		user = User.objects.filter(login=login)[0]
		if user.passwd == HashPassword(passwd):
			response['redirect'] = '/home'
			response['success'] = True
			request.session['id'] = user.id
	response['message'] = 'Неправильный логин или пароль!'
	return buildJSONResponse(response)

def deAuth(request):
	if 'id' in request.session:
		del request.session['id']
	return HttpResponseRedirect('/')

def error_404(request, exception):
	return render(request, 'ServerError/index.html', {'error_type': 404, 'error_header': 'Page not found!'})

def error_500(request):
	return render(request, 'ServerError/index.html', {'error_type': 500, 'error_header': 'Internal server error'})

def check_error_page(request):
	return render(request, 'ServerError/index.html', {'error_type': 500, 'error_header': 'Internal server error'})

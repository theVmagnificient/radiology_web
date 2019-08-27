from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from .models import User, ExtendedUser

def viewProfile(request, login):
	if "id" not in request.session:
		return HttpResponseRedirect("/")
	user = User.objects.get(id=request.session["id"])
	# print("USER ID: ", user.id)
	exUser = ExtendedUser.objects.get(userID=user.id)
	
	if User.objects.filter(login=login):
		profileUser = User.objects.get(login=login)
		profileExUser = ExtendedUser.objects.get(userID=profileUser.id)
		return render(request, "Profile/profile.html", {"user": user, "extUser": exUser, 
			"profileUser": profileUser, "profileExUser": profileExUser})
	return HttpResponse("<h1>User with login %s does not exists</h1>" % login)
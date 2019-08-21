from django.http import HttpResponseRedirect

def redirect(request):
	if 'id' in request.session:
		return HttpResponseRedirect('/home')
	return HttpResponseRedirect('/landpage')
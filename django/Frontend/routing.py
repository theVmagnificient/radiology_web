from django.urls import path
from django.conf.urls import url
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator, OriginValidator

# from Slicer.consumers import UploadResearchConsumer, UploadPredictionMask
from Account.consumers import ChatConsumer

application = ProtocolTypeRouter({
	# http->django views is added by default
	"websocket": AllowedHostsOriginValidator(
		AuthMiddlewareStack(
			URLRouter(
				[
					# path("uploadResearch/", UploadResearchConsumer),
					# path("uploadPredictionMask/", UploadPredictionMask),
					url(r"^chat/(?P<username>[\w]+)/$", ChatConsumer),
				]
			)
		)
	)
})
# from django.contrib.auth import get_
from channels.consumer import AsyncConsumer
from channels.db import database_sync_to_async
from asgiref.sync import async_to_sync
from django.db.models import Q
from django.core import serializers
import asyncio, json
from .models import User, Chat

class ChatConsumer(AsyncConsumer):
	async def websocket_connect(self, event):
		print("connected", event)

		userLogin = self.scope["url_route"]["kwargs"]["username"]
		myID = self.scope["session"]["id"]

		user = await self.getUser(login=userLogin)
		chatName = self.getChatName(user.id, myID)

		# async_to_sync(self.channel_layer.group_add)(chatName, self.channel_name)
		await self.channel_layer.group_add(chatName, self.channel_name)
		await self.send({
			"type": "websocket.accept"
		})

		chat = await self.getChatMessages(myID, user.id)
		await self.send({
			"type": "websocket.send",
			"text": chat,
		})

	async def websocket_receive(self, event):
		# print("receive", event)
		text = event.get("text", None)
		if text is not None:
			data = json.loads(text)
			msg = data.get("message")

			myID = self.scope["session"]["id"]
			userLogin = self.scope["url_route"]["kwargs"]["username"]

			user = await self.getUser(login=userLogin)
			me = await self.getUser(id=myID)
			
			insertedMessage = await self.createChatMessage(me.id, user.id, msg)

			chatName = self.getChatName(myID, user.id)
			await self.channel_layer.group_send(
				chatName,
				{
					"type": "chat_message",
					"response": serializers.serialize("json", [insertedMessage,])[1:-1],
				},
			)

	async def chat_message(self, event):
		await self.send({
			"type": "websocket.send",
			"text": event["response"],
		})

	async def websocket_disconnect(self, event):
		userLogin = self.scope["url_route"]["kwargs"]["username"]
		myID = self.scope["session"]["id"]

		user = await self.getUser(login=userLogin)
		chatName = self.getChatName(user.id, myID)
		await self.channel_layer.group_discard(chatName, self.channel_name)

	@database_sync_to_async
	def getUser(self, id=0, login=""):
		if id != 0:
			if User.objects.filter(id=id):
				return User.objects.get(id=id)
			return None
		if login != "":
			if User.objects.filter(login=login):
				return User.objects.get(login=login)
			return None

	@database_sync_to_async
	def getChatMessages(self, user1, user2):
		messages = Chat.objects.filter(Q(sender=user1, to=user2) | Q(sender=user2, to=user1))
		json_messages = serializers.serialize("json", messages)
		return json_messages

	@database_sync_to_async
	def createChatMessage(self, sender, to, msg):
		Chat.objects.create(sender=sender, to=to, message=msg)
		return Chat.objects.latest("id")

	def getChatName(self, user1, user2):
		return "{}-{}".format(min(user1, user2), max(user1, user2))

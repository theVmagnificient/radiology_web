# from django.contrib.auth import get_
from channels.consumer import AsyncConsumer
from multiprocessing import Process, Value
from channels.db import database_sync_to_async
from .slicer import SliceResearch, AddPredictionMask
from .models import ImageSeries, PredictionMask, SeriesInfo
import asyncio, json, time

SlicerProcesses = {}
PredictionMaskProcesses = {}

class UploadResearchConsumer(AsyncConsumer):
	async def websocket_connect(self, event):
		await self.send({
			"type": "websocket.accept"
		})

	async def websocket_receive(self, event):
		text = event.get("text", None)
		if text is not None:
			data = json.loads(text)
			filename = data.get("filename", None)

			myID = self.scope["session"]["id"]

			if filename == None:
				return
			if myID not in SlicerProcesses.keys():
				progress = Value('d', 0.0)
				status = Value('i', 1)
				process = Process(target=SliceResearch, args=(filename, status, progress))
				process.start()

				SlicerProcesses[myID] = {
					"process": process,
					"status": status, 
					"progress": progress,
				}

			if SlicerProcesses[myID]["process"].is_alive():
				status = SlicerProcesses[myID]["status"].value
			
				response = {
					"progress": SlicerProcesses[myID]["progress"].value,
					"status": status,
				}

				await self.send({
					"type": "websocket.send",
					"text": json.dumps(response),
				}) 
			else:
				response = {
					"progress": 0,
					"status": 5,
				}

				await self.send({
					"type": "websocket.send",
					"text": json.dumps(response),
				}) 

				del SlicerProcesses[myID]


	async def websocket_disconnect(self, event):
		print("disconnected", event)


class UploadPredictionMask(AsyncConsumer):
	async def websocket_connect(self, event):
		await self.send({
			"type": "websocket.accept"
		}) 

	async def websocket_receive(self, event):
		text = event.get("text", None)
		if text is not None:
			data = json.loads(text)

			myID = self.scope["session"]["id"]
			maskID = data["maskID"]
			mask = await self.getMask(maskID)
			researchID = mask.seriesID
			research = await self.getResearch(researchID)
			resInfo = await self.getResearchInfo(researchID)

			if myID not in PredictionMaskProcesses.keys():
				progress = Value('d', 0.0)
				status = Value('i', 1)
				process = Process(target=AddPredictionMask, args=(research.zipFileName,
					resInfo, mask, status, progress))
				process.start()
				PredictionMaskProcesses[myID] = {
					"process": process,
					"status": status, 
					"progress": progress,
				}

			status = PredictionMaskProcesses[myID]["status"].value
			if PredictionMaskProcesses[myID]["process"].is_alive():			
				response = {
					"progress": PredictionMaskProcesses[myID]["progress"].value,
					"status": status,
				}

				await self.send({
					"type": "websocket.send",
					"text": json.dumps(response),
				}) 
			else:
				if status == 3: 
					response = {
						"progress": 0,
						"status": 5,
					}					
				else:
					response = {
						"progress": 0,
						"status": 235,
					}
				await self.send({
						"type": "websocket.send",
						"text": json.dumps(response),
				}) 
				del PredictionMaskProcesses[myID]

	async def websocket_disconnect(self, event):
		print("disconnected", event)

	@database_sync_to_async
	def getMask(self, id):
		if PredictionMask.objects.filter(id=id):
			return PredictionMask.objects.get(id=id)
		return None

	@database_sync_to_async
	def getResearch(self, id):
		if ImageSeries.objects.filter(id=id):
			return ImageSeries.objects.get(id=id)
		return None

	@database_sync_to_async
	def getResearchInfo(self, id):
		if SeriesInfo.objects.filter(seriesID=id):
			return SeriesInfo.objects.get(seriesID=id)
		return None

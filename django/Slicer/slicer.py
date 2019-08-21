from Frontend import settings
import time, os, zipfile, random, string, tempfile, gdcm, json, io
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pydicom as dicom
from .models import ImageSeries
import pandas as pd

COLOR_MAPS = ("COLORMAP_BONE", "COLORMAP_JET", "COLORMAP_HOT")
cmaps = {"bone": plt.cm.bone, "gray": plt.cm.gray}

def GetCurrentPredictFromCSV(fileBytes, sourceID):
	df = pd.read_csv(io.BytesIO(fileBytes), encoding='utf8')
	if "source_id" not in df.columns:
		return {"ok": False, "err": "Колонка 'source_id' отсутствует в файле"}
	res = df[df.source_id == sourceID]
	if res.shape[0] == 0:
		return {"ok": False, "err": "В данном файле отсутствуют данные о исследовании %s" % sourceID}
	return {"ok": True, "res": res} 


def dicom_dataset_to_dict(dicom_header):
	dicom_dict = {}
	repr(dicom_header)
	for dicom_value in dicom_header.values():
		if dicom_value.tag == (0x7fe0, 0x0010):
			# discard pixel data
			continue
		if type(dicom_value.value) == dicom.dataset.Dataset:
			dicom_dict[dicom_value.keyword] = dicom_dataset_to_dict(dicom_value.value)
		else:
			v = _convert_value(dicom_value.value)
			if dicom_value.keyword == "SeriesInstanceUID":
				v = v[1:-1]
			dicom_dict[dicom_value.keyword] = v
	return dicom_dict

def _sanitise_unicode(s):
	return s.replace(u"\u0000", "").strip()

def _convert_value(v):
	t = type(v)
	if t in (list, int, float):
		cv = v
	elif t == str:
		cv = _sanitise_unicode(v)
	elif t == bytes:
		s = v.decode('ascii', 'replace')
		cv = _sanitise_unicode(s)
	elif t == dicom.valuerep.DSfloat:
		cv = float(v)
	elif t == dicom.valuerep.IS:
		cv = int(v)
	elif t == dicom.valuerep.PersonName3:
		cv = str(v)
	else:
		cv = repr(v)
	return cv

def getDicomListFromZip(zipFileName):
	zipFilePath = "{}/zips/{}".format(settings.MEDIA_ROOT, zipFileName)	

	datasets = list()
	with zipfile.ZipFile(zipFilePath) as researchZip:
		for entry in researchZip.namelist():
			if entry.endswith('/'):
				continue  # skip directories
			entryPseudoFile = researchZip.open(entry)

			dicomTempFile = tempfile.NamedTemporaryFile()
			dicomTempFile.write(entryPseudoFile.read())
			dicomTempFile.flush()
			dicomTempFile.seek(0)

			try:
				dataset = dicom.read_file(dicomTempFile)
				dataset.pixel_array
				datasets.append(dataset)
			except dicom.errors.InvalidDicomError as e:
				print(e)
				pass

	datasets.sort(key=lambda x: x.ImagePositionPatient[2])
	return datasets

def SliceResearch(filename, status, progress):
	status.value = 1
	
	datasets = getDicomListFromZip(filename)
	datasetsCnt = len(datasets)
	if datasetsCnt == 0:
		status.value = 124
		return

	dicom_dict = dicom_dataset_to_dict(datasets[0])
	dicom_dict["slicesCnt"] = len(datasets)
	dicom_dict_json = json.dumps(dicom_dict)
	status.value = 2

	slicesDir = ''.join(random.choice(string.ascii_lowercase) for x in range(10))

	imageDirPath = "{}/images/{}".format(settings.MEDIA_ROOT, slicesDir)
	while os.path.exists(imageDirPath):
		imageDirPath += random.choice(string.ascii_lowercase)

	os.makedirs(imageDirPath)

	for n, ds in enumerate(datasets):
		image = str(n) + "_{}.png"
		for cmName, cm in cmaps.items():
			fig = plt.figure(frameon=False, dpi=300)
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)
			ax.imshow(ds.pixel_array, cmap=cm)     
			plt.savefig(os.path.join(imageDirPath, image.format(cmName)))
			plt.close()
		progress.value = (n * 100) / datasetsCnt

	status.value = 3
	ImageSeries.objects.create(dcm_meta=dicom_dict_json, slices_dir=slicesDir, zipFileName=filename)


def AddPredictionMask(zipFileName, seriesInfo, mask, status, progress):
	status.value = 1

	datasets = getDicomListFromZip(zipFileName)
	datasetsCnt = len(datasets)
	if datasetsCnt == 0:
		status.value = 124
		return

	status.value = 2

	file_csv = open(settings.MEDIA_ROOT + "/masks_model/" + mask.fileName, "rb").read()
	df = GetCurrentPredictFromCSV(file_csv, seriesInfo.source_id)["res"]
	
	imageDirPath = "{}/images/{}/{}".format(settings.MEDIA_ROOT, seriesInfo.slices_dir,
		mask.maskFolder)
	for n, ds in enumerate(datasets):
		image = str(n) + "_{}.png"
		for cmName, cm in cmaps.items():
			fig = plt.figure(frameon=False, dpi=300)
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)

			ax.set_aspect('equal')
			ax.imshow(ds.pixel_array, cmap=cm)

			for i, row in df.iterrows():
				ellipse = Ellipse(xy=(row.locX, row.locY), width=row.diamX, height=row.diamY,
				 edgecolor="r", lw=2, fill=False)
				ax.add_patch(ellipse)

			plt.savefig(os.path.join(imageDirPath, image.format(cmName)))
			plt.close()
		progress.value = (n * 100) / datasetsCnt

	status.value = 3
	

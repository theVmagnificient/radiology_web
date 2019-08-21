from django.conf import settings
from django.db import models
from Slicer.dicom_import import dicom_datasets_from_zip
from Slicer.dicom_export import export_to_png
import pydicom as dicom
import json, shutil
#from django.conf.urls.static import static
#from django.core.files.base import ContentFile
#import numpy as np

import os, shutil, zipfile, datetime, hashlib

class SeriesInfo(models.Model):
    doctorComment = models.CharField(max_length=256, default="")
    doctorCommentDate = models.DateTimeField(default=datetime.date.today, null=True, blank=True)
    seriesID = models.IntegerField()
    slicesCnt = models.IntegerField(null=True, blank=True)

    AccessionNumber = models.CharField(max_length=128)
    AcquisitionDate = models.CharField(max_length=128)
    FilterType = models.CharField(max_length=128)
    PatientID = models.CharField(max_length=128)
    PatientAge = models.CharField(max_length=128)
    PatientBirthDate = models.CharField(max_length=128)
    PatientPosition = models.CharField(max_length=128)
    StudyID = models.CharField(max_length=128)
    PatientSex = models.CharField(max_length=128)
    ScanOptions = models.CharField(max_length=128)
    SeriesDate = models.CharField(max_length=128)
    SeriesDescription = models.CharField(max_length=128)
    SeriesTime = models.CharField(max_length=128)
    SoftwareVersions = models.CharField(max_length=128)
    StationName = models.CharField(max_length=128)
    StudyDate = models.CharField(max_length=128)
    StudyStatusID = models.CharField(max_length=128)
    SeriesInstanceUID = models.CharField(max_length=128)
    StudyTime = models.CharField(max_length=128)
    Manufacturer = models.CharField(max_length=128)

    previewSlice = models.CharField(max_length=32, default="0_gray.png")
    slices_dir = models.CharField(max_length=128)

    @property
    def source_id(self):
        return "+{}_{}".format(self.AccessionNumber, self.StudyID)
    

class ImageSeries(models.Model):
    patient_id = models.CharField(max_length=64, null=True)
    study_uid = models.CharField(max_length=64)
    series_uid = models.CharField(max_length=64)
    slices_dir = models.CharField(max_length=64)
    dcm_meta = models.CharField(max_length=64)
    zipFileName = models.CharField(max_length=128)
    
    @property
    def images_path(self):
        return settings.MEDIA_ROOT + "/images/" + self.slices_dir

    def getMetaDict(self):
        return json.loads(self.dcm_meta)

    def save(self, *args, **kwargs):
        meta = self.getMetaDict()
        self.patient_id = meta["PatientID"]
        self.study_uid = meta["StudyInstanceUID"]
        self.series_uid = meta["SeriesInstanceUID"]

        super(ImageSeries, self).save(*args, **kwargs)

        si = SeriesInfo.objects.create(seriesID=self.id)
        si.slices_dir = self.slices_dir
        si.slicesCnt = meta["slicesCnt"]

        for att in meta.keys():
            try:
                setattr( si, att, meta[att] )
            except:
                pass
        si.save()

        
    def delete(self, *args, **kwargs):
        try:
           # Delete the images folder as well
            shutil.rmtree(settings.MEDIA_ROOT + "/images/" + self.slices_dir)
        except:
            pass
        super(ImageSeries, self).delete(*args, **kwargs)
        
    class Meta:
        verbose_name_plural = 'Image Series'

class PredictionMask(models.Model):
    seriesID = models.IntegerField()
    maskName = models.CharField(max_length=64)
    fileName = models.CharField(max_length=64)
    maskDescription = models.CharField(max_length=128)
    maskFolder = models.CharField(max_length=64)
    processProgress = models.IntegerField(default=0)

    def __str__(self):
        return self.maskName
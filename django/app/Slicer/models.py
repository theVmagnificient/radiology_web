from django.db import models
from datetime import datetime
from django.conf import settings
import json
import os

class Research(models.Model):
    study_date = models.DateField(auto_now=True)
    study_time = models.TimeField(auto_now=True)
    patient_id = models.CharField(max_length=128)
    series_instance_uid = models.CharField(max_length=128)
    series_number = models.CharField(max_length=128)
    dicom_tags = models.TextField()
    dicom_names = models.TextField()
    dir_name = models.CharField(max_length=256)
    zip_name = models.CharField(max_length=256)
    preview_image = models.CharField(max_length=128)
    predictions_dir = models.CharField(max_length=128)

    @property
    def dicom_names_list(self):
        return json.loads(self.dicom_names)

    @property
    def zip_path(self):
        return os.path.join(settings.BASE_DIR, "research_storage", "zips", self.zip_name)

    @property
    def dir_path(self):
        return os.path.join(settings.BASE_DIR, "research_storage", self.zip_name)

    def __str__(self):
        return self.series_instance_uid 

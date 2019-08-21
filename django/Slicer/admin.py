from django.contrib import admin

from .models import ImageSeries, SeriesInfo, PredictionMask

admin.site.register(PredictionMask)

@admin.register(SeriesInfo)
class SeriesInfoAdmin(admin.ModelAdmin):
    readonly_fields = ('doctorComment', 'doctorCommentDate', 'seriesID', 'AccessionNumber',
        'AcquisitionDate', 'FilterType', 'PatientAge', 'PatientBirthDate',
        'PatientPosition', 'PatientSex', 'ScanOptions',
        'SeriesDate', 'SeriesDescription', 'SeriesTime', 'SoftwareVersions',
        'StationName', 'StudyDate', 'StudyStatusID', 'StudyTime', 'Manufacturer')

@admin.register(ImageSeries)
class ImageSeriesAdmin(admin.ModelAdmin):
    readonly_fields = ('patient_id', 'study_uid', 'series_uid')

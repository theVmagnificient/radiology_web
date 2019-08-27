from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import Research
import matplotlib.pyplot as plt
from matplotlib import cm
import pydicom as dicom
import zipfile
import json
import os

MAX_RESEARCH_SIZE = 1000 # in megabytes

def zip_validation(research):
    if research.name.split(".")[-1] != "zip":
        return {"ok": False, "error": "Zip file required"}
    if research.size > 1024 * 1024 * MAX_RESEARCH_SIZE:
        return {"ok": False, "error": "File is too large"}
    return {"ok": True}

def extract_zip(zip_path):
    extract_dir = os.path.join(settings.BASE_DIR, "research_storage", "".join(zip_path.split("/")[-1].split(".")[:-1]))

    if not zipfile.is_zipfile(zip_path):
        return {"ok": False, "error": "Invalid zip file"}

    dicoms = 0
    filenames = list()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for fn in zip_ref.filelist:
            if fn.filename.split(".")[-1] != "dcm":
                return {"ok": False, "error": "Expected archive with .dicom files"}
            filenames.append(fn.filename)
            dicoms += 1
        if dicoms == 0:
            return {"ok": False, "error": "No dicom files in root of archive"}
        zip_ref.extractall(extract_dir)
        
    return {"ok": True, "extract_dir":  extract_dir, "filenames": filenames}

def export_to_png(dcm, path):
    fig = plt.figure(frameon=False, dpi=300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(dcm.pixel_array, cmap=plt.cm.gray)
    plt.savefig(path)
    plt.close()

def process(params):
    dcm = dicom.dcmread(os.path.join(params["extract_dir"], os.listdir(params["extract_dir"])[0]))
    export_to_png(dcm, os.path.join(params["extract_dir"], "preview.png"))
    research = Research.objects.create(study_date=dcm.StudyDate, study_time=dcm.StudyTime, 
                    patient_id=dcm.PatientID, series_instance_uid=dcm.SeriesInstanceUID,
                    series_number=dcm.SeriesNumber, dir_name=os.path.basename(params["extract_dir"]),
                    zip_name=params["zip_name"], dicom_names=json.dumps(params["filenames"]))
    
    research.save()

def handle_research(research):
    fs = FileSystemStorage()
    valid = zip_validation(research)

    if not valid["ok"]:
        return valid

    zip_path = os.path.join(settings.BASE_DIR, "research_storage", "zips", research.name)
    zip_path = fs.save(zip_path, research)

    resp = extract_zip(zip_path)
    if not resp["ok"]:
        return resp
    resp["zip_name"] = os.path.basename(zip_path)

    process(resp)

    return {"ok": True}
    
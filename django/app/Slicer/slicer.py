from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import Research
import matplotlib.pyplot as plt
from .kafka_client import KafkaProducer
from matplotlib import cm
import pydicom as dicom
import zipfile
import tempfile
import json
import os

MAX_RESEARCH_SIZE = 1000 # in megabytes
KAFKA_PRODUCER = KafkaProducer("/app/Slicer/avro_sch/res_prod.json", os.environ.get('KAFKA_BROKER_URL'), os.environ.get('PRODUCER_TOPIC'),
        'http://schema_registry:8081')


def zip_validation(research):
    if research.name.split(".")[-1] != "zip":
        return {"ok": False, "error": "Zip file required"}
    if research.size > 1024 * 1024 * MAX_RESEARCH_SIZE:
        return {"ok": False, "error": "File is too large"}
    return {"ok": True}

def extract_zip(zip_path):
    extract_dir = os.path.join(settings.BASE_DIR, "static", "research_storage", "dicoms", "".join(zip_path.split("/")[-1].split(".")[:-1]))

    if not zipfile.is_zipfile(zip_path):
        return {"ok": False, "error": "Invalid zip file"}

    filenames = list()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for fn in zip_ref.namelist():
            entry_pseudo_file = zip_ref.open(fn)

            # the pseudo file does not support `seek`, which is required by
            # dicom's lazy loading mechanism; use temporary files to get around this;
            # relies on the temporary files not being removed until the temp
            # file is garbage collected, which should be the case because the
            # dicom datasets should retain a reference to the temp file
            temp_file = tempfile.NamedTemporaryFile()
            temp_file.write(entry_pseudo_file.read())
            temp_file.flush()
            temp_file.seek(0)
            
            try:
                filenames.append((fn, dicom.read_file(temp_file)))
            except dicom.errors.InvalidDicomError as e:
                print("Skipping invalid dicom file...")

                
        if len(filenames) == 0:
            return {"ok": False, "error": "No dicom files in root of archive"}
        zip_ref.extractall(extract_dir)

    filenames.sort(key=lambda x: int(x[1].ImagePositionPatient[2]))

    return {"ok": True, "extract_dir":  extract_dir, "filenames": [el[0] for el in filenames]}

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
    return research

def call_prediction(research_db):
    kafka_msg = {
        "command": "start",
        "id": str(research_db.id),
        "path": research_db.zip_name
    }
    KAFKA_PRODUCER.produce_msg(kafka_msg)
    print("Message produced!")

def handle_research(research):
    fs = FileSystemStorage()
    valid = zip_validation(research)

    if not valid["ok"]:
        return valid

    zip_path = os.path.join(settings.BASE_DIR, "static", "research_storage", "zips", research.name)
    zip_path = fs.save(zip_path, research)

    resp = extract_zip(zip_path)
    if not resp["ok"]:
        return resp
    resp["zip_name"] = os.path.basename(zip_path)

    research_db = process(resp)
    call_prediction(research_db)

    return {"ok": True}


from django.test import TestCase

from .models import Research
from .slicer import extract_zip, process, call_prediction
from confluent_kafka import avro
from confluent_kafka import KafkaError
from confluent_kafka.avro import AvroConsumer
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.avro import AvroProducer

from shutil import copyfile
from django.conf import settings
import time
import os

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')

class UploadResearchTest(TestCase):
    research_instance_uid = "1.3.12.2.1107.5.1.4.74203.30000018121306150842600002074"
#    test_prediction_nods = {"command": "start", "path": "research123.zip", "id": "1"} # TODO: change!
    test_prediction_nods = {"code": "success", "path": "none", "id": "django_test", "nods": []}

    def setUp(self):
        zip_path = os.path.join(settings.BASE_DIR, "static", "research_storage", "zips", "research.zip")
        copyfile(os.path.join(settings.BASE_DIR, "tests/research123.zip"), zip_path) 
        print("Zipfile copied!")

        try:
            resp = extract_zip(zip_path)
            resp["zip_name"] = os.path.basename(zip_path)

            self.dir_path = resp["extract_dir"]

            research_db = process(resp)
            call_prediction(research_db)
        except Exception as ex:
            self.assert_(True, str(ex))
    
    def test_research_loaded(self):
        self.assertEqual(os.path.exists(self.dir_path), True, f"Directory {self.dir_path}")
        self.assertEqual(len(os.listdir(self.dir_path)) > 0, True, f"Dicom directory is empty!")

    def test_database_updated(self):
        res = Research.objects.filter(series_instance_uid=self.research_instance_uid)
        self.assertEqual(res.count() != 0, True, "Research doesn't signed up in database")
        return res[0]
    
    def test_django_consumer(self):
        value_schema = avro.load('/app/Slicer/avro_sch/dnn_res_prod.json')
        avroProducer = AvroProducer({
            'bootstrap.servers': KAFKA_BROKER_URL,
            'schema.registry.url': 'http://schema_registry:8081'
        }, default_value_schema=value_schema)

        avroProducer.produce(topic='dnn.results', value=self.test_prediction_nods)
        print("msg produced")
        print("Waiting for consumer`s answer")

        res = self.test_database_updated()
        self.assertEqual(res.predictions_nods, self.test_prediction_nods)

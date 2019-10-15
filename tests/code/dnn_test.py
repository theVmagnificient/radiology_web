import pytest

from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
from confluent_kafka import KafkaError
from confluent_kafka.avro import AvroConsumer
from confluent_kafka.avro.serializer import SerializerError

import os
import time
import glob
from shutil import copyfile

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')

def setup_module(module):
    #init_something()
    pass

def teardown_module(module):
    #teardown_something()
    pass

def test_dnn_success():
    copyfile('research.zip', '/mnt/archives/research.zip') 
    value_schema = avro.load('avro_sch/res_prod.json')
    value = {"command": "start", "path": "research.zip", "id": "test_dnn_success"}

    avroProducer = AvroProducer({
           'bootstrap.servers': KAFKA_BROKER_URL,
           'schema.registry.url': 'http://schema_registry:8081'
          }, default_value_schema=value_schema)

    avroProducer.produce(topic='dnn.data', value=value)

    c = AvroConsumer({
    'bootstrap.servers': "broker:9092",
    'group.id': 'groupid',
    'schema.registry.url': 'http://schema_registry:8081'})

    c.subscribe(["dnn.results"])

    while True:
        try:
            print("Start polling")
            msg = c.poll(10)

        except SerializerError as e:
            print("Message deserialization failed for {}: {}".format(msg, e))
            break

        if msg is None:
            continue
        if msg.error():
            print("AvroConsumer error: {}".format(msg.error()))
            continue
        msg = msg.value()
        
        if msg["id"] != "test_dnn_success":
            print("Skipping msg with id", msg["id"])
            continue

        assert msg["code"] == "success", "Inference failed" 

        assert type(msg) == dict, "Wrong type of msg variable"

        assert 'path' in msg, "No path field in returned message"

        for res in os.listdir(os.path.join("/mnt/results/experiments", msg["path"])):
            if os.path.isdir(os.path.join("/mnt/results/experiments", msg["path"], res)):
                png_files = glob.glob(os.path.join("/mnt/results/experiments", msg["path"], res, "*.png"))
                assert len(png_files) > 0, "No png output files found for {}".format(res)

        assert len(msg["nods"]) < 5, "Too many nodules found"
        break
    c.close()


def test_dnn_broken_msg():
    value_schema = avro.load('avro_sch/res_prod.json')
    value = {"command": "start", "path": "research8913enadkjnasdnksajdn", "id": "test_dnn_fail"}

    avroProducer = AvroProducer({
           'bootstrap.servers': KAFKA_BROKER_URL,
           'schema.registry.url': 'http://schema_registry:8081'
          }, default_value_schema=value_schema)

    avroProducer.produce(topic='dnn.data', value=value)
    c = AvroConsumer({
    'bootstrap.servers': "broker:9092",
    'group.id': 'groupid',
    'schema.registry.url': 'http://schema_registry:8081'})

    c.subscribe(["dnn.results"])

    while True:
        try:
            print("Start polling")
            msg = c.poll(10)

        except SerializerError as e:
            print("Message deserialization failed for {}: {}".format(msg, e))
            break

        if msg is None:
            continue

        if msg.error():
            print("AvroConsumer error: {}".format(msg.error()))
            continue
        msg = msg.value()
 
        if msg["id"] != "test_dnn_fail":
            print("Skipping msg with id ", msg["id"])
            continue
        
        assert msg["code"] == "failed", "Inference failed"
        break
    c.close()

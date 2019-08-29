from confluent_kafka import KafkaError
from confluent_kafka.avro import AvroConsumer
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.avro import AvroProducer
from confluent_kafka.avro.serializer import SerializerError
from confluent_kafka import avro

import os
import json


class KafkaConsumer:

    def __init__(self, broker_url=os.environ.get('KAFKA_BROKER_URL'),
                  topic=os.environ.get('TRANSACTIONS_TOPIC'), group='dnn_aimed_group',
                  schema_registry_url='http://schema_registry:8081'):
        self.c = AvroConsumer({
           'bootstrap.servers': broker_url,
           'group.id': group,
           'auto.offset.reset': 'earliest',
           'enable.auto.offset.store': False,
           'enable.auto.commit': False,
           'schema.registry.url': schema_registry_url,
           'max.poll.interval.ms': '3600000' # one hour
        })

        self.c.subscribe([topic])

    def get_next_msg(self):
        while 1:
            try:
                msg = self.c.poll(10)

            except SerializerError as e:
                print("Message deserialization failed for {}: {}".format(msg, e))
                break

            if msg is None:
                continue

            if msg.error():
                print("AvroConsumer error: {}".format(msg.error()))
                continue

            return dict(msg.value()), msg

    def __del__(self):
        self.c.close()

class KafkaProducer:

    def __init__(self, path_to_scheme, broker_url=os.environ.get('KAFKA_BROKER_URL'),
                  topic=os.environ.get('PRODUCER_TOPIC'), schema_registry_url='http://schema_registry:8081'):
        self.value_scheme = avro.load(path_to_scheme)

        self.p = AvroProducer({
            'bootstrap.servers': broker_url,
            'schema.registry.url': schema_registry_url
        }, default_value_schema=self.value_scheme)
        self.topic = topic

    def produce_msg(self, value):
        try:
            self.p.produce(topic=self.topic, value=value) #self.topic
        except Exception as e:
            print(str(e)) #TODO logging

    def __del__(self):
        self.p.flush()










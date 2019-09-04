from kafka_client import KafkaConsumer, KafkaProducer
from confluent_kafka import TopicPartition
import threading
import os
import time
import requests
import json
#os.environ['KAFKA_BROKER_URL'] = 'broker:9092'
#os.environ['TRANSACTIONS_TOPIC'] = 'dnn.results'
#os.environ['PRODUCER_TOPIC'] = 'dnn.data'
#os.environ['KAFKA_SCHEMA_REGISTRY_URL'] = 'http://schema_registry:8081'

class DjangoKafkaServer(threading.Thread):
    def __init__(self, broker_url=os.environ.get('KAFKA_BROKER_URL'),
                 topic=os.environ.get('TRANSACTIONS_TOPIC'), group='dnn_aimed_group',
                 schema_registry_url='http://schema_registry:8081'):
        super().__init__()

        self.daemon = True

        self.consumer = KafkaConsumer(broker_url, topic, group, schema_registry_url)


    def run(self) -> None:
        value = None
        while 1:
            msg, kafka_msg = self.consumer.get_next_msg()

            json_msg = json.dumps(msg)
            http_params = {"data": json_msg}
            print(http_params)
            #print("http://" + os.environ.get('SERVER_HOSTNAME') + ":" + os.environ.get("SERVER_PORT") + "/series/kafka_processed")
            r = requests.get(url="http://" + os.environ.get('SERVER_HOSTNAME') + ":" + os.environ.get("SERVER_PORT") + "/series/kafka_processed", params=http_params)
            response = r.text
            print("Server response: ", response)  

            self.consumer.c.commit(kafka_msg)
            exit(0)

if __name__ == '__main__':
    server = DjangoKafkaServer()

    server.start()

    # Ctrl+C waiting
    while threading.active_count() > 0:
        time.sleep(1)

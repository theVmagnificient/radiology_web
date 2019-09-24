from Slicer.kafka_client import KafkaConsumer, KafkaProducer
from confluent_kafka import TopicPartition
from .models import Research
import threading
import os
import time

#os.environ['KAFKA_BROKER_URL'] = 'broker:9092'
#os.environ['TRANSACTIONS_TOPIC'] = 'dnn.results'
#os.environ['PRODUCER_TOPIC'] = 'dnn.data'
#os.environ['KAFKA_SCHEMA_REGISTRY_URL'] = 'http://schema_registry:8081'

class DjangoKafkaServer(threading.Thread):
    def __init__(self, broker_url=os.environ.get('KAFKA_BROKER_URL'),
                 topic=os.environ.get('TRANSACTIONS_TOPIC'), producer_topic=os.environ.get('PRODUCER_TOPIC'),
                 group='dnn_aimed_group',
                 schema_registry_url='http://schema_registry:8081'):
        super().__init__()

        self.daemon = True

        self.consumer = KafkaConsumer(broker_url, topic, group, schema_registry_url)
        self.producer = KafkaProducer('avro_sch/res_prod.json', broker_url, producer_topic, schema_registry_url)

        self.volume_path = '/mnt/archives'

    def run(self) -> None:
        value = None
        while 1:
            msg, kafka_msg = self.consumer.get_next_msg()

            if msg["code"] == "success":
                path = msg["path"]
                research_id = msg["id"]

                research = Research.objects.filter(id=research_id)
                
                if research.count() != 1:
                    print("Invalid research id recieved from kafka!")
                    continue
                
                research = research[0]
                research.predictions_dir = path
                research.save()
            else:
                print("An error occured during the prediction!!!")

            self.consumer.c.commit(kafka_msg)
            exit(0)

if __name__ == '__main__':
    server = dnnServer()

    server.start()

    # Ctrl+C waiting
    while threading.active_count() > 0:
        time.sleep(1)

from kafka_client import KafkaConsumer, KafkaProducer
from confluent_kafka import TopicPartition
from exec import Executor
from config import Config
import threading
import os
import time
from utils import get_random_hash

os.environ['KAFKA_BROKER_URL'] = 'broker:9092'
os.environ['TRANSACTIONS_TOPIC'] = 'dnn.data'
os.environ['PRODUCER_TOPIC'] = 'dnn.results'
os.environ['KAFKA_SCHEMA_REGISTRY_URL'] = 'http://schema_registry:8081'

class dnnServer(threading.Thread):
    def __init__(self, broker_url=os.environ.get('KAFKA_BROKER_URL'),
                 topic=os.environ.get('TRANSACTIONS_TOPIC'), producer_topic=os.environ.get('PRODUCER_TOPIC'),
                 group='dnn_aimed_group',
                 schema_registry_url='http://schema_registry:8081'):
        super().__init__()

        self.daemon = True

        self.consumer = KafkaConsumer(broker_url, topic, group, schema_registry_url)
        self.producer = KafkaProducer('avro_sch/res_prod.json', broker_url, producer_topic, schema_registry_url)

        conf = Config('/mnt/results/experiments')
        self.executor = Executor(cf=conf)
        self.volume_path = '/mnt/archives'

    def run(self) -> None:
        value = None
        while 1:
            msg, kafka_msg = self.consumer.get_next_msg()

            if msg['command'] == 'start':
                try:
                    self.executor.pipe.cf.save_path = os.path.join(self.executor.pipe.cf.root_path_to_save,
                                                                    get_random_hash())
                    try:
                        print(f"Creating exp dir {self.executor.pipe.cf.save_path}")
                        os.mkdir(self.executor.pipe.cf.save_path)
                    except Exception as e:
                        print("You are very lucky | Creating new directory")
                        self.executor.pipe.cf.save_path = \
                            os.path.join(self.executor.pipe.cf.root_path_to_save,
                                                                       get_random_hash())
                        print(f"Creating exp dir {self.executor.pipe.cf.save_path}")
                        os.mkdir(self.executor.pipe.cf.save_path)

                    self.executor.unpack(os.path.join(self.volume_path, msg['path']),
                                         os.path.join(self.volume_path, msg['path'] + '_unpacked'))
                    self.executor.start()
                    value = {
                             "code": "success",
                             "path": self.executor.pipe.cf.save_path.split('/')[-1],
                             "id": msg["id"]
                             }
                except Exception as e:
                    print(str(e))
                    value = {"code": "failed", "path": "none", "id": "-1"}
                finally:
                    self.producer.produce_msg(value)


            self.consumer.c.commit(kafka_msg)
            exit(0)

if __name__ == '__main__':
    server = dnnServer()

    server.start()

    # Ctrl+C waiting
    while threading.active_count() > 0:
        time.sleep(1)





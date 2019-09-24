from confluent_kafka import Producer
from confluent_kafka import avro
import os 
import time

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')

p = Producer({'bootstrap.servers': KAFKA_BROKER_URL})

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

while 1:
     p.poll(0)
     data = 'test_msg'
     p.produce('dnn.metadata', data.encode('utf-8'), callback=delivery_report)
     time.sleep(2)


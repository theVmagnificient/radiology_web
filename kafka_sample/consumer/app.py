from confluent_kafka import Consumer, KafkaError
import os

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')
TRANSACTIONS_TOPIC = os.environ.get('TRANSACTIONS_TOPIC')

c = Consumer({
    'bootstrap.servers': KAFKA_BROKER_URL,
    'group.id': 'mygroup',
    'auto.offset.reset': 'earliest'
})

c.subscribe([TRANSACTIONS_TOPIC])

while True:
    msg = c.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        print("Consumer error: {}".format(msg.error()))
        continue

    print('Received message: {}'.format(msg.value().decode('utf-8')))

c.close()

from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
import os 
import time

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')


value_schema_str = """
{
   "namespace": "aimed",
   "name": "metadata",
   "type": "record",
   "fields" : [
     {
       "name" : "id",
       "type" : "string"
     },
     {
       "name" : "path",
       "type" : "string"
     },
     {
       "name" : "code",
       "type" : "string" 
     }
   ]
}
"""

value_schema = avro.loads(value_schema_str)
print("Loaded")
value = {"code": "success", "path": "out.zip", "id": "1"}

avroProducer = AvroProducer({
    'bootstrap.servers': KAFKA_BROKER_URL,
    'schema.registry.url': 'http://schema_registry:8081'
    }, default_value_schema=value_schema)

while 1:
    time.sleep(5)
    avroProducer.produce(topic='dnn.results', value=value)#, key=key)
    print("msg produced")
avroProducer.flush()

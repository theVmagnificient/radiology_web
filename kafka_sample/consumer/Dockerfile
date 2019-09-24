# Dockerfile
FROM python:3.6

WORKDIR /usr/app
RUN apt-get update && apt-get install -y librdkafka-dev

ADD ./requirements.txt ./
RUN pip3 install -r requirements.txt
RUN pip3 install confluent-kafka[avro]
ADD ./ ./

CMD ["python", "app_avro.py"]

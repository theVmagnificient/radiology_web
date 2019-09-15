# Dockerfile
FROM python:3.6

WORKDIR /app
RUN apt-get update && apt-get install -y librdkafka-dev

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt
RUN pip3 install confluent-kafka[avro]
COPY ./ ./

CMD ["python", "app_avro.py"]

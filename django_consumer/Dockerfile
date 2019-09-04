FROM python:3.7
MAINTAINER DIT

ENV PYTHONUNBUFFERED 1

RUN  apt-get update \
     && apt-get install -y --no-install-recommends gcc \
     && apt-get install -y --no-install-recommends librdkafka-dev \
     && pip install cryptography \
     && pip install confluent-kafka confluent-kafka[avro]

RUN mkdir /code
WORKDIR /code
COPY ./code /code

RUN pip install confluent-kafka 
RUN pip install confluent-kafka[avro]
RUN pip install requests

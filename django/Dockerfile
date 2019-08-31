FROM python:3.7
MAINTAINER DIT

ENV PYTHONUNBUFFERED 1

RUN  apt-get update \
     && apt-get install -y --no-install-recommends gcc \
     && pip install cryptography

COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt 

RUN mkdir /app
WORKDIR /app
COPY ./app /app

ENV DJANGO_ENV=prod

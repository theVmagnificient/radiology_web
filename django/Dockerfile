FROM python:3.6
MAINTAINER DIT

ENV PYTHONUNBUFFERED 1

RUN  apt-get update \
     && pip install --upgrade pip \
     && apt-get install -y --no-install-recommends gcc \
     && apt-get install -y --no-install-recommends librdkafka-dev \
     && pip install cryptography \
     && pip install confluent-kafka confluent-kafka[avro]

RUN echo "INSTALLING GDCM\n"

RUN git clone --branch master https://github.com/HealthplusAI/python3-gdcm.git && cd python3-gdcm && dpkg -i build_1-1_amd64.deb && apt-get install -f
RUN cp /usr/local/lib/gdcm.py /usr/local/lib/python3.6/site-packages/.
RUN cp /usr/local/lib/gdcmswig.py /usr/local/lib/python3.6/site-packages/.
RUN cp /usr/local/lib/_gdcmswig.so /usr/local/lib/python3.6/site-packages/.
RUN cp /usr/local/lib/libgdcm* /usr/local/lib/python3.6/site-packages/.
RUN ldconfig

RUN echo "INSTALLING requirements\n"

COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt 

RUN mkdir /app
WORKDIR /app
COPY ./app /app

RUN pip install confluent-kafka
RUN pip install confluent-kafka[avro]

ENV DJANGO_ENV=prod

#RUN echo "WORKING WITH MIGRATIONS\n"
#RUN ./manage.py makemigrations && ./manage.py migrate

#RUN echo "CREATING SUPER USER\n"
#RUN ./manage.py createsuperuser

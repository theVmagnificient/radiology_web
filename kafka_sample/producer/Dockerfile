# Dockerfile
FROM python:3.6

WORKDIR /usr/app
RUN apt-get update && apt-get install -y librdkafka-dev

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
ADD ./ ./

CMD ["python", "app.py"]

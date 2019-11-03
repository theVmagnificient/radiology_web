FROM python:3.7

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN chmod +x env.sh && ./env.sh
EXPOSE 5000
CMD ["/bin/bash", "-c", "python -m flask run --host=0.0.0.0"]

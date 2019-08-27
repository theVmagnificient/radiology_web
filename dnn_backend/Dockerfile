FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL Lopatin Kirill <kirill.v.lopatin@gmail.com>

RUN apt-get update && apt-get install -y software-properties-common build-essential cmake curl librdkafka-dev \
    liblapack-dev libatlas-base-dev libopenblas-dev \
    git wget 

WORKDIR /install

COPY ./environment.yml /install/

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN /bin/bash -c "conda create --name aimed python=3.7"
RUN /bin/bash -c "source activate aimed && \
    conda env update --name aimed --file environment.yml && \
    source deactivate"

COPY ./radio_dep /install/radio_dep

RUN echo "Installing radio..."

WORKDIR /install/radio_dep

RUN /bin/bash -c "source activate aimed && pip install . && pip install ./models && pip install confluent-kafka confluent-kafka[avro] && source deactivate"
COPY ./radio_dep/config.yml /miniconda/envs/aimed/lib/python3.6/site-packages/radio/config/

WORKDIR /app

COPY source /app/source 
COPY weights /app/weights 

ENTRYPOINT ["/bin/bash", "-c", "while :; do echo 'Hit CTRL+C'; sleep 1; done"]


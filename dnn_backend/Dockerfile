FROM thev1/dnn_libs:noentry
LABEL Lopatin Kirill <kirill.v.lopatin@gmail.com>

WORKDIR /app

COPY source /app/source 
COPY weights /app/weights 

ENV PATH /miniconda/bin:$PATH

CMD ["/bin/bash"]
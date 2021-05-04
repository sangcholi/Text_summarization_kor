FROM gpuci/miniconda-cuda:11.2-runtime-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://github.com/SKT-AI/KoBART#egg=kobart
RUN git clone https://github.com/seujung/KoBART-summarization.git
WORKDIR /KoBART-summarization/data
RUN tar xvf train.tar.gz
RUN tar xvf test.tar.gz

WORKDIR /KoBART-summarization
RUN pip install -r requirements.txt
RUN pip install torchtext==0.8.1
RUN mkdir checkpoint


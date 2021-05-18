FROM gpuci/miniconda-cuda:11.2-runtime-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b feat/docker https://github.com/LeeSeongYeob/Text_summarization_kor.git
WORKDIR /KoBART-summarization/data
RUN tar xvf train.tar.gz
RUN tar xvf test.tar.gz
RUN rm test.tar.gz
RUN rm train.tar.gz

WORKDIR /KoBART-summarization
RUN sh install_kobart.sh
RUN pip3 install -r requirements.txt
RUN pip3 install torchtext==0.8.1
RUN Python3 download_binary.py


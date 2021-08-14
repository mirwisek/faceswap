FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install wget python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

COPY . /app/
WORKDIR /app

RUN bash -c "cd /tmp; wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh; ls; pwd"
RUN bash /tmp/Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm /tmp/Anaconda3-5.0.1-Linux-x86_64.sh

ENV PATH /root/anaconda3/bin:$PATH

RUN /root/anaconda3/bin/conda create --name faceapp python=3.6 pip

RUN /root/anaconda3/envs/faceapp/bin/pip install -r /app/requirements.txt

EXPOSE 8080
CMD [ "/root/anaconda3/envs/faceapp/bin/gunicorn", "index:app", "-b", ":8080"]

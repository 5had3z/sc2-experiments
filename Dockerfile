FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y libboost-iostreams1.74-dev

RUN pip install --upgrade pip && pip install git+https://github.com/5had3z/sc2-serializer.git git+https://github.com/5had3z/konductor

WORKDIR /app
COPY . .
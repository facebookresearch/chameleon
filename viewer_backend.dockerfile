FROM nvcr.io/nvidia/pytorch:23.12-py3

COPY ./chameleon/viewer/backend/requirements.txt /workspace/chameleon/viewer/backend/requirements.txt
RUN pip install -r /workspace/chameleon/viewer/backend/requirements.txt

COPY ./chameleon/viewer/backend/ /workspace/chameleon/viewer/backend/
COPY ./chameleon/inference/ /workspace/chameleon/inference/
COPY ./config/ /workspace/config/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/
WORKDIR /workspace/

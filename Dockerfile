# syntax=docker/dockerfile:1.3
FROM nvcr.io/nvidia/deepstream:6.2-devel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN mkdir -p /workspace
WORKDIR /workspace
RUN apt-get update

COPY requirements.txt /workspace/
RUN python3 -m pip install pip --upgrade
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip --timeout=1000 install --ignore-installed -r requirements.txt


WORKDIR /code


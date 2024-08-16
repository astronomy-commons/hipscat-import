FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install .
RUN pip install --no-cache-dir -r requirements.lock

FROM osrf/ros:jazzy-desktop

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
      python3-pip \
      ros-jazzy-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed -r /app/requirements.txt

COPY . /app

ENV PYTHONPATH=/app

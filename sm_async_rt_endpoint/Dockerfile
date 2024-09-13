FROM nvcr.io/nvidia/pytorch:23.12-py3 as build

RUN apt update && DEBIAN_FRONTEND="noninteractive" apt -y install tzdata && \
    apt install -y zip libgl1-mesa-glx netbase libopencv-dev libopenblas-dev nginx && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install pip==24.2 --no-cache-dir

COPY ./scripts /opt/program

# Set work directory and install Python dependencies
WORKDIR /opt/program
## specific torch and torchvision version
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

RUN pip install -r requirements.txt --no-cache-dir

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/program:${PATH}"

RUN chmod +x /opt/program/serve
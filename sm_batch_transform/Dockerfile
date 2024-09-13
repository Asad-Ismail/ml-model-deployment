# Base image from NVIDIA for PyTorch with CUDA support
FROM nvcr.io/nvidia/pytorch:23.12-py3 as build

# Install required dependencies
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt -y install tzdata && \
    apt install -y zip libgl1-mesa-glx netbase libopencv-dev libopenblas-dev nginx && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install pip and specific versions of torch and torchvision
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install pip==24.2 --no-cache-dir

# Set working directory and copy scripts
WORKDIR /opt/program
COPY ./scripts /opt/program

# Install specific torch and torchvision versions
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

# Install other Python dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/program:${PATH}" \
    MODEL_SERVER_WORKERS=1

RUN chmod +x /opt/program/serve

# ENTRYPOINT setup for SageMaker Batch Transform
ENTRYPOINT ["./serve"]


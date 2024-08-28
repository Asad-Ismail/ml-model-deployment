FROM nvcr.io/nvidia/pytorch:23.12-py3 as build

RUN apt update && DEBIAN_FRONTEND="noninteractive" apt -y install tzdata && \
    apt install -y zip libgl1-mesa-glx netbase libopencv-dev libopenblas-dev nginx && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install pip==23.3.2 --no-cache-dir

COPY ./scripts /opt/program

# Set work directory and install Python dependencies
WORKDIR /opt/program
RUN pip install -r requirements.txt --no-cache-dir

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/program:${PATH}"

# Install PyTorch and TorchVision specific versions
#RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

# Install Detectron2
RUN python -m pip install git+https://github.com/facebookresearch/detectron2.git@bcfd464 --no-cache-dir

# Ensure serve script is executable
RUN chmod +x /opt/program/serve
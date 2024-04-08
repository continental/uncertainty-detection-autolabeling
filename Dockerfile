FROM nvidia/cuda:12.0.0-base-ubuntu20.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    apt-utils \
    nano \
    unzip \
    vim \
    wget \
    python3-tk \
    libcupti-dev \
    git \
    cuda-toolkit \
    ghostscript

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s pip3 pip

RUN pip3 install --upgrade pip

# Install required packages from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=6666", "--allow-root", "--no-browser"]
EXPOSE 6666

# # Run training
# CMD ["/entrypoint"]
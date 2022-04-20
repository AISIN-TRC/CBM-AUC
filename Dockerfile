FROM nvidia/cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-tk \
    git \
    curl \
    vim \
    python3-line-profiler \
    graphviz \
    nvidia-opencl-dev \
    nvidia-settings \
    nvidia-modprobe \
    nkf \
    emacs \
    python3-llvmlite \
     && ln -s /usr/bin/python3 /usr/bin/python \
     && mkdir sharedDir

RUN pip3 install --no-cache-dir \
    numpy==1.15.4 \
    matplotlib==2.2.3 \
    pickleshare==0.7.5 \
    PyYAML==5.1 \
    torch==1.1.0 \
    torchvision==0.2.2.post3 \
    optuna \
    pycallgraph \
    line_profiler \
    graphviz \
    gprof2dot \
    scikit-learn \
    scikit-image \
    scikit-optimize \
    jupyterlab \
    tensorflow \
    lime \
    tqdm \
    attrdict \
    pandas \



FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://de.ports.ubuntu.com/ubuntu-ports|g' \
      /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git git-lfs htop \
      libx11-6 ninja-build openssh-client pkg-config sudo tmux unzip vim zip \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_CONSTRAINT="" \
    PIP_CONFIG_FILE=/dev/null

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^\s*(torch==|causal-conv1d==|mamba-ssm==)' \
      /tmp/requirements.txt > /tmp/requirements-no-torch.txt \
    && python -m pip install --no-cache-dir --ignore-installed cryptography \
    && python -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt \
    && python -c "import torch; assert 'nv' in torch.__version__, torch.__version__"

RUN git clone --branch v1.6.0 --depth 1 \
      https://github.com/Dao-AILab/causal-conv1d.git /opt/causal-conv1d \
    && python -m pip install --no-cache-dir --no-build-isolation /opt/causal-conv1d

RUN git clone --recursive --depth 1 \
      https://github.com/state-spaces/mamba.git /opt/mamba \
    && python -m pip install --no-cache-dir --no-build-isolation /opt/mamba

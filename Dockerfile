FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git build-essential ca-certificates curl bash \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (no mamba)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Use bash for conda activation
SHELL ["/bin/bash", "-lc"]

# Create conda env with JAX CUDA 12.2
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN CONDA_OVERRIDE_CUDA=12.2 conda create --yes --strict-channel-priority \
    -c conda-forge -n skill jaxlib=*=*cuda* jax python=3.10

# Install packages inside the env
WORKDIR /workspace

# Copy only requirements and setup files for dependency installation
COPY requirements.txt .
COPY setup.py .

RUN conda run -n skill pip install craftax
RUN conda run -n skill pip install brax
RUN conda run -n skill pip install -r requirements.txt
RUN conda run -n skill pip install torch torchvision torchaudio --force
RUN conda run -n skill pip install -U --force "jax[cuda12]"

# Make the environment active by default
ENV CONDA_DEFAULT_ENV=skill
ENV PATH=$CONDA_DIR/envs/skill/bin:$PATH

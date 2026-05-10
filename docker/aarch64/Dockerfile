FROM nvcr.io/nvidia/pytorch:23.10-py3
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Only list Jetson Orin Arch - 8.7
ARG CUDA_SM_LIST="8.7"
ENV TORCH_CUDA_ARCH_LIST="${CUDA_SM_LIST}"

ENV https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

# Create user group
ARG USER_ID=1000
ARG GROUP_ID=1001
ARG USER_NAME=gms-ME
ARG HOST_GROUP_NAME=docker
RUN groupadd -f -o -g ${GROUP_ID} ${HOST_GROUP_NAME} || true
RUN useradd -r -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}
RUN chown -R ${USER_NAME}:${HOST_GROUP_NAME} /home/${USER_NAME}

# Switch to non-root mode
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# Build
RUN git config --global http.proxy http://127.0.0.1:7890
RUN git config --global https.proxy http://127.0.0.1:7890
RUN git clone https://github.com/Caprague/MinkowskiEngineCuda13.git
RUN mv ./MinkowskiEngineCuda13 ./MinkowskiEngine
WORKDIR ./MinkowskiEngine

# Apply the fix for error: Compilation: compiler finds both std::to_address and cuda::std::to_address (https://github.com/NVIDIA/MinkowskiEngine/issues/596)
USER root
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' /usr/include/c++/11/bits/shared_ptr_base.h
RUN python setup.py install --force_cuda --blas=openblas
USER ${USER_NAME}

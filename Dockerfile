FROM python:3.9-bullseye

# coral libraries
RUN apt-get update &&\
    apt-get install curl -y &&\
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - &&\
    apt-get update &&\
    apt-get install libedgetpu1-std -y &&\
    python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# libGL.so
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# add user to plugdev for TPU access
RUN sudo usermod -aG plugdev $USERNAME

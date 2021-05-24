FROM nvidia/cuda:11.0-devel-ubuntu18.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install sudo
RUN apt-get -y install python3-pip libxrender1 libsm6 xserver-xorg-core xorg python3-venv vim pciutils wget git kmod vim git
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

# work directory
WORKDIR /home/$USER_NAME

# install system dependencies
COPY ./scripts/install_deps.sh /tmp/install_deps.sh
COPY ./scripts/install_nvidia.sh /tmp/install_nvidia.sh
RUN yes "Y" | /tmp/install_deps.sh

# setup python environment
RUN cd $WORKDIR


ENV MOCA_ENV=/home/$USER_NAME/moca_env
RUN python3 -m virtualenv --python=/usr/bin/python3 $MOCA_ENV
ENV PATH="$MOCA_ENV/bin:$PATH"

RUN NVIDIA_VERSION=450.119.03 /tmp/install_nvidia.sh


# install python requirements
RUN pip install --upgrade pip
RUN pip install -U setuptools
COPY ./moca_requirements.txt /tmp/moca_requirements.txt
RUN pip install -r /tmp/moca_requirements.txt
RUN python3 -c "import ai2thor.controller; ai2thor.controller.Controller()"



# ENV VIRTUAL_ENV=/home/$USER_NAME/alfred_env
# RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # install python requirements
# RUN pip install --upgrade pip
# RUN pip install -U setuptools
# COPY ./requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt
# RUN python3 -c "import ai2thor.controller; ai2thor.controller.Controller()"

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils && \
   rm -rf /var/lib/apt/lists/*

# change ownership of everything to our user
RUN mkdir /home/$USER_NAME/alfred
RUN cd ${USER_HOME_DIR} && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .

# copy scripts
COPY ./scripts/startx.py /home/$USER_NAME/

ENTRYPOINT bash -c "export ALFRED_ROOT=~/alfred && /bin/bash"

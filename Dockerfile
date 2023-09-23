FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV SHELL=/bin/bash

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
    rsync \
    curl \
    cmake \
    git \
    wget \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libopencv-dev \
    libsuitesparse-dev \
    ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libboost-regex-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libcgal-dev \
    libceres-dev \
    liblz4-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install notebook \
                matplotlib \
                pandas \
                omegaconf \
                hydra-core \
                pytorch-ignite \
                kornia \
                scikit-image \
                opencv-python \
                Sphinx \
                joblib \
                pyproj \
                h5py \
                open3d \
                tensorboardX \
                tensorboard \
                geomloss \
                typing_extensions==4.0.0 \
                e2cnn \
                deepdish \
                ruamel.yaml

# Ceres solver
RUN \
    mkdir /source && cd /source && \
    curl -L http://ceres-solver.org/ceres-solver-1.14.0.tar.gz | tar xz && \
    cd ceres-solver-1.14.0 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_C_FLAGS=-fPIC \
             -DCMAKE_CXX_FLAGS=-fPIC \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_TESTING=OFF && \
    make -j6 install && \
    rm -rf /source/ceres-solver-1.14.0

# OpenGV
RUN \
    cd /source && \
    git clone --recursive https://github.com/paulinus/opengv.git && \
    cd opengv && \
    mkdir build && cd build && \
    cmake .. -DBUILD_TESTS=OFF \
             -DBUILD_PYTHON=ON \
             -DPYBIND11_PYTHON_VERSION=3.7 \
             -DPYTHON_INSTALL_DIR=/opt/conda/lib/python3.7/site-packages && \
    make -j6 install && \
    rm -rf /source/opengv

# G2O
RUN \
    cd /source && \
    git clone https://github.com/uoip/g2opy.git && \
    git clone https://github.com/luigifreda/pyslam && \
    rsync pyslam/thirdparty/g2opy_changes/types_six_dof_expmap.h g2opy/python/types/sba/types_six_dof_expmap.h && \
    rm -rf /source/pyslam && \
    cd g2opy && \
    mkdir build && cd build && \
    cmake .. && \
    make -j6 install && \
    cd .. && \
    cp lib/g2o.cpython-37m-x86_64-linux-gnu.so /opt/conda/lib/python3.7/site-packages && \
    rm -rf /source/g2opy

# PyDEGENSAC
RUN \
    cd /source && \
    git clone https://github.com/ducha-aiki/pydegensac.git && \
    cd pydegensac && \
    python setup.py build && \
    cp -R build/lib.linux-x86_64-3.7/pydegensac /opt/conda/lib/python3.7/site-packages && \
    rm -rf /source/pydegensac

# COLMAP
RUN \
   cd /source && \
   curl -L https://github.com/colmap/colmap/archive/refs/tags/3.6.tar.gz | tar xz && \
   cd colmap-3.6 && mkdir build && \
   cd build && \
   cmake .. -GNinja -DCMAKE_CXX_STANDARD=14 && \
   pip install ninja && ninja && ninja install && \
   rm -rf /source/colmap-3.6 \

RUN rm -rf /source

WORKDIR "/home/konstantin/personal/Summertime"

ENV PYTHONPATH="/home/konstantin/personal/Summertime:${PYTHONPATH}"

# PyDBoW3
#RUN \
#    cd source && \
#    git clone https://github.com/foxis/pyDBoW3 && \
#    cd pyDBoW3 && \
#    mkdir build && cd build && \
#    cmake .. -DBUILD_PYTHON3=ON \
#             -DBUILD_STATICALLY_LINKED=OFF \
#             -DDBoW3_DIR=$CWD/install/DBow3/build \
#             -DDBoW3_INCLUDE_DIRS=$CWD/install/DBow3/src \
#             -DCMAKE_BUILD_TYPE=Release

## DBoW3
#RUN \
#    cd /source && \
#    git clone https://github.com/rmsalinas/DBow3 && \
#    cd DBow3 && \
#    mkdir build && cd build && \
#    cmake .. -DBUILD_SHARED_LIBS=OFF \
#             -DUSE_CONTRIB=ON \
#             -DCMAKE_CXX_FLAGS="-fPIC" \
#             -DCMAKE_C_FLAGS="-fPIC" \
#             -DBUILD_UTILS=OFF \
#             && \
#    make -j6 install && \
#    rm -rf /source/DBow3

#    rsync pyslam/thirdparty/g2opy_changes/sparse_optimizer.h g2opy/python/core/sparse_optimizer.h && \
#    rsync pyslam/thirdparty/g2opy_changes/python_CMakeLists.txt g2opy/python/CMakeLists.txt && \
#    rsync pyslam/thirdparty/g2opy_changes/eigen_types.h g2opy/python/core/eigen_types.h && \

## OpenSfM
#RUN \
#    cd /source && \
#    git clone --recursive https://github.com/mapillary/OpenSfM && \
#    cd OpenSfM && \
#    python setup.py build && \
#    cp -R /source/OpenSfM/build/lib/opensfm /opt/conda/lib/python3.7/site-packages && \
#    rm -rf /source/OpenSfM
# python3-pip
# rm -r /opt/conda/

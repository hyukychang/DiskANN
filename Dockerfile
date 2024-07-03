#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

FROM ubuntu:jammy

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libcpprest-dev python3.10
# RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10
RUN apt install -y wget
RUN wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146.sh
RUN sh l_BaseKit_p_2022.1.2.146.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s


WORKDIR /app
RUN git clone https://github.com/microsoft/DiskANN.git 
WORKDIR /app/DiskANN
RUN mkdir build
RUN cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release
RUN CXXFLAGS="-Wno-error=deprecated-copy" cmake --build build -- -j

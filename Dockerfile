# syntax=docker/dockerfile:1.4-labs
FROM docker.io/intel/oneapi-basekit:2024.2.0-1-devel-ubuntu22.04
FROM docker.io/nvidia/cuda:12.5.0-devel-ubuntu22.04
ENV CUDADIR=/usr/local/cuda

# Install Intel oneAPI
COPY --from=0 /opt/intel/ /opt/intel/

# Proxy settings for apt inside the container
RUN echo Acquire::ftp::proxy   "\"${http_proxy}\";"      > /etc/apt/apt.conf \
	&& echo Acquire::http::proxy  "\"${https_proxy}\";" >> /etc/apt/apt.conf \
	&& echo Acquire::https::proxy "\"${http_proxy}\";"  >> /etc/apt/apt.conf

# Install basic tools
RUN apt-get update \
	&& apt-get install -y sudo wget htop zsh vim screen ca-certificates gpg

# Install developer tools
RUN /bin/bash -c "wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg \
	&& echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list \
	&& apt-get update \
	&& apt-get install -y build-essential git clang-format python3-pip \
	&& apt-get install -y cmake libeigen3-dev \
	&& apt-get -y upgrade"

# Build MAGMA with Intel MKL_ILP64 interface
ENV MAGMA_VERSION=magma-2.8.0
RUN /bin/bash -c 'MAGMA_VERSION=magma-2.8.0 \
	&& source /opt/intel/oneapi/setvars.sh \
	&& wget https://icl.utk.edu/projectsfiles/magma/downloads/${MAGMA_VERSION}.tar.gz \
	&& tar -xf ${MAGMA_VERSION}.tar.gz \
	&& cd ${MAGMA_VERSION} \
	&& cp -v make.inc-examples/make.inc.mkl-gcc-ilp64 make.inc \
	&& sed -i -e "s/^#GPU_TARGET ?=/GPU_TARGET :=/g" make.inc \
	&& sed -i -e "s/^FORT      = gfortran/FORT      =/g" make.inc \
	&& sed -i -e "/^#LIB *= -lmkl_gf_ilp64/aLIB       = -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm" make.inc \
	&& sed -i -e "s/^\(LIB *= -lmkl_gf_ilp64\)/#\1/g" make.inc \
	&& make -j $(grep processor /proc/cpuinfo | wc -l) && make -i install'
RUN sed 's/^Cflags: .*$/Cflags: -isystem ${includedir} -DADD_ -DMKL_ILP64/' -i /usr/local/magma/lib/pkgconfig/magma.pc \
	&& cd /root && rm -rf ${MAGMA_VERSION} ${MAGMA_VERSION}.tar.gz
ENV PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig

# Setup zshrc
COPY .zshrc /root/.zshrc
RUN cat /root/.zshrc >> /etc/zsh/zshrc && echo > /root/.zshrc


RUN chsh -s /bin/zsh root \
	&& mkdir -p /work
WORKDIR /work
CMD ["/bin/zsh"]
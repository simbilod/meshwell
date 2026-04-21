FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build MMG
RUN git clone https://github.com/MmgTools/mmg.git && \
    cd mmg && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_SCOTCH=OFF -DUSE_ELAS=OFF && \
    make -j$(nproc) && \
    make install

# Build ParMMG
RUN git clone https://github.com/MmgTools/ParMmg.git && \
    cd ParMmg && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_SCOTCH=OFF -DUSE_VTK=OFF && \
    make -j$(nproc) && \
    make install

# Final stage
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/mmg* /usr/local/bin/
COPY --from=builder /usr/local/bin/parmmg* /usr/local/bin/
COPY --from=builder /usr/local/include/mmg/ /usr/local/include/mmg/
COPY --from=builder /usr/local/include/parmmg/ /usr/local/include/parmmg/
COPY --from=builder /usr/local/lib/libmmg* /usr/local/lib/
COPY --from=builder /usr/local/lib/libparmmg* /usr/local/lib/

# Update shared library cache
RUN ldconfig

WORKDIR /workdir

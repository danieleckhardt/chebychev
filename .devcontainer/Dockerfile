FROM dealii/dealii:v9.6.0-jammy

USER root

# Set default password for SSH (optional, for debugging)
RUN echo 'dealii:dealii' | chpasswd

# Expose SSH port (if you intend to run SSH later)
EXPOSE 22

# Install development packages and clean up
RUN apt-get update && \
    apt-get install -y \
    lldb \
    lld \
    doxygen \
    gdb \
    git \
    liblapacke-dev \
    python3-pip \
    libboost-all-dev \
    libomp-dev \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependency
RUN pip install --no-cache-dir matplotlib2tikz

# Install Spectra (header-only C++ library)
RUN git clone https://github.com/yixuan/spectra.git /tmp/spectra && \
    mkdir -p /usr/local/include/Spectra && \
    cp -r /tmp/spectra/include/Spectra/* /usr/local/include/Spectra/ && \
    rm -rf /tmp/spectra

# Set user and working directory
USER dealii
WORKDIR /home/dealii


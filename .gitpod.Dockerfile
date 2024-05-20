FROM gitpod/workspace-full

USER gitpod

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "debconf debconf/frontend select Noninteractive" | sudo debconf-set-selections && \
    echo "keyboard-configuration keyboard-configuration/layout select 'English (US)'" | sudo debconf-set-selections && \
    echo "keyboard-configuration keyboard-configuration/layoutcode select 'us'" | sudo debconf-set-selections && \
    echo "resolvconf resolvconf/linkify-resolvconf boolean false" | sudo debconf-set-selections && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -O cuda_keyring.deb && \
    sudo dpkg -i cuda_keyring.deb && \
    rm cuda_keyring.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    sudo add-apt-repository deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ / && \
    sudo apt-get update -q && \
    sudo apt-get install cuda-12-3 -y --no-install-recommends && \
    sudo apt-get clean autoclean && \
    sudo apt-get autoremove -y && \
    sudo rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN cargo install cargo-reaper --git https://github.com/juntyr/grim-reaper --force

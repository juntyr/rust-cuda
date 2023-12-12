FROM gitpod/workspace-full

USER gitpod

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "debconf debconf/frontend select Noninteractive" | sudo debconf-set-selections && \
    echo "keyboard-configuration keyboard-configuration/layout select 'English (US)'" | sudo debconf-set-selections && \
    echo "keyboard-configuration keyboard-configuration/layoutcode select 'us'" | sudo debconf-set-selections && \
    echo "resolvconf resolvconf/linkify-resolvconf boolean false" | sudo debconf-set-selections && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
    sudo apt-get update -q && \
    sudo apt-get install cuda -y --no-install-recommends && \
    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && \
    sudo ./llvm.sh $(rustc --version -v | grep -oP "LLVM version: \K\d+") && \
    rm llvm.sh && \
    sudo apt-get clean autoclean && \
    sudo apt-get autoremove -y && \
    sudo rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN cargo install rust-ptx-linker --git https://github.com/juntyr/rust-ptx-linker --force && \
    cargo install cargo-reaper --git https://github.com/juntyr/grim-reaper --force

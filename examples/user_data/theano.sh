#!/bin/bash

# To remove grub prompt with dist-upgrade
export DEBIAN_FRONTEND=noninteractive

# For use on GPU instancees, e.g. g2.2xlarge
sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
sudo apt-get install -y libhdf5-serial-dev
sudo git clone https://github.com/ftzeng/factory.git /etc/factory
cd /etc/factory
pip install -r requirements.txt

# Theano (latest)
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
sudo pip install cython
sudo pip install h5py
sudo pip install keras

# Grab latest cuda toolkit (7.0)
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

# Add cuda nvcc and ld_library_path
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> ~/.bashrc

# Setup the theano config
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc

# Reboot for cuda to load
sudo reboot

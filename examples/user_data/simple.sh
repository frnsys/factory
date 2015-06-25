#!/bin/bash
sudo apt-get update
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
sudo git clone https://github.com/ftzeng/factory.git /etc/factory
cd /etc/factory
pip install -r requirements.txt
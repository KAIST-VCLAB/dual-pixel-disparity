#!/bin/bash
if [[ "$(uname -s)" =~ "Linux" ]]; then
    echo "----------------------------------------------------------------"
    echo "Detected: [Linux]"
    echo "----------------------------------------------------------------"
elif [[ "$(uname -s)" =~ "MINGW64_NT" ]]; then
    echo "----------------------------------------------------------------"
    echo "Detected: [Windows 64bit]"
    echo "----------------------------------------------------------------"
else
    echo "Only Linux / Windows 64bit is supported"
    exit
fi
PYTORCH_INSTALL="pip install --no-input --timeout 180 --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113"

source "$(dirname $(dirname $(which $CONDA_EXE)))"/etc/profile.d/conda.sh
conda create -y --name sfbd python=3.8.11
conda activate sfbd

# pytorch from conda channel takes too long. get pytorch from pypi
$PYTORCH_INSTALL

# conda packages
conda install -y \
    wget unzip \
    scipy==1.7.3

# pypi packages
pip install --no-input \
    easydict \
    tqdm \
    opencv-python-headless==4.6.0.66 \
    opencv-contrib-python-headless==4.6.0.66 \
    scikit-image==0.19.3 \
    opt_einsum==3.3.0

BACKBONE_VERSION="5899261abcf773aff652d71bf32ab62298d70add"
# use conda wget and unzip
wget https://github.com/princeton-vl/RAFT-Stereo/archive/"$BACKBONE_VERSION".zip -O temp.zip
unzip -o -qq temp.zip
rm temp.zip
cp -r RAFT-Stereo-"$BACKBONE_VERSION"/core .
rm -r RAFT-Stereo-"$BACKBONE_VERSION"
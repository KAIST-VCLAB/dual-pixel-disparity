#!/bin/bash
BACKBONE_VERSION="5899261abcf773aff652d71bf32ab62298d70add"
# use conda wget and unzip
wget https://github.com/princeton-vl/RAFT-Stereo/archive/"$BACKBONE_VERSION".zip -O temp.zip
unzip -o -qq temp.zip
rm temp.zip
cp -r RAFT-Stereo-"$BACKBONE_VERSION"/core .
rm -r RAFT-Stereo-"$BACKBONE_VERSION"
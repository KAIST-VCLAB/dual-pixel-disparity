import os
os.umask(0)
from pathlib import Path
import logging

import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch


LOGGER = logging.getLogger('sfbd')
UINT16_TO_UINT8 = 0.0038910505836575876 # 255 / 65535


def read_img(fpath:str, normalize=False):
    '''
    Return image as uint8
    '''
    ext = Path(fpath).suffix.lower()
    assert (ext=='.png') or (ext=='.jpg') or (ext=='.jpeg')
    img = cv2.imread(fpath, -1)
    if img.ndim == 3: img = img[:,:,2::-1] # Only rgb

    assert (img.dtype==np.uint8) or (img.dtype==np.uint16)
    if img.dtype == np.uint16:
        img = img * UINT16_TO_UINT8
    
    if normalize:
        img /= 255.0
        return img.astype(np.float32) / 255.0
    else:
        return img.astype(np.uint8)


def load_ckpt_local(net, ckpt_path, device):
    assert Path(ckpt_path).exists(), 'No checkpoint exists'
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_local = {key.replace("module.", ""): value for key, value in ckpt.items()}
    net.load_state_dict(ckpt_local)
    LOGGER.info(f'Checkpoint loaded from: {ckpt_path}')







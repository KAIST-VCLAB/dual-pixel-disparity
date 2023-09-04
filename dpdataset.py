import os
os.umask(0)
from pathlib import Path

import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch

import utils.io


class DPDataset:
    def __init__(self):
        super().__init__()
        self.dataset_name = None
    
    # TO BE OVERRIDED
    def stem_img(self, fpath: str) -> str:
        return Path(fpath).stem
    
    def read_img(self, fpath: str) -> np.ndarray:
        return utils.io.read_img(fpath)
    
    def stem_gt(self, fpath: str) -> str:
        return None
    
    def read_gt(self, fpath: str) -> np.ndarray:
        return None
    
    def get_gt_mask(self, disp: np.ndarray) -> np.ndarray:
        return None
    
    def post_process(self, imgs, gt, disp):
        return imgs, gt, disp


    # PUBLIC
    def load(self, img1_path: str, img2_path:str, gt_path=None):
        stem1 = self.stem_img(img1_path)
        stem2 = self.stem_img(img2_path)
        assert stem1 == stem2
        stem = stem1

        imgs = {}
        imgs['L'] = self.read_img(img1_path)
        imgs['R'] = self.read_img(img2_path)

        if gt_path is not None:
            assert stem == self.stem_gt(gt_path)
            gt = {}
            gt['d'] = self.read_gt(gt_path)
            gt['conf'] = self.get_gt_mask(gt['d'])
        else:
            gt = None

        return stem, imgs, gt


    def prepare(self, imgs:dict, device:torch.device):
        imgs_input = {}
        scale_factor = self._scale_factor(imgs['L'].shape[1])

        def _a2t(img:np.ndarray) -> torch.Tensor:
            # Assert image is RGB, H W 3
            assert img.ndim == 3
            assert img.max() < 256
            return torch.from_numpy(img/255.0).float().permute(2,0,1)

        for k, v in imgs.items():
            _img = cv2.resize(v, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            imgs_input[k] = _a2t(_img).unsqueeze(0).to(device)
        
        return imgs_input


    def pred_to_disp(self, predictions, origin_size):
        '''origin_size = (w, h)'''
        _disp = predictions['LCR']
        assert isinstance(_disp, torch.Tensor)
        _disp = _disp.squeeze(0).squeeze(0).cpu().numpy()
        return cv2.resize(_disp, dsize=origin_size, interpolation=cv2.INTER_LINEAR) / self._scale_factor(origin_size[0])


    # PRIVATE
    def _scale_factor(self, w):
        return 640 / w




class CanonDPDeblur(DPDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'Abuolaim_ECCV_2020'

    # TO BE OVERRIDED
    def stem_img(self, fpath: str) -> str:
        return f"{fpath.split('/')[-1].split('.')[0].split('_')[0]}"

    def read_img(self, fpath: str) -> np.ndarray:
        return utils.io.read_img(fpath)
    
    def post_process(self, imgs, gt, disp):
        def _crop(img, patch_size=111, stride=33):
            '''
            Center crop method from Punnappurath ICCP 2020 paper
            '''
            h, w = img.shape[:2]
            m = (patch_size - 1) / 2
            mids = int((stride - 1) / 2)
            rowmax = np.arange(m + 1, h - m, stride).astype(int)
            colmax = np.arange(m + 1, w - m, stride).astype(int)
            cropped = img[rowmax[0] - mids: rowmax[-1] + mids + 1, colmax[0] - mids: colmax[-1] + mids + 1]
            return cropped
        for k, v in imgs.items(): imgs[k] = _crop(v)
        disp = _crop(disp)
        return imgs, gt, disp
    



class CanonDPDepth(DPDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'Punnappurath_ICCP_2020'
    
    # TO BE OVERRIDED
    def stem_img(self, fpath: str) -> str:
        return f"{int(fpath.split('/')[-1].split('.')[0].split('_')[0]):03d}"
    
    def read_img(self, fpath: str) -> np.ndarray:
        img = utils.io.read_img(fpath)
        assert img.shape[:2] == (2940, 5180) # The original shape of data
        img = img[125 : 125+32*81, 143: 143+32*148, :]
        # Half size
        return cv2.resize(img, dsize=(2386, 1296), interpolation=cv2.INTER_LINEAR)
    
    def stem_gt(self, fpath: str) -> str:
        return self.stem_img(fpath)
    
    def read_gt(self, fpath: str) -> np.ndarray:
        gt = (cv2.imread(fpath)[:,:,0]/255.0).astype(np.float32)
        assert gt.shape[:2] == (2940, 5180) # The original shape of data
        gt = gt[125 : 125+32*81, 143: 143+32*148]
        # Half size
        return cv2.resize(gt, dsize=(2386, 1296), interpolation=cv2.INTER_LINEAR)
    
    def get_gt_mask(self, disp: np.ndarray) -> np.ndarray:
        return np.ones_like(disp, dtype=np.float32)
    
    def post_process(self, imgs, gt, disp):
        def _crop(img, patch_size=111, stride=33):
            '''
            Center crop method from Punnappurath ICCP 2020 paper
            '''
            h, w = img.shape[:2]
            m = (patch_size - 1) / 2
            mids = int((stride - 1) / 2)
            rowmax = np.arange(m + 1, h - m, stride).astype(int)
            colmax = np.arange(m + 1, w - m, stride).astype(int)
            cropped = img[rowmax[0] - mids: rowmax[-1] + mids + 1, colmax[0] - mids: colmax[-1] + mids + 1]
            return cropped
        for k, v in imgs.items(): imgs[k] = _crop(v)
        for k, v in gt.items(): gt[k] = _crop(v)
        disp = _crop(disp)
        return imgs, gt, disp




class PixelDPXin(DPDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'Xin_ICCV_2021'
    
    # TO BE OVERRIDED
    def stem_img(self, fpath: str) -> str:
        return fpath.split('/')[-1].split('.')[0].split('_')[0]
    
    def read_img(self, fpath: str) -> np.ndarray:
        _img = cv2.imread(fpath, -1)
        if _img.dtype == np.uint8:
            return np.dstack((_img, _img, _img))
        else:
            _img = (_img.astype(np.float32) - 1024) / (2**14 -1)
            _img = np.power(np.clip(_img, 0.0 , 1.0), 1/2.4)
            _img = (_img * 255).astype(np.uint8)
        return np.dstack((_img, _img, _img))

    def stem_gt(self, fpath: str) -> str:
        return self.stem_img(fpath)
    
    def read_gt(self, fpath: str) -> np.ndarray:
        return (cv2.imread(fpath)[:,:,0]/255.0).astype(np.float32)
    
    def get_gt_mask(self, disp: np.ndarray) -> np.ndarray:
        return (disp > 0).astype(np.float32)
    
    def post_process(self, imgs, gt, disp):
        def _crop(img):
            offset_y = (img.shape[0] - 1008) // 2
            offset_x = (img.shape[1] - 1344) // 2
            return img[offset_y:offset_y+1008, offset_x:offset_x+1344]
        
        for k, v in imgs.items(): imgs[k] = _crop(v)
        for k, v in gt.items(): gt[k] = _crop(v)
        disp = _crop(disp)
        return imgs, gt, disp



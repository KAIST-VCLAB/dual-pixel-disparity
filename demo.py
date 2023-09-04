import os
os.umask(0)
import logging
from glob import glob
from pathlib import Path
import argparse
import datetime

from tqdm import tqdm
import numpy as np
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
import torch

import utils
from model import BidirDPDispNet
import dpdataset


LOGGER = utils.mylogger.init_logger('sfbd', level=logging.INFO)


@torch.no_grad()
def main(args):
    cfg = utils.config.load_config(args.config_path)

    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        '''
        Following things control the deterministic behavior of cudnn.
        If set (deterministic: False, benchmark: True), cudnn will behave non-deterministic.
        This can improve the computing performance, but result can be possibly different.
        For the exact reproduction, set as (deterministic: True, benchmark: False).
        '''
        # :torch.backends.cudnn.deterministic: Whether to use deterministic operations
        torch.backends.cudnn.deterministic = True
        # :torch.backends.cudnn.benchmark: Whether to use optimized operation of cudnn
        torch.backends.cudnn.benchmark = False
    else:
        DEVICE = torch.device('cpu')
    LOGGER.info(f'Device: {DEVICE}')
    
    
    # ================ generate directory for output ================
    result_dir = f"{args.result_root}/{args.data_name}"
    LOGGER.info(f'Output: {result_dir}')
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    if args.save_disp:
        (Path(result_dir)/'disp').mkdir(exist_ok=True)
    if args.save_invdepth:
        (Path(result_dir)/'invdepth').mkdir(exist_ok=True)


    # ================ get images path ================
    left_list, right_list = sorted(glob(args.l)), sorted(glob(args.r))
    assert (len(left_list) != 0) and (len(right_list) != 0), 'No images with given search pattern'

    if args.gt is not None:
        gt_list = sorted(glob(args.gt))
        assert len(gt_list) == len(left_list), 'Number of GT and image not matches'
    else:
        gt_list = [None] * len(left_list)
    

    # ================ get dataset ================
    assert hasattr(dpdataset, args.data_name), f'No dataset named {args.data_name}. It tmust be same as class name in dpdataset.py.'
    dataset = getattr(dpdataset, args.data_name)()
    LOGGER.debug(f"Dataset: {str(dataset)}")


    # ================ prepare network ================
    net = BidirDPDispNet(cfg.model).to(DEVICE)
    utils.io.load_ckpt_local(net, args.ckpt_path, DEVICE)


    # ================ run test ================
    evaluations = []
    for left_path, right_path, gt_path in zip(tqdm(left_list), right_list, gt_list):
        LOGGER.debug(f"[L]{left_path}, [R]{right_path}, [GT]{gt_path}")
        stem, imgs, gt = dataset.load(left_path, right_path, gt_path)
        
        _inputs = dataset.prepare(imgs, DEVICE)
        predictions = net(_inputs['L'], _inputs['R'], iters = cfg.model.valid_iters, test_mode=True)
        disp = dataset.pred_to_disp(predictions, (imgs['L'].shape[1], imgs['L'].shape[0]))
        
        imgs, gt, disp = dataset.post_process(imgs, gt, disp)

        if args.save_disp:
            cv2.imwrite(f'{result_dir}/disp/{stem}.exr', disp)
        
        # ================ evauation ================
        quant_eval = {}
        quant_eval['name'] = stem
        if gt is None:
            quant_eval['RMSE'] = utils.metrics.photometric_rmse(imgs['L'], imgs['R'], disp)
        else:
            ai1, b1 = utils.metrics.affine_invariant_1(disp, gt['d'], confidence_map=gt['conf'])
            ai2, b2 = utils.metrics.affine_invariant_2(disp, gt['d'], confidence_map=gt['conf'])
            disp_affine = disp * b2[0] + b2[1]
            sp = 1 - np.abs(utils.metrics.spearman_correlation(disp_affine, gt['d'], W=gt['conf']))
            quant_eval['AI1'] = ai1
            quant_eval['AI2'] = ai2
            quant_eval['1-|spcc|'] = sp
            if args.save_invdepth:
                cv2.imwrite(f'{result_dir}/invdepth/{stem}.exr', disp_affine)
        evaluations.append(quant_eval)
    
    
    # ================ compute average ================
    avg_metrics = {}
    for k in evaluations[0].keys():
        if k == 'name': continue
        avg_metrics[k] = 0.0
        for e in evaluations:
            avg_metrics[k] += e[k] / len(evaluations)
        LOGGER.info(f"{k}: {avg_metrics[k]:.4f}")
        
    
    # ================ write ================
    with Path(f'{result_dir}/avg_metrics.txt').open('a') as f:  
        f.write('================================================================\n')
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')}\n")
        f.write('----------------------------------------------------------------\n')
        for k, v in avg_metrics.items():
            f.write(f"{k}={v:.8f}\n")
        f.write('================================================================\n')




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ckpt_path',     type=str, default='ckpt/checkpoint.pt',                     help="Checkpoint path")
    parser.add_argument('-config_path',   type=str, default='config/config.json',                     help="Config file path")
    parser.add_argument('-result_root',   type=str, default='./result',                               help="Directory for output (will be created if not exist)")
    parser.add_argument('-data_name',     type=str, default='CanonDPDepth',                           help="Dataset name, this must be same as class name in dpdataset.py")
    parser.add_argument('-l',             type=str, default='../data/Punnappurath_ICCP_2020/*_L.jpg', help="Glob pattern for the left image")
    parser.add_argument('-r',             type=str, default='../data/Punnappurath_ICCP_2020/*_R.jpg', help="Glob pattern for the right image")
    parser.add_argument('-gt',            type=str, default=None,                                     help="Glob pattern for the gt image")
    parser.add_argument('-save_disp',     action='store_true',                                        help="Save disparity")
    parser.add_argument('-save_invdepth', action='store_true',                                        help="Save inverse depth")
    
    test_args = parser.parse_known_args()[0]

    print('================================================================')
    main(test_args)
    print('================================================================')

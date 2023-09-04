import numpy as np
import cv2
from scipy.stats import rankdata


def photometric_rmse(
        img1: np.ndarray,
        img2: np.ndarray, 
        disp1to2: np.ndarray):

    trg, src = img1, img2
    _coords_y, _coords_x = np.indices(trg.shape[:2], dtype=np.float32)
    coords_x = _coords_x + disp1to2
    remapped_src = cv2.remap(src, coords_x, _coords_y, interpolation=cv2.INTER_LINEAR)

    return np.sqrt(np.mean(np.square(remapped_src.astype(np.float32) - trg.astype(np.float32))))

'''
Following metrics originally from Garg et al., ICCV 2019,
https://github.com/google-research/google-research/blob/master/dual_pixels/eval/get_metrics.py
under Apache License, Version 2.0
'''
def affine_invariant_1(
        Y: np.ndarray,
        Target: np.ndarray,
        confidence_map = None,
        irls_iters = 5,
        eps = 1e-3):
    assert Y.shape==Target.shape
    if confidence_map is None: confidence_map = np.ones_like(Target)
    y = Y.ravel() # [N,]
    t = Target.ravel() # [N,]
    conf = confidence_map.ravel() # [N,]

    # w : IRLS weight
    # b : affine parameter
    # initialize IRLS weight
    w = np.ones_like(y, float) # [N,]
    ones = np.ones_like(y, float)
    # run IRLS
    for _ in range(irls_iters):
        w_sqrt = np.sqrt(w * conf) # [N,]
        WX = w_sqrt[:, None] * np.stack([y, ones], 1) # [N,1] * [N,2] = [N,2] (broadcast)
        Wt = w_sqrt * t # [N,]
        # solve linear system: WXb - Wt
        b = np.linalg.lstsq(WX, Wt, rcond=None)[0] # [2,]
        affine_y = y * b[0] + b[1]
        residual = np.abs(affine_y - t)
        # re-compute weight with clipping residuals
        w = 1 / np.maximum(eps, residual)
    
    # finally,
    ai1 = np.sum(conf * residual) / np.sum(conf)
    return ai1, b


def affine_invariant_2(
        Y: np.ndarray,
        Target: np.ndarray,
        confidence_map = None,
        eps = 1e-3):
    assert Y.shape==Target.shape
    if confidence_map is None: confidence_map = np.ones_like(Target)
    y = Y.ravel() # [N,]
    t = Target.ravel() # [N,]
    conf = confidence_map.ravel() # [N,]

    ones = np.ones_like(y, float)
    X = conf[:, None] * np.stack([y, ones], 1) # [N,1] * [N,2] = [N,2] (broadcast)
    t = conf * t # [N,]
    b = np.linalg.lstsq(X, t, rcond=None)[0] # [2,]
    affine_y = y * b[0] + b[1]

    # clipping residuals
    residual_sq = np.minimum(np.square(affine_y - t), np.finfo(np.float32).max)

    # finally,
    ai2 = np.sqrt(np.sum(conf * residual_sq) / np.sum(conf))
    return ai2, b


def spearman_correlation(
        X: np.ndarray,
        Y: np.ndarray,
        W = None):
    assert X.shape == Y.shape
    if W is None: W = np.ones_like(X)
    x, y, w = X.ravel(), Y.ravel(), W.ravel()

    # scale rank to -1 to 1 (for numerical stability)
    def _rescale_rank(z): return (z - len(z) // 2) / (len(z) // 2)
    rx = _rescale_rank(rankdata(x, method='dense'))
    ry = _rescale_rank(rankdata(y, method='dense'))

    def E(z): return np.sum(w * z) / np.sum(w)
    def _pearson_correlation(x, y):
        mu_x = E(x)
        mu_y = E(y)
        var_x = E(x * x) - mu_x * mu_x
        var_y = E(y * y) - mu_y * mu_y
        return (E(x * y) - mu_x * mu_y) / (np.sqrt(var_x * var_y))

    return _pearson_correlation(rx, ry)




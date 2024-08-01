import torch
import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl)

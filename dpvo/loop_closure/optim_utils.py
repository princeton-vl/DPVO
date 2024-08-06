import cuda_ba
import numba as nb
import numpy as np
import pypose as pp
import torch
from einops import parse_shape, rearrange
from scipy.spatial.transform import Rotation as R


def make_pypose_Sim3(rot, t, s):
    q = R.from_matrix(rot).as_quat()
    data = np.concatenate([t, q, np.array(s).reshape((1,))])
    return pp.Sim3(data)

def SE3_to_Sim3(x: pp.SE3):
    out = torch.cat((x.data, torch.ones_like(x.data[...,:1])), dim=-1)
    return pp.Sim3(out)

@nb.njit(cache=True)
def _format(es):
    return np.asarray(es, dtype=np.int64).reshape((-1, 2))[1:]

@nb.njit(cache=True)
def reduce_edges(flow_mag, ii, jj, max_num_edges, nms):
    es = [(-1, -1)]

    if ii.size == 0:
        return _format(es)

    Ni, Nj = (ii.max()+1), (jj.max()+1)
    ignore_lookup = np.zeros((Ni, Nj), dtype=nb.bool_)

    idxs = np.argsort(flow_mag)
    for idx in idxs: # edge index

        if len(es) > max_num_edges:
            break

        i = ii[idx]
        j = jj[idx]
        mag = flow_mag[idx]

        if ((j - i) < 30):
            continue

        if mag >= 1000: # i.e., inf
            continue

        if ignore_lookup[i, j]:
            continue

        es.append((i, j))

        for di in range(-nms, nms+1):
            i1 = i + di

            if 0 <= i1 < Ni:
                ignore_lookup[i1, j] = True

    return _format(es)



@nb.njit(cache=True)
def umeyama_alignment(x: np.ndarray, y: np.ndarray):
    """
    The following function was copied from:
    https://github.com/MichaelGrupp/evo/blob/3067541b350528fe46375423e5bc3a7c42c06c63/evo/core/geometry.py#L35

    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.sum(axis=1) / n
    mean_y = y.sum(axis=1) / n

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        return None, None, None # Degenerate covariance rank, Umeyama alignment is not possible

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s))
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

@nb.njit(cache=True)
def ransac_umeyama(src_points, dst_points, iterations=1, threshold=0.1):
    best_inliers = 0
    best_R = None
    best_t = None
    best_s = None
    for _ in range(iterations):
        # Randomly select three points
        indices = np.random.choice(src_points.shape[0], 3, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        # Estimate transformation
        R, t, s = umeyama_alignment(src_sample.T, dst_sample.T)
        if t is None:
            continue

        # Apply transformation
        transformed = (src_points @ (R * s).T) + t

        # Count inliers (not ideal because depends on scene scale)
        distances = np.sum((transformed - dst_points)**2, axis=1)**0.5
        inlier_mask = distances < threshold
        inliers = np.sum(inlier_mask)

        # Update best transformation
        if inliers > best_inliers:
            best_inliers = inliers
            best_R, best_t, best_s = umeyama_alignment(src_points[inlier_mask].T, dst_points[inlier_mask].T)

        if inliers > 100:
            break

    return best_R, best_t, best_s, best_inliers

def batch_jacobian(func, x):
  def _func_sum(*x):
    return func(*x).sum(dim=0)
  _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
  return rearrange(torch.stack((b,c)), 'N O B I -> N B O I', N=2)

def _residual(C, Gi, Gj):
    assert parse_shape(C, 'N _') == parse_shape(Gi, 'N _') == parse_shape(Gj, 'N _')
    out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
    return out.Log().tensor()

def residual(Ginv, input_poses, dSloop, ii, jj, jacobian=False):

    # prep
    device = Ginv.device
    assert parse_shape(input_poses, '_ d') == dict(d=7)
    pred_inv_poses = SE3_to_Sim3(input_poses).Inv()

    # free variables
    n, _ = pred_inv_poses.shape
    kk = torch.arange(1, n, device=device)
    ll = kk-1

    # constants
    Ti = pred_inv_poses[kk]
    Tj = pred_inv_poses[ll]
    dSij = Tj @ Ti.Inv()

    constants = torch.cat((dSij, dSloop), dim=0)
    iii = torch.cat((kk, ii))
    jjj = torch.cat((ll, jj))
    resid = _residual(constants, Ginv[iii], Ginv[jjj])

    if not jacobian:
        return resid

    J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
    return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)
    # print(f"{J_Ginv_i.shape=} {J_Ginv_j.shape=} {resid.shape=} {iii.shape=} {jjj.shape=}")

    r = iii.numel()
    assert parse_shape(J_Ginv_i, 'r do di') == parse_shape(J_Ginv_j, 'r do di') == dict(r=r, do=7, di=7)
    J = torch.zeros(r, n, 7, 7, device=device)
    rr = torch.arange(r, device=device)
    J[rr, iii] = J_Ginv_i
    J[rr, jjj] = J_Ginv_j
    J = rearrange(J, 'r n do di -> r do n di')

    return resid, J, (J_Ginv_i, J_Ginv_j, iii, jjj)

def run_DPVO_PGO(pred_poses, loop_poses, loop_ii, loop_jj, queue):
    final_est = perform_updates(pred_poses, loop_poses, loop_ii, loop_jj, iters=30)

    safe_i = loop_ii.max().item() + 1
    aa = SE3_to_Sim3(pred_poses.cpu())
    final_est = (aa[[safe_i]] * final_est[[safe_i]].Inv()) * final_est
    output = final_est[:safe_i]
    queue.put(output)

def perform_updates(input_poses, dSloop, ii_loop, jj_loop, iters, ep=0.0, lmbda=1e-6, fix_opt_window=False):
    """ Run the Levenberg Marquardt algorithm """

    input_poses = input_poses.clone()

    if fix_opt_window:
        freen = torch.cat((ii_loop, jj_loop)).max().item() + 1
    else:
        freen = -1

    Ginv = SE3_to_Sim3(input_poses).Inv().Log()

    residual_history = []

    for itr in range(iters):
        resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = residual(Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
        residual_history.append(resid.square().mean().item())
        # print("#Residual", residual_history[-1])
        delta_pose, = cuda_ba.solve_system(J_Ginv_i, J_Ginv_j, iii, jjj, resid, ep, lmbda, freen)
        assert Ginv.shape == delta_pose.shape
        Ginv_tmp = Ginv + delta_pose

        new_resid = residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
        if new_resid.square().mean() < residual_history[-1]:
            Ginv = Ginv_tmp
            lmbda /= 2
        else:
            lmbda *= 2

        if (residual_history[-1] < 1e-5) and (itr >= 4) and ((residual_history[-5] / residual_history[-1]) < 1.5):
            break

    return pp.Exp(Ginv).Inv()

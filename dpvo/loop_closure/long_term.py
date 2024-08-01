import os

import kornia as K
import kornia.feature as KF
import numpy as np
import pypose as pp
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from einops import asnumpy, rearrange, repeat
from torch_scatter import scatter_max

from .. import fastba
from .. import projective_ops as pops
from ..lietorch import SE3
from .optim_utils import SE3_to_Sim3, make_pypose_Sim3, ransac_umeyama, run_DPVO_PGO
from .retrieval import ImageCache, RetrievalDBOW


class LongTermLoopClosure:

    def __init__(self, cfg, patchgraph):
        self.cfg = cfg

        # Data structures to manage retrieval
        self.retrieval = RetrievalDBOW()
        self.imcache = ImageCache()

        # Process to run PGO in parallel
        self.lc_pool = mp.Pool(processes=1)
        self.lc_process = self.lc_pool.apply_async(os.getpid)
        self.manager = mp.Manager()
        self.result_queue = self.manager.Queue()
        self.lc_in_progress = False

        # Patch graph + loop edges
        self.pg = patchgraph
        self.loop_ii = torch.zeros(0, dtype=torch.long)
        self.loop_jj = torch.zeros(0, dtype=torch.long)

        self.lc_count = 0

        # warmup the jit compiler
        ransac_umeyama(np.random.randn(3,3), np.random.randn(3,3), iterations=200, threshold=0.01)

        self.detector = KF.DISK.from_pretrained("depth").to("cuda").eval()
        self.matcher = KF.LightGlue("disk").to("cuda").eval()

    def detect_keypoints(self, images, num_features=2048):
        """ Pretty self explanitory! Alas, we can only use disk w/ lightglue. ORB is brittle """
        _, _, h, w = images.shape
        wh = torch.tensor([w, h]).view(1, 2).float().cuda()
        features = self.detector(images, num_features, pad_if_not_divisible=True, window_size=15, score_threshold=40.0)
        return [{
            "keypoints": f.keypoints[None],
            "descriptors": f.descriptors[None],
            "image_size": wh
        } for f in features]


    def __call__(self, img, n):
        img_np = K.tensor_to_image(img)
        self.retrieval(img_np, n)
        self.imcache(img_np, n)

    def keyframe(self, k):
        self.retrieval.keyframe(k)
        self.imcache.keyframe(k)

    def estimate_3d_keypoints(self, i):
        """ Detect, match and triangulate 3D points """

        """ Load the triplet of frames """
        image_orig = self.imcache.load_frames([i-1,i,i+1], self.pg.intrinsics.device)
        image = image_orig.float() / 255
        fl = self.detect_keypoints(image)

        """ Form keypoint trajectories """
        trajectories = torch.full((2048, 3), -1, device='cuda', dtype=torch.long)
        trajectories[:,1] = torch.arange(2048)

        out = self.matcher({"image0": fl[0], "image1": fl[1]})
        i0, i1 = out["matches"][0].mT
        trajectories[i1, 0] = i0

        out = self.matcher({"image0": fl[2], "image1": fl[1]})
        i2, i1 = out["matches"][0].mT
        trajectories[i1, 2] = i2

        trajectories = trajectories[torch.randperm(2048)]
        trajectories = trajectories[trajectories.min(dim=1).values >= 0]

        a,b,c = trajectories.mT
        n, _ = trajectories.shape
        kps0 = fl[0]['keypoints'][:,a]
        kps1 = fl[1]['keypoints'][:,b]
        kps2 = fl[2]['keypoints'][:,c]

        desc1 = fl[1]['descriptors'][:,b]
        image_size = fl[1]["image_size"]

        kk = torch.arange(n).cuda().repeat(2)
        ii = torch.ones(2*n, device='cuda', dtype=torch.long)
        jj = torch.zeros(2*n, device='cuda', dtype=torch.long)
        jj[n:] = 2


        """ Construct "mini" patch graph. """
        true_disp = self.pg.patches_[i,:,2,1,1].median()
        patches = torch.cat((kps1, torch.ones(1, n, 1).cuda() * true_disp), dim=-1)
        patches = repeat(patches, '1 n uvd -> 1 n uvd 3 3', uvd=3)
        target = rearrange(torch.stack((kps0, kps2)), 'ot 1 n uv -> 1 (ot n) uv', uv=2, n=n, ot=2)
        weight = torch.ones_like(target)

        poses = self.pg.poses[:,i-1:i+2].clone()
        intrinsics = self.pg.intrinsics[:,i-1:i+2].clone() * 4

        coords = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        coords = coords[:,:,1,1]
        residual = (coords - target).norm(dim=-1).squeeze(0)

        """ structure-only bundle adjustment """
        lmbda = torch.as_tensor([1e-3], device="cuda")
        fastba.BA(poses, patches, intrinsics,
            target, weight, lmbda, ii, jj, kk, 3, 3, M=-1, iterations=6, eff_impl=False)

        """ Only keep points with small residuals """
        coords = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        coords = coords[:,:,1,1]
        residual = (coords - target).norm(dim=-1).squeeze(0)
        assert residual.numel() == 2*n
        mask = scatter_max(residual, kk)[0] < 2

        """ Un-project keypoints """
        points = pops.iproj(patches, intrinsics[:,torch.ones(n, device='cuda', dtype=torch.long)])
        points = (points[...,1,1,:3] / points[...,1,1,3:])

        return points[:,mask].squeeze(0), {"keypoints": kps1[:,mask], "descriptors": desc1[:,mask], "image_size": image_size}

    def attempt_loop_closure(self, n):
        if self.lc_in_progress:
            return

        """ Check if a loop was detected """
        cands = self.retrieval.detect_loop(thresh=self.cfg.LOOP_RETR_THRESH, num_repeat=self.cfg.LOOP_CLOSE_WINDOW_SIZE)
        if cands is not None:
            i, j = cands

            """ A loop was detected. Try to close it """
            lc_result = self.close_loop(i, j, n)
            self.lc_count += int(lc_result)

            """ Avoid multiple back-to-back detections """
            if lc_result:
                self.retrieval.confirm_loop(i, j)
            self.retrieval.found.clear()

        """ "Flush" the queue of frames into the loop-closure pipeline """
        self.retrieval.save_up_to(n - self.cfg.REMOVAL_WINDOW - 2)
        self.imcache.save_up_to(n - self.cfg.REMOVAL_WINDOW - 1)

    def terminate(self, n):
        self.retrieval.save_up_to(n-1)
        self.imcache.save_up_to(n-1)
        self.attempt_loop_closure(n)
        if self.lc_in_progress:
            self.lc_callback(skip_if_empty=False)
        self.lc_process.get()
        self.imcache.close()
        self.lc_pool.close()
        self.retrieval.close()
        print(f"LC COUNT: {self.lc_count}")


    def _rescale_deltas(self, s):
        """ Rescale the poses of removed frames by their predicted scales """

        tstamp_2_rescale = {}
        for i in range(self.pg.n):
            tstamp_2_rescale[self.pg.tstamps_[i]] = s[i]

        for t, (t0, dP) in self.pg.delta.items():
            t_src = t
            while t_src in self.pg.delta:
                t_src, _ = self.pg.delta[t_src]
            s1 = tstamp_2_rescale[t_src]
            self.pg.delta[t] = (t0, dP.scale(s1))

    def lc_callback(self, skip_if_empty=True):
        """ Check if the PGO finished running """
        if skip_if_empty and self.result_queue.empty():
            return
        self.lc_in_progress = False
        final_est = self.result_queue.get()
        safe_i, _ = final_est.shape
        res, s = final_est.tensor().cuda().split([7,1], dim=1)
        s1 = torch.ones(self.pg.n, device=s.device)
        s1[:safe_i] = s.squeeze()

        self.pg.poses_[:safe_i] = SE3(res).inv().data
        self.pg.patches_[:safe_i,:,2] /= s.view(safe_i, 1, 1, 1)
        self._rescale_deltas(s1)
        self.pg.normalize()

    def close_loop(self, i, j, n):
        """ This function tries to actually execute the loop closure """
        MIN_NUM_INLIERS = 30 # Minimum number of inlier matches
        # print("Found a match!", i, j)

        """ Estimate 3d keypoints w/ features"""
        i_pts, i_feat = self.estimate_3d_keypoints(i)
        j_pts, j_feat = self.estimate_3d_keypoints(j)
        _, _, iz = i_pts.mT
        _, _, jz = j_pts.mT
        th = 20 # a depth threshold. Far-away points aren't helpful
        i_pts = i_pts[iz < th]
        j_pts = j_pts[jz < th]
        for key in ['keypoints', 'descriptors']:
            i_feat[key] = i_feat[key][:,iz < th]
            j_feat[key] = j_feat[key][:,jz < th]

        # Early exit
        if i_pts.numel() < MIN_NUM_INLIERS:
            # print(f"Too few inliers (A): {i_pts.numel()=}")
            return False

        """ Match between the two point clouds """
        out = self.matcher({"image0": i_feat, "image1": j_feat})
        i_ind, j_ind = out["matches"][0].mT
        i_pts = i_pts[i_ind]
        j_pts = j_pts[j_ind]
        assert i_pts.shape == j_pts.shape, (i_pts.shape, j_pts.shape)
        i_pts, j_pts = asnumpy(i_pts.double()), asnumpy(j_pts.double())

        # Early exit
        if i_pts.size < MIN_NUM_INLIERS:
            # print(f"Too few inliers (B): {i_pts.size=}")
            return False

        """ Estimate Sim(3) transformation """
        r, t, s, num_inliers = ransac_umeyama(i_pts, j_pts, iterations=400, threshold=0.1) # threshold shouldn't be too low

        # Exist if number of inlier matches is too small
        if num_inliers < MIN_NUM_INLIERS:
            # print(f"Too few inliers (C): {num_inliers=}")
            return False

        """ Run Pose-Graph Optimization (PGO) """
        far_rel_pose = make_pypose_Sim3(r, t, s)[None]
        Gi = pp.SE3(self.pg.poses[:,self.loop_ii])
        Gj = pp.SE3(self.pg.poses[:,self.loop_jj])
        Gij = Gj * Gi.Inv()
        prev_sim3 = SE3_to_Sim3(Gij).data[0].cpu()
        loop_poses = pp.Sim3(torch.cat((prev_sim3, far_rel_pose)))
        loop_ii = torch.cat((self.loop_ii, torch.tensor([i])))
        loop_jj = torch.cat((self.loop_jj, torch.tensor([j])))

        pred_poses = pp.SE3(self.pg.poses_[:n]).Inv().cpu()

        self.loop_ii = loop_ii
        self.loop_jj = loop_jj

        torch.set_num_threads(1)

        self.lc_in_progress = True
        self.lc_process = self.lc_pool.apply_async(run_DPVO_PGO, (pred_poses.data, loop_poses.data, loop_ii, loop_jj, self.result_queue))
        return True
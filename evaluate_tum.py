import sys
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def tum_image_stream(queue, scene_dir, sequence, stride, skip=0):
    """ image generator """
    images_dir = scene_dir / "rgb"

    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    image_list = sorted(images_dir.glob("*.png"))[skip::stride]

    for imfile in image_list:
        image = cv2.imread(str(imfile))
        image = cv2.undistort(image, K_l, d_l)
        image = image.transpose(2,0,1)

        intrinsics = np.asarray([fx, fy, cx, cy])

        # crop image to remove distortion boundary
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        # intrinsics = intrinsics[None]
        image = image[:, 8:-8, 16:-16]

        queue.put((float(imfile.stem), image, intrinsics))

    queue.put((-1, image, intrinsics))

@torch.no_grad()
def run(cfg, network, scene_dir, sequence, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=tum_image_stream, args=(queue, scene_dir, sequence, stride, 0))
    reader.start()

    for step in range(sys.maxsize):
        (t, images, intrinsics) = queue.get()
        if t < 0: break

        images = torch.as_tensor(images, device='cuda')
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        if show_img:
            show_image(images[0], 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=images.shape[-2], wd=images.shape[-1], viz=viz)

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, images, intrinsics)

    reader.join()

    poses, tstamps = slam.terminate()
    np.save(f"poses_{sequence}.npy", poses)
    np.save(f"tstamps_{sequence}.npy", tstamps)
    return poses, tstamps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--tumdir', type=Path, default="datasets/TUM_RGBD")
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    tum_scenes = [
        "rgbd_dataset_freiburg1_360",
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_desk2",
        "rgbd_dataset_freiburg1_floor",
        "rgbd_dataset_freiburg1_plant",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg1_rpy",
        "rgbd_dataset_freiburg1_teddy",
        "rgbd_dataset_freiburg1_xyz",
    ]

    results = {}
    for scene in tum_scenes:
        scene_dir = args.tumdir / f"{scene}"
        groundtruth = scene_dir / "groundtruth.txt"
        traj_ref = file_interface.read_tum_trajectory_file(groundtruth)

        scene_results = []
        for trial_num in range(args.trials):
            traj_est, timestamps = run(cfg, args.network, scene_dir, scene, args.stride, args.viz, args.show_img)

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=timestamps)

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            ate_score = result.stats["rmse"]

            if args.plot:
                Path("trajectory_plots").mkdir(exist_ok=True)
                plot_trajectory(traj_est, traj_ref, f"TUM-RGBD Frieburg1 {scene} Trial #{trial_num+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/TUM_RGBD_Frieburg1_{scene}_Trial{trial_num+1:02d}.pdf", align=True, correct_scale=True)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/TUM_RGBD_{scene}_Trial{trial_num+1:02d}.txt", traj_est)

            scene_results.append(ate_score)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))

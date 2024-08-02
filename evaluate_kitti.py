from itertools import count
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

# From https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def kitti_image_stream(queue, kittidir, sequence, stride, skip=0):
    """ image generator """
    images_dir = kittidir / "dataset" / "sequences" / sequence
    image_list = sorted((images_dir / "image_2").glob("*.png"))[skip::stride]

    calib = read_calib_file(images_dir / "calib.txt")
    intrinsics = calib['P0'][[0, 5, 2, 6]]

    for t, imfile in enumerate(image_list):
        image_left = cv2.imread(str(imfile))
        H, W, _ = image_left.shape
        H, W = (H - H%4, W - W%4)
        image_left = image_left[..., :H, :W, :]

        queue.put((t, image_left, intrinsics))

    queue.put((-1, image_left, intrinsics))

@torch.no_grad()
def run(cfg, network, kittidir, sequence, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=kitti_image_stream, args=(queue, kittidir, sequence, stride, 0))
    reader.start()

    for step in count(start=1):
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.as_tensor(image, device='cuda').permute(2,0,1)
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        if show_img:
            show_image(image, 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[-2], wd=image.shape[-1], viz=viz)

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--kittidir', type=Path, default="datasets/KITTI")
    parser.add_argument('--backend_thresh', type=float, default=32.0)
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

    kitti_scenes = [f"{i:02d}" for i in range(11)]

    results = {}
    for scene in kitti_scenes:
        groundtruth = args.kittidir / "dataset" / "poses" / f"{scene}.txt"
        poses_ref = file_interface.read_kitti_poses_file(groundtruth)
        print(f"Evaluating KITTI {scene} with {poses_ref.num_poses // args.stride} poses")

        scene_results = []
        for trial_num in range(args.trials):
            traj_est, timestamps = run(cfg, args.network, args.kittidir, scene, args.stride, args.viz, args.show_img)

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=timestamps * args.stride)

            traj_ref = PoseTrajectory3D(
                positions_xyz=poses_ref.positions_xyz,
                orientations_quat_wxyz=poses_ref.orientations_quat_wxyz,
                timestamps=np.arange(poses_ref.num_poses, dtype=np.float64))

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            ate_score = result.stats["rmse"]

            if args.plot:
                plot_trajectory(traj_est, traj_ref, f"kitti sequence {scene} Trial #{trial_num+1}", f"trajectory_plots/kitti_seq{scene}_trial{trial_num+1:02d}.pdf", align=True, correct_scale=True)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/KITTI_{scene}.txt", traj_est)
                # file_interface.write_kitti_poses_file(f"saved_trajectories/{scene}.txt", traj_est) # standard kitti format

            scene_results.append(ate_score)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))

    

    

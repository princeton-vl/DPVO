import datetime
import glob
import os
import os.path as osp
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.data_readers.tartan import test_split as val_split
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def video_iterator(imagedir, ext=".png", preload=True):
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))

    data_list = []
    for imfile in sorted(imfiles)[::STRIDE]:
        image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image, intrinsics))

    for (image, intrinsics) in data_list:
        yield image.cuda(), intrinsics.cuda()

@torch.no_grad()
def run(imagedir, cfg, network, viz=False, show_img=False):
    slam = DPVO(cfg, network, ht=480, wd=640, viz=viz)

    for t, (image, intrinsics) in enumerate(video_iterator(imagedir)):
        if show_img:
            show_image(image, 1)
        
        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    return slam.terminate()


def ate(traj_ref, traj_est):
    assert isinstance(traj_ref, PoseTrajectory3D)
    assert isinstance(traj_est, PoseTrajectory3D)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]


@torch.no_grad()
def evaluate(config, net, split="validation", trials=1, plot=False, save=False):

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    if not os.path.isdir("TartanAirResults"):
        os.mkdir("TartanAirResults")

    scenes = test_split if split=="test" else val_split

    results = {}
    all_results = []
    for i, scene in enumerate(scenes):

        results[scene] = []
        for j in range(trials):

            # estimated trajectory
            if split == 'test':
                scene_path = os.path.join("datasets/mono", scene)
                traj_ref = osp.join("datasets/mono", "mono_gt", scene + ".txt")
            
            elif split == 'validation':
                scene_path = os.path.join("datasets/TartanAir", scene, "image_left")
                traj_ref = osp.join("datasets/TartanAir", scene, "pose_left.txt")

            # run the slam system
            traj_est, tstamps = run(scene_path, config, net, viz=False, show_img=False)

            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE, PERM]

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=tstamps)

            traj_ref = PoseTrajectory3D(
                positions_xyz=traj_ref[:,:3],
                orientations_quat_wxyz=traj_ref[:,3:],
                timestamps=tstamps)

            # do evaluation
            ate_score = ate(traj_ref, traj_est)
            all_results.append(ate_score)
            results[scene].append(ate_score)

            if plot:
                scene_name = '_'.join(scene.split('/')[1:]).title() if split == 'validation' else scene
                Path("trajectory_plots").mkdir(exist_ok=True)
                plot_trajectory(traj_est, traj_ref, f"TartanAir {scene_name.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/TartanAir_{scene_name}_Trial{j+1:02d}.pdf", align=True, correct_scale=True)

            if save:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/TartanAir_{scene_name}_Trial{j+1:02d}.txt", traj_est)

        print(scene, sorted(results[scene]))

    results_dict = dict([("Tartan/{}".format(k), np.median(v)) for (k, v) in results.items()])

    # write output to file with timestamp
    with open(osp.join("TartanAirResults", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)

    ates = list(all_results)
    results_dict["AUC"] = np.maximum(1 - np.array(ates), 0).mean()
    results_dict["AVG"] = np.mean(xs)

    return results_dict


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=18.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(1234)

    if args.id >= 0:
        scene_path = os.path.join("datasets/mono", test_split[args.id])
        traj_est, tstamps = run(scene_path, cfg, args.weights, viz=args.viz, show_img=args.show_img)

        traj_ref = osp.join("datasets/mono", "mono_gt", test_split[args.id] + ".txt")
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE,[1, 2, 0, 4, 5, 3, 6]]

        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:,:3],
            orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
            timestamps=tstamps)

        traj_ref = PoseTrajectory3D(
            positions_xyz=traj_ref[:,:3],
            orientations_quat_wxyz=traj_ref[:,3:],
            timestamps=tstamps)

        # do evaluation
        print(ate(traj_ref, traj_est))

    else:
        results = evaluate(cfg, args.weights, split=args.split, trials=args.trials, plot=args.plot, save=args.save_trajectory)
        for k in results:
            print(k, results[k])

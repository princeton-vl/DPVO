import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp

import torch
from dpvo.dpvo import DPVO
from dpvo.utils import Timer
from dpvo.config import cfg

from dpvo.data_readers.tartan import test_split as val_split

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
def run(imagedir, cfg, network, viz=False):
    slam = DPVO(cfg, network, ht=480, wd=640, viz=viz)

    for t, (image, intrinsics) in enumerate(video_iterator(imagedir)):
        if viz: 
            show_image(image, 1)
        
        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    return slam.terminate()


def ate(traj_ref, traj_est, timestamps):
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=timestamps)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]


@torch.no_grad()
def evaluate(config, net, split="validation", trials=1):

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
            traj_est, tstamps = run(scene_path, config, net)

            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE, PERM]

            # do evaluation
            all_results.append(ate(traj_ref, traj_est, tstamps))
            results[scene].append(ate(traj_ref, traj_est, tstamps))

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
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(1234)

    if args.id >= 0:
        scene_path = os.path.join("datasets/mono", test_split[args.id])
        traj_est, tstamps = run(scene_path, cfg, args.weights, viz=args.viz)

        traj_ref = osp.join("datasets/mono", "mono_gt", test_split[args.id] + ".txt")
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE,[1, 2, 0, 4, 5, 3, 6]]

        # do evaluation
        print(ate(traj_ref, traj_est, tstamps))

    else:
        results = evaluate(cfg, args.weights, split=args.split, trials=args.trials)
        for k in results:
            print(k, results[k])

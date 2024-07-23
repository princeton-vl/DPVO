import cv2
import numpy as np
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from plyfile import PlyElement, PlyData

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, end=-1, assumed_fov_degrees: float = 90.0, viz=False, timeit=False, save_reconstruction=False):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip, end))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip, end, assumed_fov_degrees))

    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    reader.join()

    if save_reconstruction:
        points = slam.points_.cpu().numpy()[:slam.m]
        colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
        points = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                          dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        el = PlyElement.describe(points, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})
        return slam.terminate(), PlyData([el], text=True)

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--assumed_fov_degrees', type=float, default=90.0)
    parser.add_argument('--buffer', type=int, default=2048)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_reconstruction', type=str, default="")
    parser.add_argument('--save_trajectory', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BUFFER_SIZE = args.buffer

    # print("Running with config...")
    # print(cfg)

    pred_traj = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.end, args.assumed_fov_degrees, args.viz, args.timeit, args.save_reconstruction)
    if not args.name:
        name = Path(args.imagedir).stem
    else:
        name = args.name

    if args.save_reconstruction:
        pred_traj, ply_data = pred_traj
        path = os.path.join(args.save_reconstruction, f"{name}.ply")
        ply_data.write(path)
        # print(f"Saved {path}")

    if args.save_trajectory:
        os.makedirs(args.save_trajectory, exist_ok=True)
        path = os.path.join(args.save_trajectory, f"{name}.txt")
        save_trajectory_tum_format(pred_traj, path)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(pred_traj, title=f"DPVO Trajectory Prediction for {name}", filename=f"trajectory_plots/{name}.pdf")


        


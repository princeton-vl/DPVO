from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from plyfile import PlyData, PlyElement


def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    assert isinstance(pred_traj, PoseTrajectory3D)

    if gt_traj is not None:
        assert isinstance(gt_traj, PoseTrajectory3D)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xz # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

def save_output_for_COLMAP(name: str, traj: PoseTrajectory3D, points: np.ndarray, colors: np.ndarray, fx, fy, cx, cy, H=480, W=640):
    """ Saves the sparse point cloud and camera poses such that it can be opened in COLMAP """

    colmap_dir = Path(name)
    colmap_dir.mkdir(exist_ok=True)
    scale = 10 # for visualization

    # images
    images = ""
    traj = PoseTrajectory3D(poses_se3=list(map(np.linalg.inv, traj.poses_se3)), timestamps=traj.timestamps)
    for idx, (x,y,z), (qw, qx, qy, qz) in zip(range(1,traj.num_poses+1), traj.positions_xyz*scale, traj.orientations_quat_wxyz):
        images += f"{idx} {qw} {qx} {qy} {qz} {x} {y} {z} 1\n\n"
    (colmap_dir / "images.txt").write_text(images)

    # points
    points3D = ""
    colors_uint = (colors * 255).astype(np.uint8).tolist()
    for i, (p,c) in enumerate(zip((points*scale).tolist(), colors_uint), start=1):
        points3D += f"{i} " + ' '.join(map(str, p + c)) + " 0.0 0 0 0 0 0 0\n"
    (colmap_dir / "points3D.txt").write_text(points3D)

    # camera
    (colmap_dir / "cameras.txt").write_text(f"1 PINHOLE {H} {W} {fx} {fy} {cx} {cy}")
    print(f"Saved COLMAP-compatible reconstruction in {colmap_dir.resolve()}")

def save_ply(name: str, points: np.ndarray, colors: np.ndarray):
    points_ply = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                        dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(points_ply, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})
    PlyData([el], text=True).write(f"{name}.ply")
    print(f"Saved {name}.ply")

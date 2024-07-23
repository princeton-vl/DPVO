import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(
    queue,
    imagedir: str,
    calib: str | None = None,
    stride: int = 1,
    skip: int = 0,
    end: int = -1,
):
    """ video generator """

    K = None
    if calib is not None:
        calib = np.loadtxt(calib, delimiter=" ")

    cap = cv2.VideoCapture(imagedir)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end < 0:
        end += total_frames + 1

    t = 0
    c = 0

    for _ in range(skip):
        c += 1
        ret, image = cap.read()

    while True:
        for _ in range(stride):
            c += 1
            ret, image = cap.read()
            if not ret:
                break

        if not ret or c >= end:  # TODO: is >= or >?
            break
        
        if calib is None:
            h, w = image.shape[:2]
            # assume 90 degree fov and 0 distortion TODO: pass in with arg
            calib = np.array([w / 2, w / 2, w / 2, h / 2, 0, 0, 0, 0])
        fx, fy, cx, cy = calib[:4]

        if K is None:
            K = np.eye(3)
            K[0,0] = fx
            K[0,2] = cx
            K[1,1] = fy
            K[1,2] = cy

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()


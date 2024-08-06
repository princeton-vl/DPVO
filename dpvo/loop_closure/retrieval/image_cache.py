import os
from multiprocessing import Pool
from shutil import copytree
from tempfile import TemporaryDirectory

import cv2
import kornia as K
import numpy as np
from einops import asnumpy, parse_shape, rearrange

IMEXT = '.jpeg'
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
BLANK = np.zeros((500,500,3), dtype=np.uint8)

class ImageCache:

    def __init__(self):
        self.image_buffer = {}
        self.tmpdir = TemporaryDirectory()
        self.stored_indices = np.zeros(100000, dtype=bool)
        self.writer_pool = Pool(processes=1)
        self.write_result = self.writer_pool.apply_async(cv2.imwrite, [f"{self.tmpdir.name}/warmup.png", BLANK, JPEG_QUALITY])
        self._wait()

    def __call__(self, image, n):
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, '_ _ RGB') == dict(RGB=3)
        self.image_buffer[n] = image

    def _wait(self):
        """ Wait until the previous image is finished writing """
        self.write_result.wait()

    def _write_image(self, i):
        """ Save the image to disk (asynchronously) """
        img = self.image_buffer.pop(i)
        filepath = f"{self.tmpdir.name}/{i:08d}{IMEXT}"
        assert not os.path.exists(filepath)
        self._wait()
        self.write_result = self.writer_pool.apply_async(cv2.imwrite, [filepath, img, JPEG_QUALITY])

    def load_frames(self, idxs, device='cuda'):
        self._wait()
        assert np.all(self.stored_indices[idxs])
        frame_list = [f"{self.tmpdir.name}/{i:08d}{IMEXT}" for i in idxs]
        assert all(map(os.path.exists, frame_list))
        image_list = [cv2.imread(f) for f in frame_list]
        return K.utils.image_list_to_tensor(image_list).to(device=device)

    def keyframe(self, k):
        tmp = dict(self.image_buffer)
        self.image_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n-1) if (n > k) else n
                self.image_buffer[key] = v

    def save_up_to(self, c):
        """ Pop images from the buffer and write them to disk"""
        for n in list(self.image_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                self._write_image(n)
                self.stored_indices[n] = True

    def close(self):
        self._wait()
        # copytree(self.tmpdir.name, '/tmp/temp')
        self.tmpdir.cleanup()
        # os.rename('/tmp/temp', self.tmpdir.name)
        self.writer_pool.close()
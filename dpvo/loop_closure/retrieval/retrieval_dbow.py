import os
import time
from multiprocessing import Process, Queue, Value
import numpy as np
from einops import parse_shape

try:
    import dpretrieval
    dpretrieval.DPRetrieval
except:
    raise ModuleNotFoundError("Couldn't load dpretrieval. It may not be installed.")


NMS = 50 # Slow motion gets removed from keyframes anyway. So this is really the keyframe distance

RAD = 50

def _dbow_loop(in_queue, out_queue, vocab_path, ready):
    """ Run DBoW retrieval """
    dbow = dpretrieval.DPRetrieval(vocab_path, 50)
    ready.value = 1
    while True:
        n, image = in_queue.get()
        dbow.insert_image(image)
        q = dbow.query(n)
        out_queue.put((n, q))

class RetrievalDBOW:

    def __init__(self, vocab_path="ORBvoc.txt"):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError("""Missing the ORB vocabulary. Please download and un-tar it from """
                                  """https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Vocabulary/ORBvoc.txt.tar.gz"""
                                  f""" and place it in DPVO/""")

        # Store a record of saved and unsaved images
        self.image_buffer = {}
        self.stored_indices = np.zeros(100000, dtype=bool)

        # Keep track of detected and closed loops
        self.prev_loop_closes = []
        self.found = []

        # Run DBoW in a separate process
        self.in_queue = Queue(maxsize=20)
        self.out_queue = Queue(maxsize=20)
        ready = Value('i', 0)
        self.proc = Process(target=_dbow_loop, args=(self.in_queue, self.out_queue, vocab_path, ready))
        self.proc.start()
        self.being_processed = 0
        while not ready.value:
            time.sleep(0.01)

    def keyframe(self, k):
        """ Once we keyframe an image, we can safely cache all images
         before & including it """
        tmp = dict(self.image_buffer)
        self.image_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n-1) if (n > k) else n
                self.image_buffer[key] = v

    def save_up_to(self, c):
        """ Add frames to the image-retrieval database """
        for n in list(self.image_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                img = self.image_buffer.pop(n)
                self.in_queue.put((n, img))
                self.stored_indices[n] = True
                self.being_processed += 1

    def confirm_loop(self, i, j):
        """ Record the loop closure so we don't have redundant edges"""
        assert i > j
        self.prev_loop_closes.append((i, j))

    def _repetition_check(self, idx, num_repeat):
        """ Check that we've retrieved <num_repeat> consecutive frames """
        if (len(self.found) < num_repeat):
            return
        latest = self.found[-num_repeat:]
        (b, _), (i, j), _ = latest
        if (1 + idx - b) == num_repeat:
            return (i, max(j,1)) # max(j,1) is to avoid centering the triplet on 0

    def detect_loop(self, thresh, num_repeat=1):
        """ Keep popping off the queue until the it is empty
         or we find a positive pair """
        while self.being_processed > 0:
            x = self._detect_loop(thresh, num_repeat)
            if x is not None:
                return x

    def _detect_loop(self, thresh, num_repeat=1):
        """ Pop retrived pairs off the queue. Return if they have non-trivial score """
        assert self.being_processed > 0
        i, (score, j, _) = self.out_queue.get()
        self.being_processed -= 1
        if score < thresh:
            return
        assert i > j

        # Ensure that this edge is not redundant
        dists_sq = [(np.square(i - a) + np.square(j - b)) for a,b in self.prev_loop_closes]
        if min(dists_sq, default=np.inf) < np.square(NMS):
            return

        # Add this frame pair to the list of retrieved matches
        self.found.append((i, j))

        # Check that we've retrieved <num_repeat> consecutive frames
        return self._repetition_check(i, num_repeat)

    def __call__(self, image, n):
        """ Store the image into the frame buffer """
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, '_ _ RGB') == dict(RGB=3)
        self.image_buffer[n] = image
    
    def close(self):
        self.proc.terminate()
        self.proc.join()

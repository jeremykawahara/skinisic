import numpy as np
from skinisic.cython_converter import superpixel_to_seg


def compute_superseg(superindexes, json_arr):
    H = superindexes.shape[0]
    W = superindexes.shape[1]
    K = json_arr.shape[0]
    superseg = np.zeros(shape=(H, W, K), dtype=np.float32)
    superpixel_to_seg(H, W, K, superseg, superindexes, json_arr)  # superseg pass by ref.

    return superseg

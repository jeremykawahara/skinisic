import numpy as np


def rgb2index(rgb_pixel):
    """Converts the RGB superpixels into a indexes.

    Code is from:
    https://challenge.kitware.com/#phase/56674518cad3a56fac78678c
    """
    return \
        (rgb_pixel[..., 0].astype(np.uint64)) + \
        (rgb_pixel[..., 1].astype(np.uint64) << np.uint64(8)) + \
        (rgb_pixel[..., 2].astype(np.uint64) << np.uint64(16))

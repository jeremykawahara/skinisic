import numpy as np
from skinisic.converter import compute_superseg


def rgb2index(rgb_pixel):
    """Converts the RGB superpixels into a indexes.

    Code is from:
    https://challenge.kitware.com/#phase/56674518cad3a56fac78678c
    """
    return \
        (rgb_pixel[..., 0].astype(np.uint64)) + \
        (rgb_pixel[..., 1].astype(np.uint64) << np.uint64(8)) + \
        (rgb_pixel[..., 2].astype(np.uint64) << np.uint64(16))


def superpixels2superseg(superpixels, json_dict, dermoscopic_labels):
    # Convert so each superpixel has a unique index.
    superindexes = rgb2index(superpixels)

    # Convert the dictonary to an array.
    json_array = np.asarray([json_dict[label] for label in dermoscopic_labels], dtype=np.float32)
    # print(json_array.shape)

    # Convert the superpixel labels to a superpixel segmentation,
    # where each pixel in each of the 4 dermoscopic features are labeled.
    superseg = compute_superseg(superindexes, json_array)
    # print(superseg.shape)
    return superseg

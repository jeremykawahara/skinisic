import numpy as np
import scipy


def imresize3d(x, target_dims, interp='nearest', mode='F', dtype=np.float32):
    """Resize each channel within the third dimension.

        `scipy.misc.imresize` does not work if the number of channels is > 3.
    """
    # https://github.com/scipy/scipy/issues/6417
    if len(x.shape) is not 3:
        raise ValueError("Error: `len(x.shape)` should be 3 but is {}".format(len(x.shape)))

    x_resize = np.zeros(shape=(target_dims[0], target_dims[1], x.shape[2]), dtype=dtype)
    for idx in range(x.shape[2]):
        x_resize[:, :, idx] = scipy.misc.imresize(x[:, :, idx], target_dims, interp, mode)

    return x_resize

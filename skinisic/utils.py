import numpy as np
from PIL import Image


def imresize3d(x, target_dims, dtype=np.float32):
    """Resize each channel within the third dimension.

        `scipy.misc.imresize` does not work if the number of channels is > 3.
    """
    # https://github.com/scipy/scipy/issues/6417
    if len(x.shape) is not 3:
        raise ValueError("Error: `len(x.shape)` should be 3 but is {}".format(len(x.shape)))

    x_resize = np.zeros(shape=(target_dims[0], target_dims[1], x.shape[2]), dtype=dtype)
    for idx in range(x.shape[2]):
        # scipy no longer supports `imresize`
        # https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html
        # resized_img = scipy.misc.imresize(x[:, :, idx], target_dims, interp, mode)
        # For some reason, the dimensions are switched here.
        resized_img = Image.fromarray(x[:,:,idx]).resize((target_dims[1], target_dims[0]))
        x_resize[:, :, idx] = resized_img

    return x_resize

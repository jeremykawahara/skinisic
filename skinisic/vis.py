import matplotlib.pyplot as plt
import numpy as np


def plot_img_preds(img, y_pred, labels, figsize=(16, 4), fontsize=12, show_colorbar=True):
    n_rows = 1
    n_cols = len(labels) + 1

    plt.figure(figsize=figsize)
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(img)
    plt.title('Image', fontsize=fontsize)
    plt.axis('off')
    if show_colorbar:
        plt.colorbar(fraction=0.04, pad=0.04)

    for c, label in enumerate(labels):
        plt.subplot(n_rows, n_cols, c + 2)
        plt.imshow(y_pred[:, :, c])
        plt.title(label, fontsize=fontsize)
        plt.axis('off')
        if show_colorbar:
            plt.colorbar(fraction=0.04, pad=0.04)


def overlay_image(img, overlay, channel_to_overlay=2, scale_overlay=255):
    """Modified the pixels in `img` in-place with the `overlay`.

    Scales the image pixels within a specific channel by the given overlay.

    Args:
        img (array of size HxWxC): the image with three channels to modify
        overlay (array of size HxW): a mask to overlay the image with. Assumes each element is between 0 and 1.
        channel_to_overlay (int): the image channel to modify
        scale_overlay (int): scales the overlay image to match the input image

    Returns: Nothing. `img` is modified in place. Make a copy if you wish to perserve the image.

    """
    # Modify the image -> pass-by-reference.
    # The max() function prevents the image pixels from going out of bounds.
    img[:, :, channel_to_overlay] = np.maximum(
        img[:, :, channel_to_overlay],
        overlay * scale_overlay
    )


def plot_overlay_image(img, overlays, channels_to_overlay=(2, 1, 0), scale_overlay=255):
    """Plot the image with the pixel values in each channel overlaid with the values in `overlays`.

    Args:
        img (HxWxC array): the image to overlay
        overlays (list of HxW arrays): the specific overlays (assumes array values between 0 and 1).
        channels_to_overlay (tuple): order of the channels to overlay (cannot exceed the number of image channels C).
        scale_overlay (int): the amount to scale the overlay.

    """
    # Make a copy of the image since we modify the pixels.
    mod_img = np.copy(img)

    if len(overlays) > 3:
        raise ValueError("Can only overlay three channels, thus len(overlays)<=3)")

    for overlay, channel_to_overlay in zip(overlays, channels_to_overlay):
        overlay_image(mod_img, overlay, channel_to_overlay=channel_to_overlay,
                      scale_overlay=scale_overlay)

    plt.imshow(mod_img)
    plt.axis('off')

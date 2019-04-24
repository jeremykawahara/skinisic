import matplotlib.pyplot as plt


def plot_img_preds(img, y_pred, labels, figsize=(16,4), fontsize=12, show_colorbar=True):
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

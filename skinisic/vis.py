import matplotlib.pyplot as plt


def plot_img_preds(img, y_pred, labels):
    n_rows = 1
    n_cols = len(labels) + 1

    plt.figure(figsize=(16, 4))
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.colorbar(fraction=0.04, pad=0.04)

    for c, label in enumerate(labels):
        plt.subplot(n_rows, n_cols, c + 2)
        plt.imshow(y_pred[:, :, c])
        plt.title(label)
        plt.colorbar(fraction=0.04, pad=0.04)
        plt.axis('off')

import sys
import numpy as np
import matplotlib
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from skinisic.kerasutils import load_preprocess_image
from skinisic.kerasmodels import fcn_vgg_bottomheavy
from skinisic.vis import plot_img_preds

# Where the CNN weights are stored. e.g,. 'data/isic2017-part2_vgg_f1-batch_aug.h5'
model_path = sys.argv[1]

# The channel order that the CNN outputs the dermoscopic criteria.
dermoscopic_labels = ['pigment_network', 'negative_network', 'milia_like_cyst', 'streaks']
target_size = (336, 336, 3)  # Image size the CNN expects.

# Create the model and the trained weights.
model = fcn_vgg_bottomheavy(target_size, nb_labels=len(dermoscopic_labels))
model.load_weights(model_path)

# Original image before resizing (provided sample image from ISIC test set)
img_path = 'notebooks/data/ISIC-2017_Test_v2_Data/ISIC_0012758.jpg'
img = np.asarray(image.load_img(img_path))

# Pre-process the image for the CNN.
pre_img = load_preprocess_image(img_path, preprocess_function=preprocess_input, target_size=target_size)
y_pred = np.squeeze(model.predict(pre_img))  # CNN's forward pass.

# Plot the image and predicted dermoscopic criteria.
plot_img_preds(img, y_pred, labels=dermoscopic_labels)
matplotlib.pyplot.show()

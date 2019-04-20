import numpy as np
from keras.preprocessing import image


def load_preprocess_image(image_name, preprocess_function, target_size):
    img = image.load_img(image_name, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_function(x)
    return x

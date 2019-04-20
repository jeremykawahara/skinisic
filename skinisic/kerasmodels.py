import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import (
    Input,
    Convolution2D,
    Lambda,
    concatenate,
    LeakyReLU,
)


def fcn_vgg_bottomheavy(input_shape, nb_labels, leaky_alpha=0, freeze_base=False):
    """Fully convolutional neural network with independent output labels.

    Original model inspired by:
    https://github.com/azavea/raster-vision/blob/107e824849fb2e50dfc2644d9a24154a888ca468/src/model_training/models/fcn_vgg.py

    Args:
        input_shape: Dimensions of input image.
        nb_labels: Number of output labels.
        leaky_alpha: The leaky_alpha rate for ReLU (set to 0 for regular ReLU).
        freeze_base: If True then the parameters of the base VGG are not updated.

    Returns:
        The CNN model.

    """

    input_tensor = Input(shape=input_shape)
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    block1 = base_model.get_layer('block1_conv2').output
    block2 = base_model.get_layer('block2_conv2').output
    block3 = base_model.get_layer('block3_conv3').output
    block4 = base_model.get_layer('block4_conv3').output

    block4 = Convolution2D(64, (1, 1))(block4)
    block4 = LeakyReLU(alpha=leaky_alpha)(block4)

    block5 = base_model.get_layer('block5_conv3').output
    block5 = Convolution2D(32, (1, 1))(block5)
    block5 = LeakyReLU(alpha=leaky_alpha)(block5)

    block5_pool = base_model.get_layer('block5_pool').output
    block5_pool = Convolution2D(32, (1, 1))(block5_pool)
    block5_pool = LeakyReLU(alpha=leaky_alpha)(block5_pool)

    def resize_bilinear(block):
        return tf.image.resize_bilinear(block, [input_shape[0], input_shape[1]])

    block2_full = Lambda(resize_bilinear)(block2)
    block3_full = Lambda(resize_bilinear)(block3)
    block4_full = Lambda(resize_bilinear)(block4)
    block5_full = Lambda(resize_bilinear)(block5)
    block5_pool_full = Lambda(resize_bilinear)(block5_pool)

    x = concatenate([block1, block2_full, block3_full, block4_full, block5_full, block5_pool_full])

    x = Convolution2D(nb_labels, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model

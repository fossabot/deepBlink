"""Fully convolutional networks with / without dropout."""

import tensorflow as tf

EPS = 1.001e-5


def conv_block(x: tf.keras.layers.Layer, growth_rate: float):
    """A building block for a dense block.

    Arguments:
        x: Input tensor.
        growth_rate: Growth rate at dense layers.
    Returns:
        Output tensor for the block.
    """
    x1 = tf.keras.layers.BatchNormalization(epsilon=EPS)(x)
    x1 = tf.keras.layers.Activation("relu")(x1)
    x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1, use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization(epsilon=EPS)(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)(x1)
    x = tf.keras.layers.Concatenate()([x, x1])
    return x


def dense_block(x: tf.keras.layers.Layer, blocks: int):
    """A dense block.

    Arguments:
        x: Input tensor.
        blocks: The number of building blocks.
    Returns:
        Output tensor for the block.
    """
    for _ in range(blocks):
        x = conv_block(x, 32)
    return x


def transition_block(x: tf.keras.layers.Layer, reduction: float):
    """A transition block.

    Arguments:
        x: Input tensor.
        reduction: Compression rate at transition layers.
    Returns:
        Output tensor for the block.
    """
    x = tf.keras.layers.BatchNormalization(epsilon=EPS)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(
        int(tf.keras.backend.int_shape(x)[1] * reduction), 1, use_bias=False,
    )(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2)(x)
    return x


def dense_net():
    """Dense network implementation.

    Adapted from tensorflows internal tf.keras.applications.DenseNet121.

    ref: https://arxiv.org/abs/1608.06993
    """

    inputs = tf.keras.layers.Input(shape=(512, 512, 1))

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = tf.keras.layers.Conv2D(64, 7, strides=1, use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization(epsilon=EPS)(x)
    # x = tf.keras.layers.Activation("relu")(x)
    # x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    # x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    blocks = [6, 12, 24, 16]
    x = dense_block(x, blocks[0])
    x = transition_block(x, 0.5)
    x = dense_block(x, blocks[1])
    x = transition_block(x, 0.5)
    x = dense_block(x, blocks[2])
    # x = transition_block(x, 0.5)
    # x = dense_block(x, blocks[3])

    x = tf.keras.layers.BatchNormalization(epsilon=EPS)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs, x)

    return model

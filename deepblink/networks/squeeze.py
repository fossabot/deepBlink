"""Squeeze and excitation block implementation."""

import tensorflow as tf


# TODO: add input and type checks
def squeeze_block(x: tf.keras.layers.Layer, ratio: int = 8):
    """Squeeze and excitation block.

    ref: https://arxiv.org/pdf/1709.01507.pdf.

    Args:
        x: Input tensor.
        ratio: Number of output filters.
    """
    filters = x.shape[-1]

    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x1 = tf.keras.layers.Dense(max(filters // ratio, 1))(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)
    x1 = tf.keras.layers.Dense(filters)(x1)
    x1 = tf.keras.layers.Activation("sigmoid")(x1)
    x = tf.keras.layers.Multiply()([x1, x])
    return x

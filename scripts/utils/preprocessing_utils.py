#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import tensorflow as tf
import numpy as np


def tf_log10(x):
    """
    Compute the base-10 logarithm of a TensorFlow tensor element-wise.

    This function takes a TensorFlow tensor 'x' as input and calculates
    the base-10 logarithm of each element of the tensor element-wise.

    Parameters:
        x (tf.Tensor): The input TensorFlow tensor.

    Returns:
        tf.Tensor: A new TensorFlow tensor containing the base-10 logarithm of
                   each element of the input tensor 'x'.
    """

    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def ds9_scaling(img, a=tf.cast(500, tf.float64), offset=0.):
    """
    Apply DS9-like scaling to a TensorFlow image tensor.

    This function takes a TensorFlow image tensor 'img' and performs DS9-like
    scaling on each element of the tensor element-wise.
    The scaling is defined by the formula:
    img = log10(a * img + 1) / log10(a + 1),
    where 'a' is a scaling parameter and
    'offset' is an optional offset value.

    Parameters:
        img (tf.Tensor): The input TensorFlow image tensor.
        a (tf.float64, optional): The scaling parameter 'a' used in the
                                  DS9-like scaling.
                                  Default value is 500
        offset (float, optional): An optional offset value to be subtracted
                                  after the scaling. Default value is 0.0.

    Returns:
        tf.Tensor: A new TensorFlow tensor containing the DS9-like
                   scaled values of each element of the input image
                   tensor 'img'.
    """

    img = tf.cast(img, tf.float64)
    img = tf_log10(a * img + 1) / tf_log10(a + 1)  # - offset
    nan_free_img = tf.where(tf.math.is_nan(img), tf.cast(0, tf.float64), img)
    return nan_free_img


def z_scale_gen(x, a=500):

    x = np.log10(a * x + 1) / np.log10(a + 1)

    return x


def preprocessing_sinh(x):
    x_prep = tf.math.asinh(x)
    return x_prep


def preprocessing_zscale(x):
    x_prep = ds9_scaling(x)
    return x_prep

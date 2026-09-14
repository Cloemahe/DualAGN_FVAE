#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import tensorflow as tf
import numpy as np


def mask_fn(y_true, zero_value):
    """
    Compute a mask of the input images corresponding to the condition light/not light : one mask for each image of the batch with three filters (?)
    """

    min_image = 1.2  # Preprocessing = arcsin hyp

    mask = tf.zeros_like(y_true)
    mask = tf.where(y_true > min_image, 1., zero_value)

    return mask


def create_asymmetric_ring_mask(R, alpha, zeta, intensity, shape):
    # Create coordinate grid centered on the image
    y, x = np.indices(shape)
    cy, cx = shape[0] // 2, shape[1] // 2  # Center of the image

    # Compute the radial distance from the center
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Max radius is the diagonal length of the image (max possible distance from the center)
    max_radius = np.sqrt((shape[0] // 2)**2 + (shape[1] // 2)**2)

    # Create the mask : 0 at the center, increases steeply, then decreases smoothly
    mask = tf.zeros_like(radius)

    # Steep increase from 0 to 1 within radius R
    m_inf = tf.where(radius <= R, 1., 0.)

    radius_inf = radius * m_inf
    radius_inf = (radius_inf / R)**alpha

    # Smooth decrease from 1 back to 0 beyond radius R
    m_sup = tf.where(radius > R, 1., 0.)

    radius_sup = radius * m_sup
    radius_sup = (1 - (radius_sup - R) / (max_radius - R)) ** zeta

    # Apply limitations to mask
    mask = tf.where(radius > R, radius_sup, radius_inf)
    mask = tf.where(mask >= 0, mask, 0.)

    mask *= intensity
    mask += 1

    return mask  # Should be a Tensor of shape : (b_size, 256, 256)


def fast_fourier_shift(tensor):  # Tensor should be of shape (b_size, 256, 256) type : complex

    img = tf.complex(tensor, 0.)  # Complex tensor

    fft = tf.signal.fft2d(img)
    fft = tf.signal.fftshift(fft)  # In : complex tensor, Out : complex tensor

    return fft


def transform_from_fft(tensor, mask):

    fft = fast_fourier_shift(tensor)

    mask = tf.complex(mask, 0.)

    weighted_fft = tf.multiply(fft, mask)

    return weighted_fft  # Should be a tensor of shape : (b_size, 256, 256) type : complex

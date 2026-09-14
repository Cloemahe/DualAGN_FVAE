#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import numpy as np


def gini_coeff(array):
    """
    The input im here is a standard Numpy array : the selection of the image from the batch and its numpyisation should be done before.
    """

    array = array.flatten()  # all values are treated equally, arrays must be 1d

    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative

    array += 0.0000001  # values cannot be 0

    array = np.sort(array)  # values must be sorted

    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements

    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient


def gini_array(image, b_size):
    """
    Here the image is a batch 3D 256 x 256 pixels. It is therefore necessary to convert them to numpy array. We only apply the Gini coefficient to one of the 3 filters.
    """
    im = image.numpy()
    gini_range = []

    for j in range(b_size):

        im_to_flatten = im[j, :, :, 0]  # For each image of the batch

        gini = gini_coeff(im_to_flatten)

        if not gini > 1.:
            gini_range.append(gini)

    mean_gini = np.mean(gini_range)

    return gini_range, mean_gini

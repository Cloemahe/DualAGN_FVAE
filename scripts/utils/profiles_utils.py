#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules
import numpy as np

# Scikit-image
from skimage import filters, measure, morphology
from skimage.filters import threshold_otsu
from skimage.measure import profile_line

'''
Computes the profiles lines on images
'''

# Extend to border


def extend_to_border(x1, y1, dx, dy, width, height):
    """Finds the intersection points of the line with the image borders."""
    points = []

    # Avoid division by zero
    if dx != 0:
        # Left border (x=0)
        t = -x1 / dx
        y = y1 + t * dy
        if 0 <= y < height:
            points.append((0, int(y)))

        # Right border (x=width-1)
        t = (width - 1 - x1) / dx
        y = y1 + t * dy
        if 0 <= y < height:
            points.append((width - 1, int(y)))

    if dy != 0:
        # Top border (y=0)
        t = -y1 / dy
        x = x1 + t * dx
        if 0 <= x < width:
            points.append((int(x), 0))

        # Bottom border (y=height-1)
        t = (height - 1 - y1) / dy
        x = x1 + t * dx
        if 0 <= x < width:
            points.append((int(x), height - 1))

    return points


# Measure regions

def regions_img(img, smooth=True):

    if smooth:
        img_smooth = filters.gaussian(img, sigma=3)
    else:
        img_smooth = img

    threshold_value = threshold_otsu(img_smooth)

    binary_image = img_smooth > threshold_value
    binary_cleaned = morphology.remove_small_objects(binary_image, min_size=500)

    label_img = measure.label(binary_cleaned)

    regions = measure.regionprops(label_img)  # Sometimes no regions detected ?

    if len(regions) >= 1:
        validity = True
    else:
        validity = False

    return regions, validity


# Profile dual : compute the profile between the two galaxies detected

def profile_dual(img, regions, smooth=True):

    img_very_smooth = filters.gaussian(img, sigma=5)  # Image very smooth to obtain good profiles

    if len(regions) == 1:  # if there is only one spot
        x1, y1 = img.shape[0] // 2, img.shape[1] // 2
        x2, y2 = regions[0].centroid
    else:  # If there are more than one detected spot
        x1, y1 = regions[0].centroid
        x2, y2 = regions[1].centroid

    # Compute the direction of the line
    dx = x2 - x1
    dy = y2 - y1

    # Get extended points
    border_points = extend_to_border(x1, y1, dx, dy, img.shape[0], img.shape[1])  # Relie les points aux bords de l'image
    # Select the two farthest points as the new start and end

    if len(border_points) >= 2:
        (x_start, y_start), (x_end, y_end) = border_points[:2]
    else:
        raise ValueError("Failed to find valid intersections with image borders.")

    if smooth:
        profile = profile_line(img_very_smooth, (x_start, y_start), (x_end, y_end))
    else:
        profile = profile_line(img, (x_start, y_start), (x_end, y_end))

    return profile, x_start, x_end, y_start, y_end


# Profile maj axis : compute the profile along the major axis of the largest galaxy detected

def profile_maj_axis(img, regions, smooth=True):

    img_very_smooth = filters.gaussian(img, sigma=5)  # Image very smooth to obtain good profiles

    sizes = []
    for region in regions:
        sizes.append(region.area)

    size_max = np.max(sizes)
    i = sizes.index(size_max)

    maj_axis = regions[i].axis_major_length
    semi_maj_axis = maj_axis // 2

    theta = regions[i].orientation

    cy, cx = regions[i].centroid  # Centre des tâches trouvées

    x_m = cx - semi_maj_axis * np.sin(theta)
    y_m = cy - semi_maj_axis * np.cos(theta)

    d_x = x_m - cx
    d_y = y_m - cy

    border_points = extend_to_border(cy, cx, d_y, d_x, img.shape[0], img.shape[1])  # Relie les points aux bords de l'image

    if len(border_points) >= 2:
        (x_start, y_start), (x_end, y_end) = border_points[:2]
    else:
        raise ValueError("Failed to find valid intersections with image borders.")

    if smooth:
        profile = profile_line(img_very_smooth, (x_start, y_start), (x_end, y_end))
    else:
        profile = profile_line(img, (x_start, y_start), (x_end, y_end))

    return profile, x_start, x_end, y_start, y_end

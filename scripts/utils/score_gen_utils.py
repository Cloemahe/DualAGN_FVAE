#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2025
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import numpy as np
import pickle

from sklearn.neighbors import BallTree

# Tensorflow

import tensorflow as tf

from architecture.vae_model import model_VAE
from utils.data_generator_utils import generator
from utils.data_utils import preprocessing_type


# Save reconstruction scores into list


def save_score_to_list(latent_dim, model_path_weights, type_dataset):

    b_size = 1
    path = '/workspace/cmahe/pickle_files/objet'

    preprocessing = preprocessing_type('Arcsin hyp')

    data_generator = generator(path, type_dataset)

    data = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32))
    data = data.map(preprocessing).batch(b_size)

    model = model_VAE(latent_dim)[0]
    model.load_weights(model_path_weights)

    count = 0

    for i, features in enumerate(data):
        pred_images, encod_images = model(features)

        input_im = tf.cast(features, dtype=tf.float32)

        MSE = tf.square(tf.subtract(input_im, pred_images))
        MSE = tf.reduce_sum(MSE, axis=[1, 2, 3])
        intensity = tf.reduce_mean(input_im, axis=[1, 2, 3])  # Changed

        score_rec = MSE / intensity  # For relative rec score
        score_rec = tf.math.log(score_rec)

        score_rec = score_rec.numpy()

        if count == 0:
            scores_np = score_rec
        else:
            scores_np = np.concatenate((scores_np, score_rec), axis=0)

        count = count + 1

    dico_score = {'scores_np': scores_np}

    with open('/home/cmahe/dual_agn/pickle_files/scores_list.pkl', 'wb') as f:
        pickle.dump(dico_score, f)

    return scores_np   # = np.array

# Computes score of points


def score_points(pos, tree, scores_np, dim_to_plot, radius):

    x, y = pos[dim_to_plot[0]], pos[dim_to_plot[1]]

    ind = tree.query_radius([(x, y)], r=radius)

    if len(ind[0]) != 0:
        score_r = [scores_np[ind[0]]]
        score_im = np.mean(score_r)

    else:
        score_im = 1e9

    return score_im

# Computes score of generation


def score_gen_function(maf, x_sample, trees, scores_np, latent_dim, radius):

    MSEs = []

    for i in range(latent_dim):
        for j in range(i+1, latent_dim):

            dim_to_plot = [i, j]
            tree = trees[(i, j)]

            MSE = score_points(x_sample, tree, scores_np, dim_to_plot, radius)
            MSEs.append(MSE)

    MSE_m = np.median(MSEs)  # Rec score of x_sample (one image, all dim)

    density_m = maf.log_prob(x_sample)  # Density score of x_sample (one image, all dim)

    return MSE_m, density_m.numpy()


def score_gen_for_sample(maf, sample, encod_np, scores_np, latent_dim, radius):

    MSE_for_sample = []
    density_for_sample = []

    print('Compute generation scores for sample')
    trees = {}

    for i in range(latent_dim):
        for j in range(i + 1, latent_dim):

            pts = encod_np[:, [i, j]]
            trees[(i, j)] = BallTree(pts)

    for i in range(len(sample)):  # should be equal to nb_to_plot

        MSE_m, density_m = score_gen_function(maf,
                                              sample[i, :],
                                              trees,
                                              scores_np,
                                              latent_dim, radius)

        MSE_for_sample.append(MSE_m)
        density_for_sample.append(density_m)

    return MSE_for_sample, density_for_sample


def score_normalised(maf, sample, encod_np, encod_MSE, density_encod, scores_np, latent_dim, radius):

    MSE_for_sample, density_for_sample = score_gen_for_sample(maf,
                                                              sample,
                                                              encod_np,
                                                              scores_np,
                                                              latent_dim,
                                                              radius)

    norm_MSE = (MSE_for_sample - np.median(encod_MSE)) / np.std(encod_MSE)
    norm_density = (density_for_sample - np.median(density_encod)) / np.std(density_encod)

    score_for_sample = norm_MSE - norm_density
    score_for_sample = [int(score_for_sample[i]) + (float(str(score_for_sample[i]).split('.')[1][:2]))/1e2 for i in range(len(score_for_sample))]  # Normalize with just two significative numbers

    return score_for_sample, norm_MSE, norm_density


def score_power_spectrum(images, low_limits, high_limits, score, log):

    print('Compute power spectrum for sample')

    # 2D FFT for all images at once
    images = images[..., 0]

    fft_imgs = np.fft.fft2(images, axes=(1, 2))

    # shift zero-frequency component to center
    fft_imgs = np.fft.fftshift(fft_imgs, axes=(1, 2))

    # power spectrum
    power = np.abs(fft_imgs) ** 2      # (N, H, W)

    N, H, W = power.shape

    # frequency grid
    y, x = np.indices((H, W))

    center_y = H // 2
    center_x = W // 2

    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(np.int32)

    # max radius
    r_max = r.max() #// 2

    # radial profile for all images
    radial_power = np.zeros((N, r_max + 1))

    for k in range(r_max + 1):
        mask = (r == k)
        radial_power[:, k] = power[:, mask].mean(axis=1)
    if log:
        radial_power = np.log10(radial_power)
    low_power = np.median(radial_power[:, low_limits[0]:low_limits[1]], axis=1)
    high_power = np.median(radial_power[:, high_limits[0]:high_limits[1]], axis=1)

    score_pp = [low_power[i]/high_power[i] for i in range(len(low_power))]

    if score is False:
        return radial_power, low_power, high_power, score_pp
    else:
        score_pp = np.log(score_pp)
        score_pp = [int(score_pp[i]) + (float(str(score_pp[i]).split('.')[1][:2]))/1e2 for i in range(len(score_pp))]

        return score_pp
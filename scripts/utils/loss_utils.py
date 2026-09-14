#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import tensorflow as tf
from utils.fourier_utils import fast_fourier_shift, transform_from_fft, create_asymmetric_ring_mask, mask_fn


def loss_fn(y_true, y_predict, beta, latent_dim, R, alpha, zeta, intensity, FW):  # Normal loss

    reconstruct = y_predict[0]
    y_true = tf.cast(y_true, dtype=tf.float32)

    MSE = tf.square(tf.subtract(y_true, reconstruct))

    reduce_MSE = tf.reduce_sum(MSE, axis=[1, 2, 3])
    rec_loss = tf.reduce_mean(reduce_MSE)

    z_mean, z_sigma = y_predict[1][:, :latent_dim], y_predict[1][:, latent_dim:]
    kl_loss_batch = -0.5 * tf.reduce_sum(z_sigma - tf.square(z_mean) - tf.exp(z_sigma) + 1, axis=1)
    kl_loss = tf.reduce_mean(kl_loss_batch)
    loss = rec_loss + beta * kl_loss

    return loss, rec_loss, kl_loss


def loss_masked(y_true, y_predict, beta, latent_dim, R, alpha, zeta, intensity, FW):  # Masked loss

    y_true = tf.cast(y_true, dtype=tf.float32)
    mask_batch = mask_fn(y_true, zero_value=0.1)

    reconstruct = y_predict[0]
    MSE = tf.square(tf.subtract(y_true, reconstruct))

    masked_MSE = tf.multiply(MSE, mask_batch)
    reduce_MSE = tf.reduce_sum(masked_MSE, axis=[1, 2, 3])

    rec_loss = tf.reduce_mean(reduce_MSE)

    z_mean, z_sigma = y_predict[1][:, :latent_dim], y_predict[1][:, latent_dim:]
    kl_loss_batch = -0.5 * tf.reduce_sum(z_sigma - tf.square(z_mean) - tf.exp(z_sigma) + 1, axis=1)
    kl_loss = tf.reduce_mean(kl_loss_batch)
    loss = rec_loss + beta * kl_loss

    return loss, rec_loss, kl_loss


def loss_fourier(y_true, y_predict, beta, latent_dim, R, alpha, zeta, intensity, FW):  # Fourier loss : En cours de developpement
    # Loss fourier = loss normal + fourier

    reconstruct = y_predict[0]
    y_true = tf.cast(y_true, dtype=tf.float32)

    MSE = tf.square(tf.subtract(y_true, reconstruct))

    # Masked loss  -----------------------------------------------------------------
    mask_batch = mask_fn(y_true, zero_value=0.1)  # Geometrical mask of shape (64, 256, 256, 3)

    masked_MSE = tf.multiply(MSE, mask_batch)
    reduce_MSE_mask = tf.reduce_sum(masked_MSE, axis=[1, 2, 3])  # Ici : MSE ou masked MSE

    rec_loss_mask = tf.reduce_mean(reduce_MSE_mask)

    # Normal loss -----------------------------------------------------------------
    reduce_MSE = tf.reduce_sum(MSE, axis=[1, 2, 3])
    rec_loss_n = tf.reduce_mean(reduce_MSE)

    # Fourier loss ----------------------------------------------------------------
    # Fourier transform over each filter separately : shape = (64, 256, 256, 1) each

    mask = create_asymmetric_ring_mask(R, alpha, zeta, intensity, shape=(256, 256))  # Create the Fourier mask once for whole dataset

    # TEST

    x = transform_from_fft(y_true[:, :, :, 0], mask)
    y = fast_fourier_shift(reconstruct[:, :, :, 0])

    MSE_fourier = tf.math.abs(tf.square(tf.subtract(x, y)))
    reduce_MSE_fourier = tf.reduce_sum(MSE_fourier, axis=[1, 2])
    rec_loss_fourier = tf.reduce_mean(reduce_MSE_fourier)

    # Tot loss reconstruct --------------------------------------------------------
    rec_loss = (rec_loss_n + rec_loss_mask) + FW * rec_loss_fourier  # FW =  Fourier weight parameter

    # KL loss ---------------------------------------------------------------------
    z_mean, z_sigma = y_predict[1][:, :latent_dim], y_predict[1][:, latent_dim:]  # Normal space functions
    kl_loss_batch = -0.5 * tf.reduce_sum(z_sigma - tf.square(z_mean) - tf.exp(z_sigma) + 1, axis=1)
    kl_loss = tf.reduce_mean(kl_loss_batch)

    # Tot loss --------------------------------------------------------------------
    loss = rec_loss + beta * kl_loss

    return loss, rec_loss, kl_loss, rec_loss_n, rec_loss_mask, rec_loss_fourier


def loss_composite(y_true, y_predict, beta, latent_dim, R, alpha, zeta, intensity, FW):  # Loss composite = loss masked + loss normal

    reconstruct = y_predict[0]
    y_true = tf.cast(y_true, dtype=tf.float32)

    MSE = tf.square(tf.subtract(y_true, reconstruct))

    # Masked loss -----------------------------------------------------------------
    mask_batch = mask_fn(y_true, zero_value=0.1)

    masked_MSE = tf.multiply(MSE, mask_batch)
    reduce_MSE_mask = tf.reduce_sum(masked_MSE, axis=[1, 2, 3])  # here : MSE ou masked MSE

    rec_loss_mask = tf.reduce_mean(reduce_MSE_mask)

    # Normal loss -----------------------------------------------------------------
    reduce_MSE = tf.reduce_sum(MSE, axis=[1, 2, 3])
    rec_loss_n = tf.reduce_mean(reduce_MSE)

    # Tot loss reconstruct --------------------------------------------------------
    rec_loss = rec_loss_n + rec_loss_mask

    # KL loss ---------------------------------------------------------------------
    z_mean, z_sigma = y_predict[1][:, :latent_dim], y_predict[1][:, latent_dim:]  # Normal space functions
    kl_loss_batch = -0.5 * tf.reduce_sum(z_sigma - tf.square(z_mean) - tf.exp(z_sigma) + 1, axis=1)
    kl_loss = tf.reduce_mean(kl_loss_batch)

    # Tot loss --------------------------------------------------------------------
    loss = rec_loss + beta * kl_loss

    return loss, rec_loss, kl_loss

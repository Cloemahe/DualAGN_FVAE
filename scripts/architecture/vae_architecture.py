#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers
import tensorflow_probability as tfp

tfd = tfp.distributions


def encoder(latent_dim):

    encoder_input = tfk.Input(shape=(256, 256, 1))  # TEST

    x = layers.Conv2D(16, (3, 3), 2, 'SAME', activation='elu')(encoder_input)
    x = layers.Conv2D(32, (3, 3), 2, 'SAME', activation='elu')(x)
    x = layers.Conv2D(128, (3, 3), 2, 'SAME', activation='elu')(x)
    x = layers.Conv2D(256, (3, 3), 2, 'SAME', activation='elu')(x)
    x = layers.Conv2D(512, (3, 3), 2, 'SAME', activation='elu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2 * latent_dim)(x)  # 2x the latent space because we have mu and sigma each

    encoder_output = x
    Encoder = tfk.Model(encoder_input, encoder_output, name='encoder_VAE')
    return Encoder, encoder_output  # encoder_output is the code of the positions in latent space


def decoder(latent_dim, generation_mode=False):

    if not generation_mode:
        print('Decoding from VAE Encoder')

        decoder_input = tfk.Input(shape=2 * latent_dim)  # ignore batch size

        x = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=t[..., :latent_dim], scale_diag=tf.exp(t[..., latent_dim:])),
                                          convert_to_tensor_fn=lambda s: s.sample())(decoder_input)  # initialization du modèle
    else:
        decoder_input = tfk.Input(shape=latent_dim)  # ignore batch size
        print('Decoding from new samples')

        x = decoder_input

    x = layers.Dense(8 * 8 * 512, activation='relu')(x)
    x = layers.Reshape((8, 8, 512))(x)
    x = layers.Conv2DTranspose(256, 3, activation='elu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, activation='elu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='elu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='elu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(16, 3, activation='elu', strides=2, padding='same')(x)
    x = layers.Conv2D(1, 3, strides=1, padding='same')(x)  # TEST (1,3 = 3,3), activation = 'relu' /!\

#     out_2 = layers.Conv2D(1, 3, activation='sigmoid', strides=1, padding='same')(x)

    decoder_output = x
    Decoder = tfk.Model(inputs=decoder_input, outputs=decoder_output, name='decoder_VAE')
    return Decoder

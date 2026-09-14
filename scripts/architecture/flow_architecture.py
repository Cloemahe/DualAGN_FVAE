#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules
# Tensorflow

import tensorflow as tf
import tensorflow_probability as tfp

# Special functions

tfd = tfp.distributions
tfb = tfp.bijectors


def model_flow(nb_made, latent_dim):

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim, dtype='float32'), scale_diag=tf.ones(latent_dim, dtype='float32'))

    bijectors = []

    for i in range(0, nb_made):
        bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, hidden_units=[128, 128, 64, 32], activation='relu')))  # Hidden param is the number of layers hidden

    bijector = tfb.Chain(bijectors=list(reversed(bijectors)))  # MAF = Masked Autoregressive Flow

    maf = tfd.TransformedDistribution(distribution=base_dist, bijector=bijector)

    return maf

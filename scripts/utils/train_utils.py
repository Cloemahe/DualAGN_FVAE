#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import tensorflow as tf
from utils.loss_utils import loss_fn, loss_masked, loss_fourier, loss_composite


# Custom train function
@tf.function
def train_step(x, model, optimizer, latent_dim, Beta, type_loss, R, alpha, zeta, intensity, FW, training):
    """
    Training step of the model.

    :param x: Input tensor.
    :type nb_epoch: TF.Tensor
    :param model: VAE model = architecture of the encoder and decoder with number of layers and neurons.
    :param optimizer: Optimizer of the model : see lr_utils.py, vae_training_loop.py
    :param latent_dim: Dimension of the latent liminal space of the model.
    :type latent_dim: int
    :param Beta: Define the weight of the KL divergence on the loss (AE = 0, Default = 1).
    :type Beta: int
    :param training: Nature of the dataset used : necessary to apply the gradient descent or not
    :type training: Bool
    :param type_loss: Type of loss function used : Normal, Masked, Fourier, Composite
    :type type_loss: str
    :param a_pl: Power law coefficient
    :type a_pl: int
    :param k_pl: Power law coefficent
    :type k_pl: int
    :return: Compute the loss between the input image and the reconstruction, compute the gradient descent accordingly, apply the optimization of weights through the optimizer.
    """

    if type_loss == 'Fourier':
        loss = loss_fourier
    elif type_loss == 'Normal':
        loss = loss_fn
    elif type_loss == 'Masked':
        loss = loss_masked
    elif type_loss == 'Composite':
        loss = loss_composite

    else:
        print('Type of loss is not defined')

    with tf.GradientTape() as tape:
        prediction = model(x)
        losses = loss(x, prediction, Beta, latent_dim, R, alpha, zeta, intensity, FW)

        tot_loss = losses[0]

        if training:
            grads = tape.gradient(tot_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses


def flow_train_step(maf, batch, optimizer, training):

    with tf.GradientTape() as tape:
        tape.watch(maf.trainable_variables)

        loss = - tf.reduce_mean(maf.log_prob(batch))  # Negative log likelihood ; .log_prod <=> sens : transformed g-1(z) -> base_dist (x)

        if training:
            gradients = tape.gradient(loss, maf.trainable_variables)
            optimizer.apply_gradients(zip(gradients, maf.trainable_variables))

    return loss

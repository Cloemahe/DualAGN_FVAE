#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules

import numpy as np
import matplotlib.pyplot as plt

# Tensorflow

import tensorflow as tf
import tensorflow_probability as tfp

from architecture.flow_architecture import model_flow

from utils.train_utils import flow_train_step
from utils.diagnostics_flow_utils import corner_plots, test_flow
from utils.plot_utils import display_flow_loss
from utils.save_flow_class import FlowModel
from utils.data_utils import comp_range
from utils.data_generator_utils import encod_generator
from utils.save_encod import save_encodings_to_pickle, save_encodings_to_list

# Special functions

tfd = tfp.distributions
tfb = tfp.bijectors


'''Training loops'''


def flow(path, n_run, gen, test_on, new, type_loss, num_run, nb_epoch, nb_made, latent_dim, b_size, init_lr, end_lr, decay, power, nb_sample, nb_to_plot, nb_dim_to_plot, plot_every, validation_every):

    # Run informations
    nb_encod = comp_range(path)[4]
    Nb_step = int(nb_epoch * nb_encod / b_size)
    print(nb_encod, Nb_step)

    # Path to weights of VAE run
    model_path_weights = f'../checkpoints/{type_loss}_model_{num_run}_end'
    # decoder_path_weigths =  f'./checkpoints/{type_loss}_decoder_{num_run}_end'

    '''Save encodings (one for a given VAE run)    '''

    # -- Latent space in pickle files --
    if new:
        print(f'Saving new encodings for model {type_loss}_{num_run}')
        save_encodings_to_pickle(latent_dim, model_path_weights)

    # ## Latent space as list ###
    encod_np = save_encodings_to_list(latent_dim, model_path_weights, type_dataset='Test')

    '''Load of encodings from latent space '''

    # -- Latent space as dataset --
    encoding_generator = encod_generator(path, type_dataset='Train')

    train_dataset = tf.data.Dataset.from_generator(encoding_generator, output_types=(tf.float32))
    train_dataset = train_dataset.batch(b_size)

    # Validation dataset

    valid_generator = encod_generator(path, type_dataset='Validation')

    valid_dataset = tf.data.Dataset.from_generator(valid_generator, output_types=(tf.float32))
    valid_dataset = valid_dataset.batch(b_size)

    '''Base distribution : Multi-Variate Gaussian'''

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim, dtype='float32'), scale_diag=tf.ones(latent_dim, dtype='float32'))

    x_base_samples = base_dist.sample(nb_sample).numpy()  # Sample of base distribution

    print('Base distribution (Gaussian) :', x_base_samples[0])

    '''  Optimizers'''

    learning_rate_fct = tf.keras.optimizers.schedules.PolynomialDecay(init_lr, Nb_step, end_lr, power, cycle=False)  # Polynomial decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fct)

    '''Training loop'''

    maf = model_flow(nb_made, latent_dim)

    # ## Sample from flow before training ###
    if test_on:
        corner_plots(maf, x_base_samples, latent_dim, path, nb_dim_to_plot, encod_np, nb_sample, type_dataset='Train')

    # ## Training ###

    # Trainable variables /!\ caution : weight vectors are seen as one trainable variable
    n_trainable_variables = 0

    for weights in maf.trainable_variables:
        n_trainable_variables = n_trainable_variables + np.prod(weights.shape)

    print('Trainable variables =', n_trainable_variables)

    train_losses = []
    valid_losses = []

    for i in range(nb_epoch):
        for batch in train_dataset:  # train_dataset is (z), the latent space

            train_loss = flow_train_step(maf, batch, optimizer, training=True)

        # Save the losses for the plot
        train_losses.append(train_loss)
        print(f'Loss for epoch {i}', train_loss)

        if i % plot_every == 0 and test_on:  # Tests during the training process

            corner_plots(maf, x_base_samples, latent_dim, path, nb_dim_to_plot, encod_np, nb_sample, type_dataset='Train')

        if i % validation_every == 0:  # Validation loop
            for j, valid in enumerate(valid_dataset):
                valid_loss = flow_train_step(maf, valid, optimizer, training=False)

            valid_losses.append(valid_loss)

    print('Final loss', train_loss)

    fig_loss = display_flow_loss(train_losses, valid_losses, n_run)
    plt.savefig('training_loss_function.png')
    plt.close(fig_loss)

    # -- Save of Flow model --

    flow_model = FlowModel(maf)
    flow_model.save(f'../checkpoints/flow_model_{n_run}.keras')
    flow_model.save_weights(f'../checkpoints/flow_model_{n_run}_end')

    if test_on:

        test_flow(nb_made, latent_dim, nb_sample, path, nb_to_plot, nb_dim_to_plot, gen, n_run, num_run, type_loss)

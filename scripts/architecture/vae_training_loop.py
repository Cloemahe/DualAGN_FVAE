#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import tensorflow as tf
from utils.train_utils import train_step
from utils.plot_utils import display, display_hist, display_fourier, loss_plot
from utils.diagnostics_vae_utils import test_model_vae

from architecture.vae_model import model_VAE


def train_loop(nb_epoch, dataset, dataset_valid, latent_dim, Beta, init_lr, Nb_step,
               decay, save, loop_run, type_loss, type_preprocessing,
               R, alpha, zeta, intensity, FWi, validation_every, test_on, num_run):
    """
    Compute the training of the model on the dataset.

    :param nb_epoch: Number of epoch for the training.
    :type nb_epoch: int
    :param dataset: Dataset containing all the imnages part of the training sample.
    :type dataset: TF.Dataset
    :param latent_dim: Dimension of the latent liminal space of the model.
    :type latent_dim: int
    :param Beta: Define the weight of the KL divergence on the loss (AE = 0, Default = 1).
    :type Beta: int
    :param init_lr: Initial/Starting value of the learning rate.
    :type init_lr: float
    :param Nb_step:  Number of step in the train loop, computed with the number of epoch, batch size and number of images in the dataset.
    :type Nb_step: float
    :param decay: Decay value applied in the exponential decay, decreasing the value of the learning rate.
    :type decay: float
    :param type_loss: Type of loss function used : Normal, Masked, Fourier, Composite
    :type type_loss: str
    :param a_pl: Power law coefficient
    :type a_pl: int
    :param k_pl: Power law coefficent
    :type k_pl: int
    :return: Display of the generated images during the training, repartition of the encodings in the latent space, and the trained model.
    """

    model, encoder, decoder = model_VAE(latent_dim)
    model.summary()

    if loop_run:
        model.load_weights('../checkpoints/Fourier_model_220_end')  # To change each time /!\
        init_lr = 1e-40
    ''' Optimizer '''

    schedule = tf.keras.optimizers.schedules.ExponentialDecay
    lr_schedule = schedule(init_lr, decay_steps=Nb_step, decay_rate=decay)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    ''' Training loop '''

    tot_loss_results = []  # Total loss function

    train_results = []
    valid_results = []

    t = trange(nb_epoch, desc='Convo VAE')
    count = 0

    steps_to_print = np.logspace(0, np.log10(Nb_step), 100)
    try:
        steps_to_print = [int(i) for i in steps_to_print]
    except ValueError:
        steps_to_print = [0, Nb_step]

    def gaussian(x, mu, sig):  # Gaussian function for Fourier weights
        return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

    variating_fw = gaussian(np.linspace(0, 10000, Nb_step + 10), mu=2000, sig=2000)  # Gaussian FW

    for epoch in t:  # Training

        for features in dataset:

            if loop_run:
                FW = np.float32(FWi)  # The final weight once the training is done
            else:
                try:
                    FW = np.float32(variating_fw[count])
                except IndexError:
                    FW = np.float32(variating_fw[Nb_step])

            # Computation of the losses and gradient descent
            losses = train_step(features, model, optimizer, latent_dim, Beta, type_loss, R, alpha, zeta, intensity, FW, training=True)
            tot_loss_results.append(losses[0])  # For the progression bar

            pred_images, encod_images = model(features)
            encod_np = encod_images.numpy()[:, :latent_dim]

            count += 1
        # Visualisation of some reconstruction during training
            if count in steps_to_print and save:
                fig_display = display(features, pred_images, count, encod_np, latent_dim)
                plt.savefig(f'reconstruction_images_corner_reconst_{count}.png')
                plt.close(fig_display)

                fig_hist = display_hist(features, pred_images, count)
                plt.savefig(f'reconstruction_image_hist_{count}.png')
                plt.close(fig_hist)

                if type_loss == 'Fourier':
                    fig_fourier = display_fourier(features, pred_images, R, alpha, zeta, intensity, shape=(256, 256))
                    plt.savefig(f'reconstruction_fourier_plot_{count}.png')
                    plt.close(fig_fourier)

        train_results.append(losses[0])

        t.set_description('Conv VAE, loss=%g' % tot_loss_results[-1])

        if test_on:
            print('Saving the weights')
            model.save_weights(f'../checkpoints/{type_loss}_model_{num_run}_step_{count}')   # Save the weights of the model

            # Save weights from the encoder and the decoder
            encoder.save_weights(f'../checkpoints/{type_loss}_encoder_{num_run}_step_{count}')
            decoder.save_weights(f'../checkpoints/{type_loss}_decoder_{num_run}_step_{count}')

        # Validation step : Computation of losses on validation dataset, without training.

        if epoch % validation_every == 0:
            for j, valid_img in enumerate(dataset_valid):

                # Computation of the losses on the validation dataset
                losses_valid = train_step(valid_img, model, optimizer, latent_dim, Beta, type_loss, R, alpha, zeta, intensity, FW, training=False)

                pred_valid, encod_valid = model(valid_img)

                # Visualisation of the reconstruction during the last step of the validation
                if epoch == nb_epoch - 1 and save:
                    fig_hist_valid = display_hist(valid_img, pred_valid, epoch)
                    plt.savefig(f'reconstruction_valid_img_hist_{j}.png')
                    plt.close(fig_hist_valid)

            valid_results.append(losses_valid[0])

    fig_loss_plot = loss_plot(train_results, valid_results)
    plt.savefig('VAE_loss.png')
    plt.close(fig_loss_plot)

    # Save trained weights

    print('Saving the final weights')
    model.save_weights(f'../checkpoints/{type_loss}_model_{num_run}_end')    # Save the weights of the model

    # Save weights from the encoder and the decoder
    encoder.save_weights(f'../checkpoints/{type_loss}_encoder_{num_run}_end')
    decoder.save_weights(f'../checkpoints/{type_loss}_decoder_{num_run}_end')

    # Final tests on test dataset

    if test_on:

        print('Begin tests : Model and UMAP')
        test_model_vae(latent_dim, type_loss, type_preprocessing, num_run, R, alpha, zeta, intensity)

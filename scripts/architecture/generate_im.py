#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2025
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules

import matplotlib.pyplot as plt
import random

import tensorflow_probability as tfp

from architecture.vae_architecture import decoder

tfd = tfp.distributions

# Image generation function


def generate_images(latent_dim, samples, decoder_path_weigths, flow_on):

    Decoder = decoder(latent_dim, generation_mode=True)

    Decoder.load_weights(decoder_path_weigths)  # Automatically changed the weights

    if flow_on is True:
        print('Generate from flow')
    else:
        print('Generate from Gaussian Distribution')

    # Generation of new images
    decoded_img = Decoder(samples)

    return decoded_img


def display_gen_images(decoded_img, scores_gen, score_pp, title):

    gen_img = decoded_img.numpy()

    nb_plot = 4  # Number of images to display in the plot
    nb_im = range(len(gen_img))  # Number of images generated

    im_to_plot = random.sample(nb_im, nb_plot)  # Indexes of the images to plot

    # Plot images
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    fig.suptitle(title, fontsize=10)

    ax[0, 0].imshow(gen_img[im_to_plot[0], :, :, 0])
    ax[0, 0].set_title(rf'$S_g$ = {scores_gen[0]}, $P_g$ = {score_pp[0]}', fontsize=10)

    ax[0, 1].imshow(gen_img[im_to_plot[1], :, :, 0])
    ax[0, 1].set_title(rf'$S_g$ = {scores_gen[1]}, $P_g$ = {score_pp[1]}', fontsize=10)

    ax[1, 0].imshow(gen_img[im_to_plot[2], :, :, 0])
    ax[1, 0].set_title(rf'$S_g$ = {scores_gen[2]}, $P_g$ = {score_pp[2]}', fontsize=10)

    ax[1, 1].imshow(gen_img[im_to_plot[3], :, :, 0])
    ax[1, 1].set_title(rf'$S_g$ = {scores_gen[3]}, $P_g$ = {score_pp[3]}', fontsize=10)

    return fig

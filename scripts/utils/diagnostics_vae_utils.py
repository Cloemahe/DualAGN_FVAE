#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2025
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


#  VARIATIONAL AUTOENCODER : test the trained model to reproduce Horizon-AGN dataset ###

''' Modules '''

# Classical modules
import matplotlib.pyplot as plt
import corner

# Local functions
from utils.data_utils import print_memory_usage, preprocessing_type
from utils.data_generator_utils import generator
from utils.plot_utils import display, corner_plot, display_hist, display_gini, profile_relative_diff, display_fourier, display_umap, display_article, input_rec_param_comp

from architecture.vae_model import model_VAE

# Tensorflow
import tensorflow as tf


def test_model_vae(latent_dim, type_loss, type_preprocessing, num_run, R, alpha, zeta, intensity):

    print('Test of the trained model')

    ''' Workload parameters '''

    print_memory_usage('just starting')

    path = '../data/objet'

    # Hyperparamètres

    b_size = 512  # Batch size

    ''' Dataset '''

    preprocessing = preprocessing_type(type_preprocessing)

    img_generator = generator(path, type_dataset='Test')  # Depending on the type specified in 'type_dataset', this is our training dataset OR test dataset OR validation dataset
    dataset_test = tf.data.Dataset.from_generator(img_generator,
                                                  output_types=(tf.float64))  # Loading from generator of dataset : all images are not loaded at the same time

    # dataset_test = dataset_test.shuffle(shuffle_size).map(preprocessing).batch(b_size) # Processing of the dataset
    dataset_test = dataset_test.map(preprocessing).batch(b_size)  # same

    ''' Test loop '''

    model, Encoder, Decoder = model_VAE(latent_dim)

    model.load_weights(f'../checkpoints/{type_loss}_model_{num_run}_end')
    model.summary()

    count = 0

    for i, features in enumerate(dataset_test):

        pred_images, encod_images = model(features)
        encod_np = encod_images.numpy()[:, :latent_dim]

        fig_test = input_rec_param_comp(features, pred_images, i, save=False)
        plt.savefig(f'./test/test_comp_params_{i}.png')
        plt.close(fig_test)

        fig_article = display_article(features, pred_images)
        plt.savefig(f'test_images_comp_article_{i}.png')
        plt.close(fig_article)

        fig_display = display(features, pred_images, count, encod_np, latent_dim)
        plt.savefig(f'test_images_corner_reconst_{i}.png')
        plt.close(fig_display)

        fig_hist = display_hist(features, pred_images, count)
        plt.savefig(f'test_image_hist_{i}.png')
        plt.close(fig_hist)

        fig_corner = corner.corner(corner_plot(encod_np, latent_dim, nb_dim_to_plot=5)[0])
        plt.savefig(f'test_corner_plot_{i}.png')
        plt.close(fig_corner)

        fig_gini_hist = display_gini(features, pred_images)
        plt.savefig(f'test_image_gini_hist{i}.png')
        plt.close(fig_gini_hist)

        fig_profile, validity = profile_relative_diff(features, pred_images)
        if validity:  # Remove cases where image is to faint to compute profiles
            plt.savefig(f'test_profiles_{i}.png')
        plt.close(fig_profile)

        if type_loss == 'Fourier':
            fig_fourier = display_fourier(features, pred_images, R, alpha, zeta, intensity, shape=(256, 256))
            plt.savefig(f'test_fourier_plot_{i}.png')
            plt.close(fig_fourier)

        count += 1

    fig_umap = display_umap(type_loss, latent_dim, num_run)
    plt.savefig('umap_first_labels.png')
    plt.close(fig_umap)

    print("All done")

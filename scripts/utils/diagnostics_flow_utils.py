#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules
import numpy as np
import matplotlib.pyplot as plt
import time
import umap
import pickle

# Tensorflow

import tensorflow as tf
import tensorflow_probability as tfp

from architecture.flow_architecture import model_flow
from architecture.generate_im import generate_images, display_gen_images

from utils.save_flow_class import FlowModel
from utils.save_encod import save_encodings_to_list
from utils.plot_utils import profile_gen, score_gen_multiple, corner_flow_lt, corner_in_gauss, hist_scores, comp_scores_intensity, comp_fwhm_int, img_density, plot_power_spectra
from utils.score_gen_utils import save_score_to_list, score_gen_for_sample, score_normalised, score_power_spectrum

# Special functions

tfd = tfp.distributions
tfb = tfp.bijectors


def corner_plots(maf, x_base_samples, latent_dim, path, nb_dim_to_plot, encod_np, nb_sample, type_dataset):

    samples = maf.sample(nb_sample)

    #  Plot posingauss before training
    fig_corner_gauss, validity = corner_in_gauss(maf,
                                                 x_base_samples,
                                                 path,
                                                 nb_dim_to_plot,
                                                 type_dataset)
    if validity:
        plt.savefig('corner_plot_gauss.png')
    plt.close(fig_corner_gauss)

    # Corner plot : position of MAF sample in latent space
    fig_corner_flow, validity = corner_flow_lt(samples,
                                               latent_dim,
                                               encod_np,
                                               nb_dim_to_plot)
    if validity:
        plt.savefig('corner_plot_samples.png')
    plt.close(fig_corner_flow)


def test_flow(nb_made, latent_dim, nb_sample, path, nb_to_plot, nb_dim_to_plot, gen, old_run, num_run, type_loss):

    b_size = 100

    # -- Path to weights of VAE run --
    model_path_weights = f'./checkpoints/{type_loss}_model_{num_run}_end'
    decoder_path_weigths = f'./checkpoints/{type_loss}_decoder_{num_run}_end'

    print(f'Test on flow run {old_run}')

    # -- Base distribution : multi-variate Gaussian --
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim, dtype='float32'),
                                           scale_diag=tf.ones(latent_dim, dtype='float32'))
    x_base_samples = base_dist.sample(nb_sample).numpy()

    # -- Latent space as list --
    encod_np = save_encodings_to_list(latent_dim,
                                      model_path_weights,
                                      type_dataset='Test')
    reducer = umap.UMAP().fit(encod_np)

    # Save of VAE reconstruction scores #
    scores_np = save_score_to_list(latent_dim,
                                   model_path_weights,
                                   type_dataset='Test')

    # -- Load of MAF --
    maf = model_flow(nb_made, latent_dim)
    flow_model = FlowModel(maf)
    flow_model.load_weights(f'./checkpoints/flow_model_{old_run}_end')

    # -- Save of VAE gen scores --
    encod_MSE, density_encod = score_gen_for_sample(maf,
                                                    encod_np,
                                                    encod_np,
                                                    scores_np,
                                                    latent_dim,
                                                    radius=0.5)

    dico_norm = {'MSE_encod': encod_MSE,
                 'density_encod': density_encod}

    with open('../pickle_files/scores_vae.pkl', 'wb') as f:
        pickle.dump(dico_norm, f)

    # -- Sampling --

    corner_plots(maf,
                 x_base_samples,
                 latent_dim,
                 path,
                 nb_dim_to_plot,
                 encod_np,
                 nb_sample,
                 type_dataset='Test')
    print('Corners are done !')

    if gen:

        # -- Examples of reconstructions from VAE --

        rdm_encod = np.random.choice(np.arange(encod_np.shape[0]),
                                     nb_to_plot,
                                     replace=False)
        vae_samples = encod_np[rdm_encod]

        decoded_VAE_img = generate_images(latent_dim,
                                          vae_samples,
                                          decoder_path_weigths,
                                          flow_on=False)

        # -- Generation From Gaussian --
        x_samples = base_dist.sample(nb_to_plot)
        decoded_gauss_img = generate_images(latent_dim,
                                            x_samples,
                                            decoder_path_weigths,
                                            flow_on=False)  # flow_on = False for generation from Gaussian distribution

        score_gauss, MSE_gauss, density_gauss = score_normalised(maf,
                                                                 x_samples,
                                                                 encod_np,
                                                                 encod_MSE,
                                                                 density_encod,
                                                                 scores_np,
                                                                 latent_dim,
                                                                 radius=0.5)
        score_pp_g = score_power_spectrum(decoded_gauss_img,
                                          low_limits=[1, 5],
                                          high_limits=[50, 80],
                                          score=True,
                                          log=False)

        fig_gauss_generate = display_gen_images(decoded_gauss_img,
                                                score_gauss,
                                                score_pp_g,
                                                title='Examples of images generated from samples of Gaussian distribution')
        plt.savefig('./test/test_generation_im_from_gauss.png')
        plt.close(fig_gauss_generate)

        # -- Generation from Flow --

        b_s = nb_to_plot//b_size
        count = 0

        st = time.time()

        for i in range(b_s):
            samples = maf.sample(b_size)
            decoded_img = generate_images(latent_dim,
                                          samples,
                                          decoder_path_weigths,
                                          flow_on=True)  # flow_on = False for generation from Gaussian distribution

            if count == 0:
                decoded_flow_img = decoded_img
                plot_samples = samples
            else:
                decoded_flow_img = tf.concat([decoded_flow_img, decoded_img], 0)
                plot_samples = tf.concat([plot_samples, samples], 0)

            count = count + 1

        et = time.time()

        ds = (et - st)
        delta = ds/3600
        print('Number of image', nb_to_plot, 'Total : Time = ', ds, 'secondes, = ',  delta, 'heures')

        # -- Compute scores for sample --

        score_sample, MSE_sample, density_sample = score_normalised(maf,
                                                                    plot_samples,
                                                                    encod_np,
                                                                    encod_MSE,
                                                                    density_encod,
                                                                    scores_np,
                                                                    latent_dim,
                                                                    radius=0.5)
        score_pp = score_power_spectrum(decoded_flow_img,
                                        low_limits=[1, 5],
                                        high_limits=[50, 80],
                                        score=True,
                                        log=False)

        fig_pp_spectra = plot_power_spectra(decoded_VAE_img.numpy(),
                                            decoded_gauss_img.numpy(),
                                            decoded_flow_img.numpy())
        plt.savefig('./test/test_power_spectra.png')
        plt.close(fig_pp_spectra)

        fig_generate = display_gen_images(decoded_flow_img,
                                          score_sample,
                                          score_pp,
                                          title='Examples of images generated from normalizing flow')
        plt.savefig('./test/test_generation_im_from_flow.png')
        plt.close(fig_generate)

        fig_hist = hist_scores(decoded_flow_img,
                               score_sample,
                               MSE_sample,
                               density_sample)
        plt.savefig('./test/test_hist_scores_norm.png')
        plt.close(fig_hist)

        fig_comp_score = comp_scores_intensity(decoded_flow_img,
                                               score_sample,
                                               distri=False)
        plt.savefig('./test/test_int_fwhm_scores_norm.png')
        plt.close(fig_comp_score)

        fig_hist_scores = comp_scores_intensity(decoded_flow_img,
                                                score_sample,
                                                distri=True)
        plt.savefig('./test/test_hist_gen_score_norm.png')
        plt.close(fig_hist_scores)

        fig_comp_density = img_density(decoded_flow_img,
                                       score_sample,
                                       MSE_sample,
                                       density_sample)
        plt.savefig('./test/test_comp_density.png')
        plt.close(fig_comp_density)

        fig_comp_int = comp_fwhm_int(decoded_flow_img, save=False)
        plt.savefig('./test/test_inf_fwhm_all.png')
        plt.close(fig_comp_int)

        fig_score_gen = score_gen_multiple(decoded_flow_img.numpy(),
                                           score_sample,
                                           score_pp,
                                           nb_to_plot)
        plt.savefig('./test/test_score_generations.png')
        plt.close(fig_score_gen)

        # -- Profiles on generations --

        for i in range(nb_to_plot):
            profile_generate, validity = profile_gen(decoded_flow_img.numpy()[i, :, :, 0],
                                                     plot_samples.numpy()[i, :],
                                                     reducer,
                                                     score_sample[i])
            plt.savefig(f'./test/test_profiles_gen_im_{i}.png')
            plt.close(profile_generate)

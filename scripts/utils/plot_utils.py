#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2025
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
# import math

import tensorflow as tf
import pickle
import corner
import random

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_widths
# from skimage import io, filters, measure, morphology
# from skimage.filters import threshold_otsu
from skimage.measure import profile_line
from lmfit.models import GaussianModel

from utils.gini_utils import gini_array, gini_coeff
from utils.fourier_utils import create_asymmetric_ring_mask, fast_fourier_shift, transform_from_fft, mask_fn
from utils.umap_utils import umap_test
from utils.profiles_utils import regions_img, profile_dual, profile_maj_axis
from utils.score_gen_utils import score_power_spectrum

from utils.save_encod import save_posingauss


# Loss plot : visualise the loss as a function of the epochs of the training


def loss_plot(total_loss, validation_loss):

    fig = plt.figure()

    plt.plot(total_loss,
             color='blue',
             linestyle='dashed',
             label='Training loss')
    plt.plot(validation_loss,
             color='red',
             linestyle='dashdot',
             label='Validation loss')
    plt.xlabel('Training epoch of the model')
    plt.yscale('log')
    plt.ylabel('Value of the total loss function')
    plt.title('Loss values as function of the training epoch of the model')
    plt.grid(alpha=0.5)
    plt.legend()

    return fig

# Corner plot : representation of the latent space in the variating dimensions


def corner_plot(codes_images, latent_dim, nb_dim_to_plot):
    """
    Return the corner plot of the latent space at a given iteration.

    :param codes_images: encodings of the images in the latent space.
    :type codes_images: list[float]
    :param epoch: Number of the current iteration of the model.
    :param latent_dim: Dimension of the latent space.
    :type latent_dim: int
    """

    im_codes = []  # List of codes with important variance
    index_codes = []  # List of positions of the codes with important variance

    var_codes = np.std(codes_images, axis=0)

    m_var = np.median(var_codes, axis=0)  # Median of variance distribution
    s_var = np.std(var_codes, axis=0)  # Standard deviation

    for j in range(latent_dim):
        if (m_var - s_var) <= var_codes[j] <= (m_var + s_var):
            pass
        else:
            im_codes.append(codes_images[:, j])
            index_codes.append(j)

    im_codes = np.array(im_codes).T

    im_codes = im_codes[:, :nb_dim_to_plot]
    index_codes = index_codes[:nb_dim_to_plot]
    # figure = corner.corner(im_codes)

    return im_codes, index_codes

# Input vers output of the model, corner plot and latent space positions


def display(im_x, im_y, count, encoded_im, latent_dim):

    # -----------
    nb_plot = 5  # Number of images to display in the plot
    nb_dim_to_plot = 3
    nb_batch = range(len(im_y.numpy()))  # Number of images batch considered

    im_to_plot = random.sample(nb_batch, nb_plot)  # Indexes of images to plot:

    # -----------
    # Encoding of images with largest variance and their positions
    im_codes, index_codes = corner_plot(encoded_im,
                                        latent_dim,
                                        nb_dim_to_plot)

    # -----------
    fig = plt.figure(layout='constrained', figsize=(40, 50))
    # fig.tight_layout(pad = 2.0)
    subfigs = fig.subfigures(1, 3)

    ax_input_image = subfigs[0].subplots(nb_plot, 1)
    subfigs[0].suptitle('Input images', fontsize=40)

    fig_corner_plot = subfigs[1].subfigures(nb_plot, 1)
    subfigs[1].suptitle('Corner plots', fontsize=40)

    ax_reconst_image = subfigs[2].subplots(nb_plot, 1)
    subfigs[2].suptitle('Reconstructed images', fontsize=40)

    # -----------
    for i in range(nb_plot):

        a = ax_input_image[i].imshow(im_x[im_to_plot[i], :, :, 0])
        ax_input_image[i].tick_params(axis='both', labelsize=20)

        b = ax_reconst_image[i].imshow(im_y.numpy()[im_to_plot[i], :, :, 0])
        ax_reconst_image[i].tick_params(axis='both', labelsize=20)

        ca = plt.colorbar(a, label='Pixel value')
        ca.set_label(label='Pixel value', size=40)
        ca.ax.tick_params(labelsize=30)

        cb = plt.colorbar(b, label='Pixel value')
        cb.set_label(label='Pixel value', size=40)
        cb.ax.tick_params(labelsize=30)

        im_corner = corner.corner(im_codes, fig=fig_corner_plot[i])
        axes = im_corner.axes
        axes = np.reshape(axes, (nb_dim_to_plot, nb_dim_to_plot))

        for c in range(nb_dim_to_plot):
            for k in range(nb_dim_to_plot):
                if c <= k:
                    axes[k, c].scatter(
                        encoded_im[im_to_plot[i], index_codes[c]],
                        encoded_im[im_to_plot[i], index_codes[k]],
                        marker="+",
                        c="red",
                        s=1e3,
                        linewidths=3,
                    )

    return fig

# Input versus output images and their respectives histograms


def display_hist(im_x, im_y, epoch):
    """
    Return the figure displaying the original image extracted from the dataset
    and its prediction generated through the model, along with the histograms
    of both images.

    :param im_x: Original image from the Tensorflow dataset.
    :param im_y: Predicted image by the model.
    :param epoch: Number of the current iteration of the model.
    """

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=2.0)

    # TEST

    axs[0, 0].imshow(im_x.numpy()[0, :, :, 0])  # features is here the input
    axs[0, 0].set_title("Input image", fontsize=10)
    axs[0, 1].imshow(
        im_y.numpy()[0, :, :, 0]
    )  # p is the prediction made by the model at the given epoch
    axs[0, 1].set_title("Reconstructed image", fontsize=10)
    axs[1, 0].hist(
        im_x.numpy()[0, :, :, 0].flatten(),
        bins=40,
        alpha=0.4,
        label="SDSS_r",
        ec="black",
    )
    axs[1, 0].set_title("Input image Histogram", fontsize=10)
    axs[1, 0].set_yscale("log")
    axs[1, 1].hist(
        im_y.numpy()[0, :, :, 0].flatten(),
        bins=40,
        alpha=0.4,
        label="SDSS_r",
        ec="black",
    )
    axs[1, 1].set_title("Reconstructed image Histogram", fontsize=10)
    axs[1, 1].set_yscale("log")

    return fig


def display_article(im_x, im_y):

    nb_batch = len(im_y.numpy())  # Number of images inside the batch considered

    gini_range_in, mean_gini_in = gini_array(im_x, nb_batch)  # Array of Gini values for each image in the input batch
    gini_range_out, mean_gini_out = gini_array(im_y, nb_batch)  # Array of Gini values for each image in the input batch

    #Im to plot
    im_x_to_show = im_x.numpy()[0, :, :, 0]
    im_y_to_show = im_y.numpy()[0, :, :, 0]

    #Gini values of im to plot
    gini_value_x = gini_coeff(im_x_to_show)
    gini_value_x = int(gini_value_x) + (float(str(gini_value_x).split('.')[1][:2]))/1e2

    gini_value_y = gini_coeff(im_y_to_show)
    gini_value_y = int(gini_value_y) + (float(str(gini_value_y).split('.')[1][:2]))/1e2

    fig, axs = plt.subplots(3, 2, figsize=(15, 18))

    axs[0, 0].imshow(im_x.numpy()[0, :, :, 0])  # features is here the input
    axs[0, 0].set_title('Input image', fontsize=20)
    axs[0, 1].imshow(im_y.numpy()[0, :, :, 0])  # p is the prediction made by the model at the given epoch
    axs[0, 1].set_title('Reconstructed image', fontsize=20)

    axs[1, 0].hist(im_x.numpy()[0, :, :, 0].flatten(),
                   bins=40,
                   alpha=0.4,
                   ec='black')
    axs[1, 0].set_title('Input image Histogram', fontsize=20)
    axs[1, 0].set_yscale('log')
    axs[1, 1].hist(im_y.numpy()[0, :, :, 0].flatten(),
                   bins=40,
                   alpha=0.4,
                   ec='black')
    axs[1, 1].set_title('Reconstructed image Histogram', fontsize=20)
    axs[1, 1].set_yscale('log')

    axs[2, 0].hist(gini_range_in, bins=20)
    axs[2, 0].scatter(gini_value_x,
                      1,
                      marker='+',
                      color='red',
                      s=5e2,
                      linewidths=4,
                      label='Gini factor of image')
    axs[2, 0].set_title(f'Input Gini histogram, value = {gini_value_x}', fontsize=20)
    axs[2, 0].legend(fontsize=20)

    axs[2, 1].hist(gini_range_out, bins=20)
    axs[2, 1].scatter(gini_value_y,
                      1,
                      marker='+',
                      color='red',
                      s=5e2,
                      linewidths=4,
                      label='Gini factor of image')
    axs[2, 1].set_title(f'Reconstructed Gini histogram, value = {gini_value_y}', fontsize=20)
    axs[2, 1].legend(fontsize=20)

    return fig


# Input versus output image, along with the computation of the Gini value of the batch (as histograms) and the value of the given object

def display_gini(im_x, im_y):
    """
    Return the figure displaying the original image extracted from the dataset and its prediction generated through the model, along with the histograms of both images. 

    :param im_x: Original image from the Tensorflow dataset.
    :param im_y: Predicted image by the model.
    :param epoch: Number of the current iteration of the model.
    """

    nb_batch = len(im_y.numpy())  # Number of images inside the batch considered

    gini_range_in, mean_gini_in = gini_array(im_x, nb_batch)  # Array of Gini values for each image in the input batch
    gini_range_out, mean_gini_out = gini_array(im_y, nb_batch)  # Array of Gini values for each image in the input batch

    # Im to plot
    im_x_to_show = im_x.numpy()[0, :, :, 0]
    im_y_to_show = im_y.numpy()[0, :, :, 0]

    # Gini values of im to plot
    gini_value_x = gini_coeff(im_x_to_show)
    gini_value_x = int(gini_value_x) + (float(str(gini_value_x).split('.')[1][:2]))/1e2

    gini_value_y = gini_coeff(im_y_to_show)
    gini_value_y = int(gini_value_y) + (float(str(gini_value_y).split('.')[1][:2]))/1e2

    # Plotting
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=2.0)

    axs[0, 0].imshow(im_x.numpy()[0, :, :, 0])  # features is here the input
    axs[0, 0].set_title('Input image', fontsize=10)

    axs[0, 1].imshow(im_y.numpy()[0, :, :, 0])  # p is the prediction made by the model at the given epoch
    axs[0, 1].set_title('Reconstructed image', fontsize=10)

    axs[1, 0].hist(gini_range_in, bins=20)
    axs[1, 0].scatter(gini_value_x,
                      1,
                      marker='+',
                      color='red',
                      label='Image Gini factor')
    axs[1, 0].set_title(f'Input Gini histogram, value = {gini_value_x}', fontsize=10)
    axs[1, 0].legend()

    axs[1, 1].hist(gini_range_out, bins=20)
    axs[1, 1].scatter(gini_value_y,
                      1,
                      marker='+',
                      color='red',
                      label='Image Gini factor')
    axs[1, 1].set_title(f'Reconstructions Gini histogram, value = {gini_value_y}', fontsize=10)
    axs[1, 1].legend()

    return fig


# Input versus TF versus Output, giving the information about the impact of the Fourier transform on the reconstruction

def display_fourier(im_x, im_y, R, alpha, zeta, intensity, shape):

    mask = create_asymmetric_ring_mask(R, alpha, zeta, intensity, shape)

    imx_to_plot = tf.cast(im_x[..., 0], dtype=tf.float32)
    imy_to_plot = tf.cast(im_y[..., 0], dtype=tf.float32)

    fft_x = transform_from_fft(imx_to_plot, mask)  # Weighted FFT
    fft_y = fast_fourier_shift(imy_to_plot)  # Non weighted FFT

    rec_x = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(fft_x)))  # Inverse Fourier Transform of weighted FFT 
    rec_y = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(fft_y)))  # Inverse Fourier Transform of non weighted FFT 

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    axs[0, 0].imshow(imx_to_plot.numpy()[0, :, :])
    axs[0, 0].set_title('Input image', fontsize=10)
    axs[0, 1].imshow(imy_to_plot.numpy()[0, :, :])
    axs[0, 1].set_title('Reconstructed image', fontsize=10)

    axs[1, 0].imshow(rec_x.numpy()[0, :, :])
    axs[1, 0].set_title('Inverse FFT of input image', fontsize=10)
    axs[1, 1].imshow(rec_y.numpy()[0, :, :])
    axs[1, 1].set_title('Inverse FFT of reconstructed image', fontsize=10)

    return fig


# Display of the UMAP object for three values of the labels : duality, separation and redshift

def display_umap(type_loss, latent_dim, num_run):

    embedding, labels = umap_test(type_loss, latent_dim, num_run)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))

    a = ax[0].scatter(embedding[:, 0], embedding[:, 1], c=labels[2], alpha=0.5)
    ax[0].set_title('UMAP with duality', fontsize=15)
    ax[0].legend()

    handles, legends = a.legend_elements(prop="colors")
    legend = ax[0].legend(handles, legends, loc="upper right", title="Duality")

    b = ax[1].scatter(embedding[:, 0], embedding[:, 1], c=labels[1], alpha=0.5)
    ax[1].set_title('UMAP with separation', fontsize=15)

    c = ax[2].scatter(embedding[:, 0], embedding[:, 1], c=labels[0], alpha=0.5)
    ax[2].set_title('UMAP with redshift', fontsize=15)

    plt.colorbar(b)
    plt.colorbar(c)

    return fig


# Find parameters associated with profiles : Intensity, FWHM, separations

def estimate_coef(x, y):

    x = np.array(x)
    y = np.array(y)
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return b_0, b_1


def fit_gauss_profiles(y, amplitude, center, sigma, nb_gauss, single):

    x = np.linspace(0, len(y), len(y))

    if nb_gauss == 1:  # Only one peak
        model = GaussianModel(prefix='p1_')
        params = model.make_params(p1_amplitude=amplitude[0],
                                   p1_center=center[0],
                                   p1_sigma=sigma[0],
                                   c=10)

        result = model.fit(y, params, x=x, weights=1.0)

    elif single is True:
        cx = 128  # Center of profile
        dists = np.array([np.abs(cx-center[i]) for i in range(len(center))])

        j = np.where(dists == min(dists))[0][0]  # Closest peak to center

        model = GaussianModel(prefix='p1_')
        params = model.make_params(p1_amplitude=amplitude[j],
                                   p1_center=center[j],
                                   p1_sigma=sigma[j],
                                   c=10)

        result = model.fit(y, params, x=x, weights=1.0)

    else:
        model = GaussianModel(prefix='p1_') + GaussianModel(prefix='p2_')
        params = model.make_params(p1_amplitude=amplitude[0],
                                   p1_center=center[0],
                                   p1_sigma=sigma[0],
                                   c=10,
                                   p2_amplitude=amplitude[1],
                                   p2_center=center[1],
                                   p2_sigma=sigma[1])

        result = model.fit(y, params, x=x, weights=1.0)

    return result.best_fit


def identify_peaks(profile):

    peaks_, properties_ = find_peaks(profile, height=1.)  # Position
    fwhm = peak_widths(profile, peaks_, rel_height=0.5)[0]  # FWHM

    if len(peaks_) == 0:
        # print('No peaks found in profile')
        peak_height = properties_['peak_heights']
        peak_pres = False

    else:
        peak_pres = True

        # peak_height = properties_['peak_heights']  ### Valeur mediane des 5 plus haut / Area under the profiles 
        peak_height = []  # Intensity

        for i in range(len(peaks_)):

            peaks_around = [peaks_[i]-2, peaks_[i]-1, peaks_[i], peaks_[i]+1, peaks_[i]+2]
            heigths = [profile[j-1] for j in peaks_around]
            # peak_height.append(np.mean(heigths))
            peak_height.append(np.median(heigths))

    return peak_pres, peaks_, peak_height, fwhm


def measure_light_peaks(dual_profile, majax_profile, single):

    peak_pres_dual, peaks_dual, peak_height_dual, fwhm_dual = identify_peaks(dual_profile)
    peak_pres_majax, peaks_majax, peak_height_majax, fwhm_majax = identify_peaks(majax_profile)

    if peak_pres_dual is True and peak_pres_majax is True:
        nb_gauss_dual = len(peaks_dual)
        nb_gauss_majax = len(peaks_majax)

        fit_dual = fit_gauss_profiles(dual_profile,
                                      peak_height_dual,
                                      peaks_dual,
                                      fwhm_dual,
                                      nb_gauss_dual,
                                      single)
        fit_majax = fit_gauss_profiles(majax_profile,
                                       peak_height_majax,
                                       peaks_majax,
                                       fwhm_majax,
                                       nb_gauss_majax,
                                       single)

        param_dual = identify_peaks(fit_dual)
        param_majax = identify_peaks(fit_majax)

    else:
        param_dual = 0
        param_majax = 0

    return param_dual, param_majax


def input_rec_param_comp(im_x, im_y, batch, save):

    count = 0

    # --- Get all points from batch ---

    sep_dual_in = []
    sep_dual_out = []

    sep_majax_in = []
    sep_majax_out = []

    for i in range(len(im_x)):
        im_x_to_show = im_x.numpy()[i, :, :, 0]
        im_y_to_show = im_y.numpy()[i, :, :, 0]

        regions, validity = regions_img(im_x_to_show)

        if validity is True:

            dual_profile_in, x_dual_s, x_dual_e, y_dual_s, y_dual_e = profile_dual(im_x_to_show,
                                                                                   regions,
                                                                                   sigma=1,
                                                                                   smooth=True)
            majax_profile_in, x_majax_s, x_majax_e, y_majax_s, y_majax_e = profile_maj_axis(im_x_to_show,
                                                                                            regions,
                                                                                            sigma=1,
                                                                                            smooth=True)

            dual_profile_out = profile_line(im_y_to_show,
                                            (x_dual_s, y_dual_s),
                                            (x_dual_e, y_dual_e))
            majax_profile_out = profile_line(im_y_to_show,
                                             (x_majax_s, y_majax_s),
                                             (x_majax_e, y_majax_e))

            # --- Computation of separations --- #

            param_dual_in, param_majax_in = measure_light_peaks(dual_profile_in,
                                                                majax_profile_in,
                                                                single=False)
            param_dual_out, param_majax_out = measure_light_peaks(dual_profile_out,
                                                                  majax_profile_out,
                                                                  single=False)

            if param_dual_in == 0 or param_dual_out == 0 or param_majax_in == 0 or param_majax_out == 0:
                continue
            elif param_dual_in[0] is False or param_dual_out[0] is False or param_majax_in[0] is False or param_majax_out[0] is False:
                continue
            elif len(param_dual_in[2]) != len(param_dual_out[2]) or len(param_majax_in[2]) != len(param_majax_out[2]):
                continue

            if len(param_dual_in[1]) >= 2 and len(param_dual_out[1]) >= 2:

                param_sep_din = abs(param_dual_in[1][0]-param_dual_in[1][1])
                param_sep_dout = abs(param_dual_out[1][0]-param_dual_out[1][1])

                sep_dual_in.append(param_sep_din)
                sep_dual_out.append(param_sep_dout)

            if len(param_majax_in[1]) >= 2 and len(param_majax_out[1]) >=2:

                param_sep_min = abs(param_majax_in[1][0]-param_majax_in[1][1])
                param_sep_mout = abs(param_majax_out[1][0]-param_majax_out[1][1])

                sep_majax_in.append(param_sep_min)
                sep_majax_out.append(param_sep_mout)

            # --- Computation of FWHM and heights --- #

            param_dual_in, param_majax_in = measure_light_peaks(dual_profile_in,
                                                                majax_profile_in,
                                                                single=True)
            param_dual_out, param_majax_out = measure_light_peaks(dual_profile_out,
                                                                  majax_profile_out,
                                                                  single=True)

            if param_dual_in == 0 or param_dual_out == 0 or param_majax_in == 0 or param_majax_out == 0:
                continue
            elif param_dual_in[0] is False or param_dual_out[0] is False or param_majax_in[0] is False or param_majax_out[0] is False:
                continue
            elif len(param_dual_in[2]) != len(param_dual_out[2]) or len(param_majax_in[2]) != len(param_majax_out[2]):
                continue

            if count == 0:

                intensities_dual_in = param_dual_in[2]
                intensities_dual_out = param_dual_out[2]
                intensities_majax_in = param_majax_in[2]
                intensities_majax_out = param_majax_out[2]

                FWHM_dual_in = param_dual_in[3]
                FWHM_dual_out = param_dual_out[3]
                FWHM_majax_in = param_majax_in[3]
                FWHM_majax_out = param_majax_out[3]

            else:

                intensities_dual_in = np.concatenate((intensities_dual_in, param_dual_in[2]), axis=0)
                intensities_dual_out = np.concatenate((intensities_dual_out, param_dual_out[2]), axis=0)
                intensities_majax_in = np.concatenate((intensities_majax_in, param_majax_in[2]), axis=0)
                intensities_majax_out = np.concatenate((intensities_majax_out, param_majax_out[2]), axis=0)

                FWHM_dual_in = np.concatenate((FWHM_dual_in, param_dual_in[3]), axis=0)
                FWHM_dual_out = np.concatenate((FWHM_dual_out, param_dual_out[3]), axis=0)
                FWHM_majax_in = np.concatenate((FWHM_majax_in, param_majax_in[3]), axis=0)
                FWHM_majax_out = np.concatenate((FWHM_majax_out, param_majax_out[3]), axis=0)

            count = count + 1

    sep_dual_in = np.array(sep_dual_in)
    sep_dual_out = np.array(sep_dual_out)
    sep_majax_in = np.array(sep_majax_in)
    sep_majax_out = np.array(sep_majax_out)

    print("Information is stocked in arrays of length", len(intensities_dual_in), len(intensities_dual_out))
    if save is True:
        params_comp = {'int dual in': intensities_dual_in,
                       'int dual out': intensities_dual_out,
                       'int majax in': intensities_majax_in,
                       'int majax out': intensities_majax_out,
                       'fwhm dual in': FWHM_dual_in,
                       'fwhm dual out': FWHM_dual_out,
                       'fwhm majax in': FWHM_majax_in,
                       'fwhm majax out': FWHM_majax_out}

        with open(f'/home/cmahe/dual_agn/pickle_files/params_comp_batch_{batch}.pkl', 'wb') as f:
            pickle.dump(params_comp, f)

    # --- Relative errors --- Percentages ?

    err_intensities_dual = [(intensities_dual_out[i] - intensities_dual_in[i])/intensities_dual_in[i] for i in range(len(intensities_dual_in))]
    err_intensities_majax = [(intensities_majax_out[i] - intensities_majax_in[i])/intensities_majax_in[i] for i in range(len(intensities_majax_in))]

    print('Median errors on int are :', np.median(err_intensities_dual), '+/-', np.std(err_intensities_dual), 'and', np.median(err_intensities_majax), '+/-', np.std(err_intensities_majax))

    err_fwhm_dual = [(FWHM_dual_out[i] - FWHM_dual_in[i])/FWHM_dual_in[i] for i in range(len(FWHM_dual_in))]
    err_fwhm_majax = [(FWHM_majax_out[i] - FWHM_majax_in[i])/FWHM_majax_in[i] for i in range(len(FWHM_majax_in))]

    print('Median errors on fwhm are :', np.median(err_fwhm_dual), '+/-', np.std(err_fwhm_dual), 'and', np.median(err_fwhm_majax), '+/-', np.std(err_fwhm_majax))

    err_sep_dual = [(sep_dual_out[i] - sep_dual_in[i])/sep_dual_in[i] for i in range(len(sep_dual_in))]
    err_sep_majax = [(sep_majax_out[i] - sep_majax_in[i])/sep_majax_in[i] for i in range(len(sep_majax_in))]

    print('Median errors on sep are :', np.median(err_sep_dual), '+/-', np.std(err_sep_dual), 'and', np.median(err_sep_majax), '+/-', np.std(err_sep_majax))

    # --- Plots ---

    fig, ax = plt.subplots(3, 2, figsize=(18, 22))

    # Intensity

    ax[0, 0].scatter(intensities_dual_in,
                     intensities_dual_out,
                     marker='.',
                     color='red',
                     label='Dual')
    ax[0, 0].scatter(intensities_majax_in,
                     intensities_majax_out,
                     marker='+',
                     color='deepskyblue',
                     label='Maj-ax')
    # ax[0, 0].plot(intensities_majax_in, int_majax_pred, '-.', color='black', alpha=0.5, label='Maj-ax LR')
    ax[0, 0].set_title('Peaks heights', fontsize=20)
    ax[0, 0].set_xlabel('Input images', fontsize=18)
    ax[0, 0].set_ylabel('Reconstructions', fontsize=18)
    ax[0, 0].set_xlim(min(min(intensities_dual_in), min(intensities_majax_in)),
                      max(max(intensities_dual_in), max(intensities_majax_in)))
    ax[0, 0].set_ylim(min(min(intensities_dual_in), min(intensities_majax_in)),
                      max(max(intensities_dual_in), max(intensities_majax_in)))

    x = np.linspace(min(min(intensities_dual_in), min(intensities_majax_in)), max(max(intensities_dual_in), max(intensities_majax_in)), 1000)
    ax[0, 0].plot(x, x, '-.', color='black', alpha=0.5)

    ax[0, 0].tick_params(axis='both', labelsize=15)
    ax[0, 0].legend(fontsize=18)

    lim_int_min = min(min(err_intensities_dual), min(err_intensities_majax))
    lim_int_max = max(max(err_intensities_dual), max(err_intensities_majax))

    bins_int = np.linspace(lim_int_min, lim_int_max, 50)

    ax[0, 1].hist(err_intensities_dual,
                  bins=bins_int,
                  facecolor='red',
                  edgecolor='red',
                  linestyle='-',
                  linewidth=2.,
                  alpha=0.5,
                  label='Dual')
    ax[0, 1].hist(err_intensities_majax,
                  bins=bins_int,
                  facecolor='deepskyblue',
                  edgecolor='deepskyblue',
                  linestyle=':',
                  linewidth=2.,
                  alpha=0.5,
                  label='Maj-ax')
    ax[0, 1].set_xlabel('Relative error', fontsize=18)
    ax[0, 1].set_yscale('log')
    # ax[0, 1].set_xlim(min(min(err_intensities_dual), min(err_intensities_majax)), :)
    ax[0, 1].set_title('Distribution of error on peaks heights', fontsize=20)
    ax[0, 1].tick_params(axis='both', labelsize=15)
    ax[0, 1].legend(fontsize=18)

    # FWHM

    ax[1, 0].scatter(FWHM_dual_in,
                     FWHM_dual_out,
                     marker='.',
                     color='red',
                     label='Dual')
    ax[1, 0].scatter(FWHM_majax_in,
                     FWHM_majax_out,
                     marker='+',
                     color='deepskyblue',
                     label='Maj-ax')
    # ax[1, 0].plot(FWHM_majax_in, fwhm_majax_pred, '-.', color='black', alpha=0.5, label='Maj-ax LR')
    ax[1, 0].set_title('Peaks FWHM', fontsize=20)
    ax[1, 0].set_xlabel('Input images', fontsize=18)
    ax[1, 0].set_ylabel('Reconstructions', fontsize=18)
    ax[1, 0].set_xlim(min(min(FWHM_dual_in), min(FWHM_majax_in)),
                      max(max(FWHM_dual_in), max(FWHM_majax_in)))
    ax[1, 0].set_ylim(min(min(FWHM_dual_in), min(FWHM_majax_in)),
                      max(max(FWHM_dual_in), max(FWHM_majax_in)))

    x = np.linspace(min(min(FWHM_dual_in), min(FWHM_majax_in)), max(max(FWHM_dual_in), max(FWHM_majax_in)), 1000)
    ax[1, 0].plot(x, x, '-.', color='black', alpha=0.5)

    ax[1, 0].tick_params(axis='both', labelsize=15)
    ax[1, 0].legend(fontsize=18)

    lim_f_min = min(min(err_fwhm_dual), min(err_fwhm_majax))
    lim_f_max = max(max(err_fwhm_dual), max(err_fwhm_majax))

    bins_fwhm = np.linspace(lim_f_min, lim_f_max, 50)

    ax[1, 1].hist(err_fwhm_dual,
                  bins=bins_fwhm,
                  facecolor='red',
                  edgecolor='red',
                  linestyle='-',
                  linewidth=2.,
                  alpha=0.5,
                  label='Dual')
    ax[1, 1].hist(err_fwhm_majax,
                  bins=bins_fwhm,
                  facecolor='deepskyblue',
                  edgecolor='deepskyblue',
                  linestyle=':',
                  linewidth=2.,
                  alpha=0.5,
                  label='Maj-ax')
    ax[1, 1].set_xlabel('Relative error', fontsize=18)
    ax[1, 1].set_yscale('log')
    # ax[1, 1].set_xlim(min(min(err_fwhm_dual), min(err_fwhm_majax)), 10.0)
    ax[1, 1].set_title('Distribution of error on peaks FWHM', fontsize=20)
    ax[1, 1].tick_params(axis='both', labelsize=15)
    ax[1, 1].legend(fontsize=18)

    # Separations

    ax[2, 0].scatter(sep_dual_in,
                     sep_dual_out,
                     marker='.',
                     color='red',
                     label='Dual')
    ax[2, 0].scatter(sep_majax_in,
                     sep_majax_out,
                     marker='+',
                     color='deepskyblue',
                     label='Maj-ax')
    # ax[2, 0].plot(sep_majax_in, sep_majax_pred, '-.', color='black', alpha=0.5, label='Maj-ax LR')
    ax[2, 0].set_title('Peak separations', fontsize=20)
    ax[2, 0].set_xlabel('Input images', fontsize=18)
    ax[2, 0].set_ylabel('Reconstructions', fontsize=18)
    ax[2, 0].set_xlim(min(min(sep_dual_in), min(sep_majax_in)), max(max(sep_dual_in), max(sep_majax_in)))
    ax[2, 0].set_ylim(min(min(sep_dual_in), min(sep_majax_in)), max(max(sep_dual_in), max(sep_majax_in)))

    x = np.linspace(min(min(sep_dual_in), min(sep_majax_in)), max(max(sep_dual_in), max(sep_majax_in)), 1000)
    ax[2, 0].plot(x, x, '-.', color='black', alpha=0.5)

    ax[2, 0].tick_params(axis='both', labelsize=15)
    ax[2, 0].legend(fontsize=18)

    lim_sep_min = min(min(err_sep_dual), min(err_sep_majax))
    lim_sep_max = max(max(err_sep_dual), max(err_sep_majax))

    bins_sep = np.linspace(lim_sep_min, lim_sep_max, 50)

    ax[2, 1].hist(err_sep_dual,
                  bins=bins_sep,
                  facecolor='red',
                  edgecolor='red',
                  linestyle='-',
                  linewidth=2.,
                  alpha=0.5,
                  label='Dual')
    ax[2, 1].hist(err_sep_majax,
                  bins=bins_sep,
                  facecolor='deepskyblue',
                  edgecolor='deepskyblue',
                  linestyle=':',
                  linewidth=2.,
                  alpha=0.5,
                  label='Maj-ax')
    ax[2, 1].set_xlabel('Relative error', fontsize=18)
    ax[2, 1].set_yscale('log')
    # ax[2, 1].set_xlim((min(min(err_sep_dual), min(err_sep_majax)), 10.0))
    ax[2, 1].set_title('Distribution of error on peaks separations', fontsize=20)
    ax[2, 1].tick_params(axis='both', labelsize=15)
    ax[2, 1].legend(fontsize=18)

    return fig

# Input versus output image with two profiles detected (dual and maj_axis)


def profile_relative_diff(im_x, im_y):

    # Im to plot
    im_x_to_show = im_x.numpy()[0, :, :, 0]
    im_y_to_show = im_y.numpy()[0, :, :, 0]

    regions, validity = regions_img(im_x_to_show)

    if validity is True:

        dual_profile_in, x_dual_s, x_dual_e, y_dual_s, y_dual_e = profile_dual(im_x_to_show,
                                                                               regions,
                                                                               sigma=1,
                                                                               smooth=True)
        majax_profile_in, x_majax_s, x_majax_e, y_majax_s, y_majax_e = profile_maj_axis(im_x_to_show,
                                                                                        regions,
                                                                                        sigma=1,
                                                                                        smooth=True)

        dual_profile_out = profile_line(im_y_to_show, (x_dual_s, y_dual_s), (x_dual_e, y_dual_e))
        majax_profile_out = profile_line(im_y_to_show, (x_majax_s, y_majax_s), (x_majax_e, y_majax_e))

        # Fig ------------------------------------------------------

        fig = plt.figure(layout="constrained", figsize=(30, 25))
        subfigs = fig.subfigures(2, 1)

        ax_image = subfigs[0].subplots(1, 2)
        ax_profiles = subfigs[1].subplots(1, 2)

        ax_image[0].imshow(im_x_to_show)
        ax_image[0].set_title("Input image with profile lines", fontsize=25)
        ax_image[0].plot(
            (y_dual_s, y_dual_e),
            (x_dual_s, x_dual_e),
            color="red",
            linestyle="solid",
            linewidth=5.0,
            label="Dual",
        )
        ax_image[0].plot(
            (y_majax_s, y_majax_e),
            (x_majax_s, x_majax_e),
            color="deepskyblue",
            linestyle="dashed",
            linewidth=5.0,
            label="Maj-ax",
        )
        ax_image[0].legend(fontsize=20)
        ax_image[0].tick_params(axis='both', labelsize=18)

        ax_image[1].imshow(im_y_to_show)
        ax_image[1].set_title("Reconstructed image with profile lines", fontsize=25)
        ax_image[1].plot(
            (y_dual_s, y_dual_e),
            (x_dual_s, x_dual_e),
            color="darkorange",
            linestyle="dashdot",
            linewidth=5.0,
            label="Dual",
        )
        ax_image[1].plot(
            (y_majax_s, y_majax_e),
            (x_majax_s, x_majax_e),
            color="turquoise",
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linewidth=5.0,
            label="Maj-ax",
        )  # Linestyle is dashdotdotted
        ax_image[1].legend(fontsize=20)
        ax_image[1].tick_params(axis='both', labelsize=18)

        # -------------------------------------------------------------

        ax_profiles[0].plot(
            dual_profile_in,
            color="red",
            linestyle="solid",
            linewidth=2.5,
            label="Input image",
        )
        ax_profiles[0].plot(
            dual_profile_out,
            color="darkorange",
            linestyle="dashdot",
            linewidth=2.5,
            label="Reconstruction",
        )
        ax_profiles[0].set_title('Dual profiles on the input images and its reconstruction', fontsize=25)
        ax_profiles[0].set_ylabel('Pixel value', fontsize=20)
        ax_profiles[0].tick_params(axis='both', labelsize=18)
        ax_profiles[0].legend(fontsize=20)

        # --------------------------------------------------------------------

        ax_profiles[1].plot(
            majax_profile_in,
            color="deepskyblue",
            linestyle="dashed",
            linewidth=2.5,
            label="Input image",
        )
        ax_profiles[1].plot(
            majax_profile_out,
            color="turquoise",
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linewidth=2.5,
            label="Reconstruction",
        )
        ax_profiles[1].set_title('Major axis profiles on the input image and its reconstruction', fontsize=25)
        ax_profiles[1].set_ylabel('Pixel value', fontsize=20)
        ax_profiles[1].tick_params(axis='both', labelsize=18)
        ax_profiles[1].legend(fontsize=20)

    else:
        print('No profile for this image : Error = too faint')
        fig = plt.figure()
        plt.title('No profile : empty image')

    return fig, validity


'''
Plot Flow

'''

# Fig profile on gen images, with generation scores and UMAP


def profile_gen(im_to_show, x_pos, reducer, score):

    embedding = reducer.embedding_

    x_pos = np.array([x_pos])
    trans_x_pos = reducer.transform(x_pos)

    # --------
    regions, validity = regions_img(im_to_show)

    if validity is True:

        dual_profile, x_dual_s, x_dual_e, y_dual_s, y_dual_e = profile_dual(im_to_show,
                                                                            regions,
                                                                            sigma=1)
        majax_profile, x_majax_s, x_majax_e, y_majax_s, y_majax_e = profile_maj_axis(im_to_show,
                                                                                     regions,
                                                                                     sigma=1)

        # --------------------------------------------------------------------------

        fig = plt.figure(layout='constrained', figsize=(20, 20))
        subfigs = fig.subfigures(2, 1)

        ax_image = subfigs[0].subplots(1, 2)
        ax_profiles = subfigs[1].subplots(1,1)

        ax_image[1].imshow(im_to_show)
        ax_image[1].plot((y_dual_s, y_dual_e),
                         (x_dual_s, x_dual_e),
                         color='red',
                         linestyle='solid',
                         linewidth=5.,
                         label='Dual')
        ax_image[1].plot((y_majax_s, y_majax_e),
                         (x_majax_s, x_majax_e),
                         color='deepskyblue',
                         linestyle='dashdot',
                         linewidth=5.,
                         label='Maj-ax')
        ax_image[1].set_title('Generated image with profile lines detected', fontsize=28)
        ax_image[1].tick_params(axis='both', labelsize=18)
        ax_image[1].legend(fontsize=22)

        ax_image[0].scatter(embedding[:, 0],
                            embedding[:, 1],
                            marker='.',
                            color='b',
                            label='UMAP of the LS distribution')
        ax_image[0].scatter(trans_x_pos[0, 0],
                            trans_x_pos[0, 1],
                            marker='+',
                            color='r',
                            label='Sample position',
                            s=1e3,
                            linewidths=5.)
        ax_image[0].set_title(rf'Position of sample in latent space, $S_g =$ {score}', fontsize=28)
        ax_image[0].tick_params(axis='both', labelsize=18)
        ax_image[0].legend(fontsize=22)

        ax_profiles.plot(dual_profile,
                         color='red',
                         linestyle='solid',
                         linewidth=2.,
                         label='Dual profile')
        ax_profiles.plot(majax_profile,
                         color='deepskyblue',
                         linestyle='dashdot',
                         linewidth=2.,
                         label='Maj-ax profile')
        ax_profiles.set_title('Profiles of generated image', fontsize=28)
        ax_profiles.set_ylabel('Pixel value', fontsize=20)
        ax_profiles.tick_params(axis='both', labelsize=18)
        ax_profiles.legend(fontsize=22)

    else:
        print('No profile for this image : Error = too faint')

        fig = plt.figure(layout='constrained', figsize=(20, 20))
        subfigs = fig.subfigures(2, 1)

        ax_image = subfigs[0].subplots(1, 2)
        ax_profiles = subfigs[1].subplots(1,1)

        ax_image[1].imshow(im_to_show)
        ax_image[1].set_title('Generated image without profile lines', fontsize=28)
        ax_image[1].tick_params(axis='both', labelsize=18)
        ax_image[1].legend(fontsize=22)

        ax_image[0].scatter(embedding[:, 0],
                            embedding[:, 1],
                            marker='.',
                            color='b',
                            label='UMAP of the LS distribution')
        ax_image[0].scatter(trans_x_pos[0, 0],
                            trans_x_pos[0, 1],
                            marker='+',
                            color='r',
                            label='Sample position',
                            s=1e3,
                            linewidths=5.)
        ax_image[0].set_title(rf'Position of sample in latent space, $S_g =$ {score}', fontsize=28)
        ax_image[0].tick_params(axis='both', labelsize=18)
        ax_image[0].legend(fontsize=22)

        ax_profiles.set_title('No profile : empty image', fontsize=28)

    return fig, validity


# Fig lot of gen with their generation scores


def score_gen_multiple(gen_im_batch, score_sample, score_pp, nb_img):  # Here gen_im_batch should be numpyied

    nb_to_plot = int(np.sqrt(nb_img))

    fig, ax = plt.subplots(nb_to_plot, nb_to_plot, figsize=(40, 40))

    nb_image = 0

    for i in range(nb_to_plot):
        for j in range(nb_to_plot):

            ax[i, j].imshow(gen_im_batch[nb_image, :, :, 0])
            #ax[i, j].set_title(rf'$S_r$ = {MSE_sample[nb_image]}, $\rho$ = {density_sample[nb_image]}, $S_g$ = {score_sample[nb_image]}', fontsize=25)
            ax[i, j].set_title(rf'$S_g$ = {score_sample[nb_image]}, $P_g$ = {score_pp[nb_image]}', fontsize=23)
            ax[i, j].set_xticks([])  # Remove x-axis ticks
            ax[i, j].set_yticks([])  # Remove y-axis ticks

            nb_image = nb_image + 1

    return fig


def hist_scores(gen_im_batch, score_sample, MSE_sample, density_sample):

    I_img = []
    for i in range(len(gen_im_batch)):
        I_img.append(np.mean(gen_im_batch[i]))

    quant_10 = np.quantile(I_img, 0.1)
    quant_90 = np.quantile(I_img, 0.9)

    scores_below = []
    scores_above = []

    MSE_above = []
    MSE_below = []

    density_above = []
    density_below = []

    for i in range(len(I_img)):
        if I_img[i] <= quant_10:

            scores_below.append(score_sample[i])
            MSE_below.append(MSE_sample[i])
            density_below.append(density_sample[i])

        elif I_img[i] >= quant_90:

            scores_above.append(score_sample[i])
            MSE_above.append(MSE_sample[i])
            density_above.append(density_sample[i])

    index_test = random.sample(range(len(score_sample)), len(scores_below))
    score_test = [score_sample[i] for i in index_test]
    MSE_test = [MSE_sample[i] for i in index_test]
    density_test = [density_sample[i] for i in index_test]

    # --- Plot ---

    print('Lens are ', len(score_sample), len(scores_above), len(scores_below))

    fig, ax = plt.subplots(1, 3, figsize=(25, 10))
    fig.suptitle(r"Distributions of mean reconstruction scores $S_r$, density $\rho$ and generation score $S_g$", fontsize=25)

    lim_MSE_min = min(min(MSE_test), min(MSE_below), min(MSE_above))
    lim_MSE_max = max(max(MSE_test), max(MSE_below), max(MSE_above))

    bins_MSE = np.linspace(lim_MSE_min, lim_MSE_max, 20)

    ax[0].hist(MSE_test,
               bins=bins_MSE,
               facecolor='blue',
               edgecolor='blue',
               linestyle='-',
               linewidth=2.,
               label='Total sample')
    ax[0].hist(MSE_below,
               bins=bins_MSE,
               facecolor='deepskyblue',
               edgecolor='deepskyblue',
               linestyle='--',
               linewidth=2.,
               alpha=0.6,
               label='Faintest')
    ax[0].hist(MSE_above,
               bins=bins_MSE,
               facecolor='violet',
               edgecolor='violet',
               linestyle=':',
               linewidth=2.,
               alpha=0.6,
               label='Brightest')
    ax[0].set_xlabel(r'Mean reconstruction score $S_{r}$', fontsize=20)
    # ax[0].set_title(r'Distributions of $S_{r}$ of faintest and brightest images', fontsize=20)
    # ax[0].set_yscale('log')
    ax[0].set_ylabel('Quantity', fontsize=20)
    ax[0].tick_params(axis='both', labelsize=15)
    ax[0].legend(fontsize=18)

    lim_d_min = min(min(density_test), min(density_below), min(density_above))
    lim_d_max = max(max(density_test), max(density_below), max(density_above))

    bins_d = np.linspace(lim_d_min, lim_d_max, 20)

    ax[1].hist(density_test,
               bins=bins_d,
               facecolor='blue',
               edgecolor='blue',
               linestyle='-',
               linewidth=2.,
               label='Total sample')
    ax[1].hist(density_below,
               bins=bins_d,
               facecolor='deepskyblue',
               edgecolor='deepskyblue',
               linestyle='--',
               linewidth=2.,
               alpha=0.6,
               label='Faintest')
    ax[1].hist(density_above,
               bins=bins_d,
               facecolor='violet',
               edgecolor='violet',
               linestyle=':',
               linewidth=2.,
               alpha=0.6,
               label='Brightest')
    ax[1].set_xlabel(r'Latent space density $\rho$', fontsize=20)
    # ax[1].set_title(r'Distributions of $\rho$ of faintest and brightest images', fontsize=20)
    # ax[1].set_yscale('log')
    # ax[1].set_ylabel('Number of image', fontsize=20)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[1].legend(fontsize=18)

    lim_score_min = min(min(score_test), min(scores_below), min(scores_above))
    lim_score_max = max(max(score_test), max(scores_below), max(scores_above))

    bins_score = np.linspace(lim_score_min, lim_score_max, 20)

    ax[2].hist(score_test,
               bins=bins_score,
               facecolor='blue',
               edgecolor='blue',
               linestyle='-',
               linewidth=2.,
               label='Total sample')
    ax[2].hist(scores_below,
               bins=bins_score,
               facecolor='deepskyblue',
               edgecolor='deepskyblue',
               linewidth=2.,
               linestyle='--',
               alpha=0.6,
               label='Faintest')
    ax[2].hist(scores_above,
               bins=bins_score,
               facecolor='violet',
               edgecolor='violet',
               linestyle=':',
               linewidth=2.,
               alpha=0.6,
               label='Brightest')
    ax[2].set_xlabel(r'Generation score $S_{g}$', fontsize=20)
    # ax[2].set_title(r'Distributions of $S_{g}$ of faintest and brightest images', fontsize=20)
    # ax[2].set_yscale('log')
    # ax[2].set_ylabel('Number of image', fontsize=20)
    ax[2].tick_params(axis='both', labelsize=15)
    ax[2].legend(fontsize=18)

    return fig


def cut_numbers(x, number):
    y = int(x) + (float(str(x).split('.')[1][:number]))/1e2
    return y


def img_density(gen_im_batch, score_sample, MSE_samples, density_sample):

    lim_density = np.quantile(density_sample, 0.05)
    lim_low_MSE = np.quantile(MSE_samples, 0.20)
    lim_high_MSE = np.quantile(MSE_samples, 0.8)

    print('Lim "good" scores', np.quantile(score_sample, 0.95))

    im_low_density = []
    MSE_im = []
    density_im = []
    score_im = []

    for i in range(len(density_sample)):
        if density_sample[i] <= lim_density:
            im_low_density.append(gen_im_batch.numpy()[i, :, :, 0])

            MSE_im.append(MSE_samples[i])
            density_im.append(density_sample[i])
            score_im.append(score_sample[i])

    print(len(im_low_density))

    im_low_high_MSE = []
    im_low_low_MSE = []

    MSE_im_high = []
    density_im_high = []
    score_im_high = []

    MSE_im_low = []
    density_im_low = []
    score_im_low = []

    for i in range(len(im_low_density)):
        if MSE_im[i] <= lim_low_MSE:
            im_low_low_MSE.append(im_low_density[i])

            MSE_im_low.append(MSE_im[i])
            density_im_low.append(density_im[i])
            score_im_low.append(score_im[i])

        elif MSE_im[i] >= lim_high_MSE:
            im_low_high_MSE.append(im_low_density[i])

            MSE_im_high.append(MSE_im[i])
            density_im_high.append(density_im[i])
            score_im_high.append(score_im[i])

    print('high', len(im_low_high_MSE), 'low', len(im_low_low_MSE))

    if len(im_low_high_MSE) >= 3 and len(im_low_low_MSE) >= 3:
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle('Generated images from low-density vectors, with or without high MSE', fontsize=18)

        ax[0, 0].imshow(im_low_high_MSE[0])
        ax[0, 0].set_title(rf'Im MSE={cut_numbers(MSE_im_high[0], 3)}, density={cut_numbers(density_im_high[0], 2)}, $S_g$={cut_numbers(score_im_high[0], 2)}')
        ax[1, 0].imshow(im_low_high_MSE[1])
        ax[1, 0].set_title(rf'Im MSE={cut_numbers(MSE_im_high[1], 3)}, density={cut_numbers(density_im_high[1], 2)}, $S_g$={cut_numbers(score_im_high[1], 2)}')
        ax[2, 0].imshow(im_low_high_MSE[2])
        ax[2, 0].set_title(rf'Im MSE={cut_numbers(MSE_im_high[2], 3)}, density={cut_numbers(density_im_high[2], 2)}, $S_g$={cut_numbers(score_im_high[2], 2)}')

        ax[0, 1].imshow(im_low_low_MSE[0])
        ax[0, 1].set_title(rf'Im MSE={cut_numbers(MSE_im_low[0], 3)}, density={cut_numbers(density_im_low[0], 2)}, $S_g$={cut_numbers(score_im_low[0], 2)}')
        ax[1, 1].imshow(im_low_low_MSE[1])
        ax[1, 1].set_title(rf'Im MSE={cut_numbers(MSE_im_low[1], 3)}, density={cut_numbers(density_im_low[1], 2)}, $S_g$={cut_numbers(score_im_low[1], 2)}')
        ax[2, 1].imshow(im_low_low_MSE[2])
        ax[2, 1].set_title(rf'Im MSE={cut_numbers(MSE_im_low[2], 3)}, density={cut_numbers(density_im_low[2], 2)}, $S_g$={cut_numbers(score_im_low[2], 2)}')

    else:
        fig = plt.figure()
        print('Empty image density')

    return fig


def comp_scores_intensity(gen_im_batch, score_sample, distri):  # Add colourbars

    quant_95 = np.quantile(score_sample, 0.95)
    quant_95 = int(quant_95) + (float(str(quant_95).split('.')[1][:2]))/1e2

    count = 0

    scores_sel = []

    # --- Get all images from batch ---

    for i in range(len(gen_im_batch)):
        im = gen_im_batch.numpy()[i, :, :, 0]

        regions, validity = regions_img(im)

        if validity is True:

            dual_profile, x_dual_s, x_dual_e, y_dual_s, y_dual_e = profile_dual(im, regions, sigma=1)
            majax_profile, x_majax_s, x_majax_e, y_majax_s, y_majax_e = profile_maj_axis(im, regions, sigma=1)

            param_dual, param_majax = measure_light_peaks(dual_profile, majax_profile, single=True)

            if param_dual == 0 or param_majax == 0:
                continue
            elif param_dual[0] is False or param_majax[0] is False:
                continue
            else:
                scores_sel.append(score_sample[i])
                if count == 0:
                    int_dual = param_dual[2]
                    int_majax = param_majax[2]

                    FWHM_dual = param_dual[3]
                    FWHM_majax = param_majax[3]
                else:
                    int_dual = np.concatenate((int_dual, param_dual[2]), axis=0)
                    int_majax = np.concatenate((int_majax, param_majax[2]), axis=0)

                    FWHM_dual = np.concatenate((FWHM_dual, param_dual[3]), axis=0)
                    FWHM_majax = np.concatenate((FWHM_majax, param_majax[3]), axis=0)

            count = count+1

    # --- Plot ---

    bin_sample = np.linspace(min(score_sample), max(score_sample), 20)

    fig_scatter = plt.figure(figsize=(10, 8))

    a = plt.scatter(int_dual,
                    FWHM_dual,
                    s=20,
                    c=scores_sel,
                    marker='o',
                    label='Dual')
    plt.scatter(int_majax,
                FWHM_majax,
                s=20,
                c=scores_sel,
                marker='d',
                label='Maj-ax')
    plt.xlabel('Peak heights', fontsize=16)
    plt.ylabel('Peak FWHMs', fontsize=16)
    # plt.title(r'Distribution of the images parameters as function of the generation score $S_g$', fontsize=18)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(fontsize=15)

    cb = plt.colorbar(a)
    cb.set_label(label=r'Generation score $S_g$', size=16)
    cb.ax.tick_params(labelsize=14)

    if distri is False:

        return fig_scatter

    else:

        fig_distri = plt.figure(figsize=(10, 7))
        plt.hist(score_sample,
                 bins=bin_sample,
                 facecolor='deepskyblue',
                 edgecolor='blue',
                 alpha=0.7)
        # plt.title(r'Distribution of generation score $S_g$', fontsize=20)
        plt.xlabel(r'Generation score $S_g$', fontsize=18)
        plt.ylabel('Quantity', fontsize=18)
        plt.yscale('log')
        plt.tick_params(axis='both', labelsize=15)
        plt.axvline(quant_95, ymin=0, ymax=1)
        plt.text(quant_95, 100, quant_95, size=15, color='black', backgroundcolor='lightgrey')

        return fig_distri


# Comp FWHM/Intensities input, rec, gen

def compute_kde_grid(x, y, grid_size=100):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    X, Y = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size),
        np.linspace(y.min(), y.max(), grid_size)
    )
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z, kde


def kde_contour_levels(Z, levels=[0.1, 0.25, 0.5, 0.75, 0.9]):
    Z_flat = Z.flatten()
    Z_sort = np.sort(Z_flat)[::-1]  # descending
    cumsum = np.cumsum(Z_sort)
    cumsum /= cumsum[-1]
    thresholds = []
    for lev in levels:
        idx = np.searchsorted(cumsum, lev)
        thresholds.append(Z_sort[idx])
    thresholds = np.array(thresholds)
    thresholds.sort()
    thresholds = np.append(thresholds, Z.max())  # fill densest region
    return thresholds


def plot_kde_with_outliers(ax, x, y, color='red', cmap='Reds', label=None,
                           levels=[0.1, 0.25, 0.5, 0.75, 0.9], grid_size=100,
                           fill=True, alpha_fill=0.5, scatter_size=5):
    X, Y, Z, kde = compute_kde_grid(x, y, grid_size)
    thresholds = kde_contour_levels(Z, levels)

    if fill:
        ax.contourf(X, Y, Z, levels=thresholds, cmap=cmap, alpha=alpha_fill)
    ax.contour(X, Y, Z, levels=thresholds, colors=color, linewidths=1)

    densities_at_points = kde(np.vstack([x, y]))
    outlier_mask = densities_at_points < thresholds[0]
    ax.scatter(x[outlier_mask], y[outlier_mask], c=color, s=scatter_size, alpha=0.5, label=label)

    return kde, thresholds


def comp_fwhm_int(gen_im_batch, save):

    count = 0

    # --- Get parameters from all images from gen batch ---

    for i in range(len(gen_im_batch)):
        im = gen_im_batch.numpy()[i, :, :, 0]

        regions, validity = regions_img(im)

        if validity is True:

            dual_profile, x_dual_s, x_dual_e, y_dual_s, y_dual_e = profile_dual(im, regions, sigma=1)
            majax_profile, x_majax_s, x_majax_e, y_majax_s, y_majax_e = profile_maj_axis(im, regions, sigma=1)

            param_dual, param_majax = measure_light_peaks(dual_profile, majax_profile, single=True)

            if param_dual == 0 or param_majax == 0:
                continue
            elif param_dual[0] is False or param_majax[0] is False:
                continue

            if count == 0:

                intensities_dual = param_dual[2]
                intensities_majax = param_majax[2]

                FWHM_dual = param_dual[3]
                FWHM_majax = param_majax[3]

            else:

                intensities_dual = np.concatenate((intensities_dual, param_dual[2]), axis=0)
                intensities_majax = np.concatenate((intensities_majax, param_majax[2]), axis=0)

                FWHM_dual = np.concatenate((FWHM_dual, param_dual[3]), axis=0)
                FWHM_majax = np.concatenate((FWHM_majax, param_majax[3]), axis=0)

            count = count + 1

    if save is True:
        params_comp = {'int dual ': intensities_dual,
                       'int majax ': intensities_majax,
                       'fwhm dual ': FWHM_dual,
                       'fwhm majax': FWHM_majax,
                       }

        with open(f'/home/cmahe/dual_agn/pickle_files/params_comp_gen.pkl', 'wb') as f:
            pickle.dump(params_comp, f)

    # --- Get info from VAE ---

    with open('/home/cmahe/dual_agn/pickle_files/params_comp_batch_1.pkl', 'rb') as f:
        dico_load = pickle.load(f)

        intensities_dual_in = dico_load['int dual in']
        intensities_dual_out = dico_load['int dual out']
        intensities_majax_in = dico_load['int majax in']
        intensities_majax_out = dico_load['int majax out']

        FWHM_dual_in = dico_load['fwhm dual in']
        FWHM_dual_out = dico_load['fwhm dual out']
        FWHM_majax_in = dico_load['fwhm majax in']
        FWHM_majax_out = dico_load['fwhm majax out']

    # --- Plot ---

    intensities_vae_in = np.concatenate((intensities_dual_in, intensities_majax_in), axis=0)
    intensities_vae_out = np.concatenate((intensities_dual_out, intensities_majax_out), axis=0)
    intensities_gen = np.concatenate((intensities_dual, intensities_majax), axis=0)

    FWHM_vae_in = np.concatenate((FWHM_dual_in, FWHM_majax_in), axis=0)
    FWHM_vae_out = np.concatenate((FWHM_dual_out, FWHM_majax_out), axis=0)
    FWHM_gen = np.concatenate((FWHM_dual, FWHM_majax), axis=0)

    fig_comp, ax = plt.subplots(2, 2, figsize=(12, 12))
    # fig_comp.suptitle('Distribution of peak heights and FWHMs for the input images, reconstructions and generations', fontsize=18)

    lim_min_int = min(min(intensities_vae_in), min(intensities_vae_out), min(intensities_gen))
    lim_max_int = max(max(intensities_vae_in), max(intensities_vae_out), max(intensities_gen))
    bins_int = np.linspace(lim_min_int, lim_max_int, 20)

    ax[0, 0].hist(intensities_vae_in,
                  bins=bins_int,
                  histtype='step',
                  edgecolor='blue',
                  linestyle='-',
                  linewidth=2.)
    ax[0, 0].hist(intensities_vae_out,
                  bins=bins_int,
                  histtype='step',
                  edgecolor='red',
                  linestyle='--',
                  linewidth=2.)
    ax[0, 0].hist(intensities_gen,
                  bins=bins_int,
                  histtype='step',
                  edgecolor='green',
                  linestyle=':',
                  linewidth=2.)
    ax[0, 0].set_xlim(0.5, 3.5)
    ax[0, 0].tick_params(axis='both', labelsize=12)
    ax[0, 0].set_ylabel('Quantity', fontsize=15)

    lim_min_FWHM = min(min(FWHM_vae_in), min(FWHM_vae_out), min(FWHM_gen))
    lim_max_FWHM = max(max(FWHM_vae_in), max(FWHM_vae_out), max(FWHM_gen))
    bins_FWHM = np.linspace(lim_min_FWHM, lim_max_FWHM, 20)

    ax[1, 1].hist(FWHM_vae_in,
                  bins=bins_FWHM,
                  histtype='step',
                  edgecolor='blue',
                  linestyle='-',
                  linewidth=2.,
                  orientation='horizontal')
    ax[1, 1].hist(FWHM_vae_out,
                  bins=bins_FWHM,
                  histtype='step',
                  edgecolor='red',
                  linestyle='--',
                  linewidth=2.,
                  orientation='horizontal')
    ax[1, 1].hist(FWHM_gen,
                  bins=bins_FWHM,
                  histtype='step',
                  edgecolor='green',
                  linestyle=':',
                  linewidth=2.,
                  orientation='horizontal')
    ax[1, 1].set_ylim(0, 180)
    ax[1, 1].tick_params(axis='both', labelsize=12)
    ax[1, 1].set_xlabel('Quantity', fontsize=15)

    plot_kde_with_outliers(ax[1, 0],
                           intensities_vae_in,
                           FWHM_vae_in,
                           color='red',
                           cmap='Reds',
                           label='H-AGN input')
    plot_kde_with_outliers(ax[1, 0],
                           intensities_vae_out,
                           FWHM_vae_out,
                           color='blue',
                           cmap='Blues',
                           label='Reconstructions')
    plot_kde_with_outliers(ax[1, 0],
                           intensities_gen,
                           FWHM_gen,
                           color='green',
                           cmap='Greens',
                           label='Generations')

    ax[1, 0].set_xlabel('Peak heights', fontsize=15)
    ax[1, 0].set_ylabel('Peak FWHMs', fontsize=15)
    ax[1, 0].set_xlim(0.5, 3.5)
    ax[1, 0].set_ylim(0, 180)
    ax[1, 0].tick_params(axis='both', labelsize=12)

    patch_r = mpatches.Patch(color='red', alpha=0.5, label='H-AGN input')
    patch_b = mpatches.Patch(color='blue', alpha=0.5, label='Reconstructions')
    patch_g = mpatches.Patch(color='green', alpha=0.5, label='Generations')

    ax[1, 0].legend(handles=[patch_r, patch_b, patch_g], fontsize=12)
    fig_comp.delaxes(ax[0][1])

    return fig_comp


# Plot loss of flow training

def display_flow_loss(train_losses, valid_losses, n_run):  # train_losses is a list

    fig = plt.figure()

    plt.plot(train_losses, color='blue', linestyle='dashed', label='Training loss')
    plt.plot(valid_losses, color='red', linestyle='dashdot', label='Validation loss')
    plt.xlabel('Training epoch of the model')
    # plt.yscale('log')
    plt.ylabel('Value of the flow loss function')
    # plt.title(f'Loss function of flow, run {n_run}')
    plt.grid(alpha=0.5)
    plt.legend()

    return fig

# Power spectra


def plot_power_spectra(decoded_VAE_img, decoded_gauss_img, decoded_flow_img):

    VAE_PP = score_power_spectrum(decoded_VAE_img,
                                  low_limits=[1, 50],
                                  high_limits=[50, 80],
                                  score=False,
                                  log=False)[0]

    m_vae_pp = np.mean(VAE_PP, axis=0)
    s_vae_pp = np.std(VAE_PP, axis=0)
    x_vae = np.arange(VAE_PP.shape[1])

    Gauss_PP = score_power_spectrum(decoded_gauss_img,
                                    low_limits=[1, 50],
                                    high_limits=[50, 80],
                                    score=False,
                                    log=False)[0]

    m_gauss_pp = np.mean(Gauss_PP, axis=0)
    s_gauss_pp = np.std(Gauss_PP, axis=0)
    x_gauss = np.arange(Gauss_PP.shape[1])

    Flow_PP = score_power_spectrum(decoded_flow_img,
                                   low_limits=[1, 50],
                                   high_limits=[50, 80],
                                   score=False,
                                   log=False)[0]

    m_flow_pp = np.mean(Flow_PP, axis=0)
    s_flow_pp = np.std(Flow_PP, axis=0)
    x_flow = np.arange(Flow_PP.shape[1])

    fig = plt.figure(figsize=(7, 7))

    plt.plot(x_vae, m_vae_pp, color='red', label='HAGN images')
    plt.fill_between(x_vae, m_vae_pp - s_vae_pp, m_vae_pp + s_vae_pp, alpha=0.3, color='red')

    plt.plot(x_gauss, m_gauss_pp, color='blue', label='Gauss sample')
    plt.fill_between(x_gauss, m_gauss_pp - s_gauss_pp, m_gauss_pp + s_gauss_pp, alpha=0.3, color='blue')

    plt.plot(x_flow, m_flow_pp, color='green', label='Flow sample')
    plt.fill_between(x_flow, m_flow_pp - s_flow_pp, m_flow_pp + s_flow_pp, alpha=0.3, color='green')

    plt.xlabel('Spatial frequencies')
    plt.ylabel('Power')
    plt.legend()

    return fig


# Corner plot, transformed Gaussian in LT


def corner_flow_lt(samples, latent_dim, encod_np, nb_dim_to_plot):

    print('Nb encod = ', len(encod_np))

    im_codes = []
    sample_codes = []

    for i in range(nb_dim_to_plot):
        im_codes.append(encod_np[:, i])
        sample_codes.append(samples[:, i])

    im_codes = np.array(im_codes).T
    sample_codes = np.array(sample_codes).T

    ndim = im_codes.shape[1]

    ranges = []
    for i in range(ndim):
        xmin = min(im_codes[:, i].min(), sample_codes[:, i].min())
        xmax = max(im_codes[:, i].max(), sample_codes[:, i].max())

        # small padding
        pad = 0.02 * (xmax - xmin)
        ranges.append((xmin - pad, xmax + pad))

    try:
        fig_corner_flow = corner.corner(im_codes, bins=20, range=ranges, color = 'blue', hist_kwargs={"density": True})
        corner.corner(sample_codes, bins=20, range=ranges, fig = fig_corner_flow, color = 'red', hist_kwargs={"density": True})

        axes = np.array(fig_corner_flow.axes)
        axes = axes.reshape(int(np.sqrt(len(axes))), int(np.sqrt(len(axes))))
        axes[0,1].scatter([],[], color = 'blue', label = r"$z \sim \mathcal{Z}$")
        axes[0,1].scatter([], [], color = 'red', label = r"$z' = g(z_{\mathcal{Q}})$")
        axes[0,1].legend(loc = 'lower left', fontsize = 20)

        validity = True

        for i in range(nb_dim_to_plot):
            for j in range(nb_dim_to_plot):
                if j > i :
                    continue

                elif j == i :

                    axes[i, j].set_xlim([-12, 12])

                else:

                    axes[i, j].set_xlim([-12, 12])
                    axes[i, j].set_ylim([-12, 12])

    except ValueError:
        print('Invalid range in LT corner plot')
        validity = False
        fig_corner_flow = corner.corner(im_codes, color='blue', label='Latent dim')

    return fig_corner_flow, validity


def corner_in_gauss(maf, x_base_samples, path, nb_dim_to_plot, type_dataset):

    pos_in_gauss = save_posingauss(maf, path, type_dataset)

    pig = []
    xbs = []

    for i in range(nb_dim_to_plot):
        pig.append(pos_in_gauss[:, i])
        xbs.append(x_base_samples[:, i])

    pig = np.array(pig).T
    xbs = np.array(xbs).T

    ndim = pig.shape[1]

    ranges = []
    for i in range(ndim):
        xmin = min(pig[:, i].min(), xbs[:, i].min())
        xmax = max(pig[:, i].max(), xbs[:, i].max())

        # small padding
        pad = 0.02 * (xmax - xmin)
        ranges.append((xmin - pad, xmax + pad))

    # print('Pig is',pig)
    try :
        fig_corner_gauss = corner.corner(pig, bins=20, range=ranges, color = 'blue', hist_kwargs={"density": True})
        corner.corner(xbs, bins=20, range=ranges, fig = fig_corner_gauss, color = 'red', hist_kwargs={"density": True})

        axes = np.array(fig_corner_gauss.axes)
        axes = axes.reshape(int(np.sqrt(len(axes))), int(np.sqrt(len(axes))))
        axes[0,1].scatter([],[], color = 'red', label = r'$z_{\mathcal{Q}} \sim \mathcal{Q}$')
        axes[0,1].scatter([],[], color = 'blue', label = r"$z'_{\mathcal{Q}} = g^{-1}(z)$")
        axes[0,1].legend(loc = 'lower left', fontsize = 20) ### Keep lower left ?

        for i in range(nb_dim_to_plot):
            for j in range(nb_dim_to_plot):
                if j > i:
                    continue

                elif j == i:
                    axes[i, j].set_xlim([-5, 5])

                else:

                    axes[i,j].set_xlim([-5, 5])
                    axes[i,j].set_ylim([-5, 5])

        validity = True

    except ValueError:
        print('Invalid range in Gauss corner plot')
        validity = False
        fig_corner_gauss = corner.corner(xbs, color='red', label='Gauss distribution')

    return fig_corner_gauss, validity

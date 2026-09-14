#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules

import numpy as np
import pickle

# Tensorflow

import tensorflow as tf

from architecture.vae_model import model_VAE
from utils.data_generator_utils import generator, encod_generator
from utils.data_utils import preprocessing_type

# Save encod as pickles


def save_encodings_to_pickle(latent_dim, model_path_weights):

    b_size = 1
    path = '../data/objet'

    # # Load images to obtain corresponding encodings given the right model weights
    preprocessing = preprocessing_type('Arcsin hyp')

    data_generator = generator(path, type_dataset='All')  # All the encodings from each image of the dataset : separation is done after

    data = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32))
    data = data.map(preprocessing).batch(b_size)

    # Load of weights

    model = model_VAE(latent_dim)[0]
    model.load_weights(model_path_weights)  # Automatically changed to correspond to the model used in flow and generation

    count = 0

    for i, features in enumerate(data):
        pred_images, encod_images = model(features)

        # -- Save encodings --
        encoding = encod_images.numpy()[:, :latent_dim]

        dico_encoding = {'encoding': encoding}

        count = count + 1

        with open(f'../data/encoding_{i}.pkl', 'wb') as f:
            pickle.dump(dico_encoding, f)

        # -- Save reconstruction scores --
        input_im = tf.cast(features, dtype=tf.float32)

        MSE = tf.square(tf.subtract(input_im, pred_images))
        score_rec = tf.reduce_sum(MSE, axis=[1, 2, 3])
        score_rec = tf.math.log(score_rec)

        score_rec = score_rec.numpy()

        # Save of scores into pickle files
        dico_score = {'score': score_rec}

        with open(f'../data/score_{i}.pkl', 'wb') as f:
            pickle.dump(dico_score, f)


# Save encod as lists
def save_encodings_to_list(latent_dim, model_path_weights, type_dataset):

    b_size = 256
    path = '../data/objet'

    preprocessing = preprocessing_type('Arcsin hyp')

    data_generator = generator(path, type_dataset)

    data = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32))
    data = data.map(preprocessing).batch(b_size)

    model = model_VAE(latent_dim)[0]
    model.load_weights(model_path_weights)

    count = 0

    for i, features in enumerate(data):
        pred_images, encod_images = model(features)
        encod = encod_images.numpy()[:, :latent_dim]

        if count == 0:
            encod_np = encod
        else:
            encod_np = np.concatenate((encod_np, encod), axis=0)

        count = count + 1

    dico_encod = {'encod_np': encod_np}

    with open('../pickle_files/encod_list.pkl', 'wb') as f:
        pickle.dump(dico_encod, f)

    return encod_np


def save_posingauss(maf, path, type_dataset):

    encoding_generator = encod_generator(path, type_dataset)

    dataset = tf.data.Dataset.from_generator(encoding_generator, output_types=(tf.float32))
    dataset = dataset.batch(64)

    count = 0

    for features in dataset:

        pos = maf.bijector.inverse(features)

        if count == 0:
            pos_in_gauss = pos
        else:
            pos_in_gauss = np.concatenate((pos_in_gauss, pos), axis=0)

        count = count + 1

    print('Shape posingauss', np.shape(pos_in_gauss))

    return pos_in_gauss

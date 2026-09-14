#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import numpy as np
import umap

# Tensorflow
# Local functions
from utils.data_utils import print_memory_usage, comp_range
from utils.data_generator_utils import labels_generator
from utils.save_encod import save_encodings_to_list


def umap_test(type_loss, latent_dim, num_run):
    print('UMAP test model : redshift, separations, duality')

    path = '../data/objet'
    nb_images = comp_range(path)[5]  # 5 is for TEST dataset -> change for other if necessary
    print('Nb images :', nb_images)

    # Load of data from encodings
    print_memory_usage('UMAP just starting')

    type_dataset = 'Test'
    model_path_weights = f'../checkpoints/{type_loss}_model_{num_run}_end'

    encod_np = save_encodings_to_list(latent_dim, model_path_weights, type_dataset)
    print('Shape encodings :', np.shape(encod_np))

    # Load labels from generator
    labels = labels_generator(path, type_dataset)
    print('Length labels :', len(labels[0]))  # [0] is redshift, [1] is separation

    # Construction of UMAP object
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(encod_np)
    print('Finished embedding', np.shape(embedding))

    return embedding, labels

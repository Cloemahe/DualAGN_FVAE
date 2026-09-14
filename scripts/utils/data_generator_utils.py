#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import numpy as np
import pickle
from utils.data_utils import comp_range


def generator(path, type_dataset):
    """
    Return a Tensorflow generator, taking images from pickle
    files stored in the folder indicated in ''path''
    parameter. ''type_dataset'' parameter defines the type
    of data required, either train, test or validation data.
    The same function can therefore be used to load data at
    every step of the model.

    :param path: Pathway to folder containing the pickle files used in the model.
    :type path: str
    :param type_dataset: Type of data that will be generated : either training, testing or validation
    :type type_dataset: str
    :return: Tensorflow data generator
    :rtype TF callable

    """
    if type_dataset == 'All':
        data_range = comp_range(path)[7]
    elif type_dataset == 'Train':
        data_range = comp_range(path)[0]
    elif type_dataset == 'Test':
        data_range = comp_range(path)[1]
    elif type_dataset == 'Validation':
        data_range = comp_range(path)[2]
    else:
        print('Type of dataset is not specified')

    def data_generator():
        for i in data_range:
            with open(path + f'_{i}.pkl', 'rb') as f:
                dico_load = pickle.load(f)

                # Our native dataset includes three photometric bands, but we train using only one
                image_r = dico_load['image r']

                # the labels are retrievable here, as they are in the same dico as their images :
                # just open a different list for labels, that will be aligned the same way as the images, then shuffle them

                img = np.zeros((image_r.shape[0], image_r.shape[1], 1), dtype=np.float32)
                img[:, :, 0] = image_r
                yield img

    return data_generator


def labels_generator(path, type_dataset):  # Load some labels

    if type_dataset == 'Train':
        data_range = comp_range(path)[0]
    elif type_dataset == 'Test':
        data_range = comp_range(path)[1]
    elif type_dataset == 'Validation':
        data_range = comp_range(path)[2]
    else:
        print('Type of dataset is not specified')

    redshift = []
    separation = []
    duality = []

    for i in data_range:
        with open(path + f'_{i}.pkl', 'rb') as f:
            dico_load = pickle.load(f)

            # the labels are retrievable here, as they are in the same dico as their images :
            # just open a different list for labels, that will be aligned the same way as the images, then shuffle them

            redshift.append(dico_load['redshift'])
            separation.append(dico_load['separation'])
            duality.append(dico_load['duality'])

    return redshift, separation, duality  # Should be lists


def encod_generator(path, type_dataset):  # Generator of only encodings

    if type_dataset == 'Train':
        data_range = comp_range(path)[0]
    elif type_dataset == 'Test':
        data_range = comp_range(path)[1]
    elif type_dataset == 'Validation':
        data_range = comp_range(path)[2]
    else:
        print('Type of dataset is not specified')

    def encoding_generator():
        for i in data_range:
            with open(path + f'_{i}.pkl', 'rb') as f:
                dico_load = pickle.load(f)

                encoding = dico_load['encoding'][0]

                yield encoding

    return encoding_generator

#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import os
import psutil
import glob

from utils.preprocessing_utils import preprocessing_sinh, preprocessing_zscale

# Define the memory usage of the code, can be used during training, 'info' is a string parameter, ex : 'before training'


def print_memory_usage(info):
    """
    Return the memory usage of the model at a given position or moment of the computation.

    :param info: Message displayed when memory usage is printed. Example : "Before the computation"
    :type info: str
    :return: Print the memory usage on the used processor in Megabytes
    """

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage in {info}: {mem_info.rss / (1024 ** 2)} MB")  # Memory usage in MB


# Compute the repartition of the whole dataset between training, test and validation + Separation of training loop, dependent on type of learning specified in 'type_dataset'

def comp_range(path):
    """
    Return the index ranges separating the whole dataset between train dataset, test dataset and validation dataset, as well as the number of images.

    :param path: Pathway to the folder where the whole dataset in pickle files is stored
    :type path: str
    :return: The indexes ranges of the train, test and validation datasets, along with the number of images listed in the folder.
    """

    all_files = glob.glob(path + "*.pkl")
    nb_images = len(all_files)  # Number of images to load

    # Computation of indexes separating train, test and validation datasets + transformation into ranges of indexes

    index_train_min = 0
    index_train_max = int(0.80 * nb_images)
    index_test_max = int(0.90 * nb_images)

    train_range = range(index_train_min, index_train_max)
    test_range = range(index_train_max, index_test_max)
    valid_range = range(index_test_max, nb_images)
    all_range = range(index_train_min, nb_images)

    nb_train = len(train_range)
    nb_test = len(test_range)
    nb_valid = len(valid_range)

    return train_range, test_range, valid_range, nb_images, nb_train, nb_test, nb_valid, all_range


# Preprocessing of the dataset : can be arcsin hyperbolic or Z-scale Ds9 function
def preprocessing_type(type_preprocessing):
    '''
    Return the preprocessing function of the Tensorflow dataset

    :param type_preprocessing: type of preprocessing (arcsin hyp or z-scale)
    :type type_preprocessing: str
    :return: Tensorflow preprocessing function of the dataset
    :rtype: TF.function of preprocessing
    '''
    if type_preprocessing == 'Arcsin hyp':
        preprocessing = preprocessing_sinh

    elif type_preprocessing == 'Z-scale':
        preprocessing = preprocessing_zscale

    else:
        print('Type of preprocessing non specified')

    return preprocessing

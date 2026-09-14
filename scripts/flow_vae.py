#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Classical modules
import matplotlib.pyplot as plt
import argparse
import yaml

# Local functions
from utils.data_utils import print_memory_usage, preprocessing_type, comp_range
from utils.data_generator_utils import generator

from architecture.vae_training_loop import train_loop
from architecture.flow_training_loop import flow, flow_cond

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tf.keras.backend.clear_session()  # For easy reset

plt.ioff()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "True"):
        return True
    if v.lower() in ("no", "false", "f", "0", "False"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


''' Workload parameters '''


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Workload hyperparameters
    parser.add_argument("--config", type=str)

    parser.add_argument("--path", type=str, default='../data/objet')
    parser.add_argument("--path_encod", type=str, default="../data/encoding")  # Path for encodings

    parser.add_argument("--type_preprocessing", type=str, default='Arcsin hyp')
    parser.add_argument("--type_loss", type=str, default='Fourier')

    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--test_on", type=bool, default=True)
    parser.add_argument("--loop_run", type=bool, default=False)
    parser.add_argument("--gen", type=bool, default=True)

    parser.add_argument("--new", type=bool, default=True)
    parser.add_argument("--num_run", type=int, default=221)
    parser.add_argument("--n_run", type=int, default=None)
    parser.add_argument("--old_run",  type=int, default=55)

    parser.add_argument("--nb_epoch_vae", type=int, default=400)
    parser.add_argument("--nb_epoch_flow", type=int, default=100)
    parser.add_argument("--nb_made", type=int, default=8)

    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--shuffle_size", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)

    parser.add_argument("--Beta", type=float, default=0.1)
    parser.add_argument("--init_lr_vae", type=float, default=1e-4)
    parser.add_argument("--init_lr_flow", type=float, default=1e-3)
    parser.add_argument("--end_lr", type=float, default=1e-6)
    parser.add_argument("--decay", type=float, default=2e-3)
    parser.add_argument("--power", type=float, default=0.5)

    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--zeta", type=int, default=40)
    parser.add_argument("--intensity", type=int, default=1)

    parser.add_argument("--nb_sample", type=int, default=10000)
    parser.add_argument("--nb_to_plot", type=int, default=50)
    parser.add_argument("--nb_dim_to_plot", type=int, default=5)
    parser.add_argument("--plot_every", type=int, default=10)

    parser.add_argument("--FWi", type=float, default=1e-6)
    parser.add_argument("--validation_every", type=int, default=1)

    # parse CLI to get the config file
    args = parser.parse_args()

    # load config file
    if hasattr(args, "config"):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # overwrite defaults with config
        for k, v in cfg.items():
            setattr(args, k, v)

    else:
        raise ValueError("You must specify a config file with the command line argument --config your_config.yml")

    return args


def main(args):

    print_memory_usage('Just starting')

    ''' VAE '''

    ''' Dataset '''

    nb_images = comp_range(args.path)[4]  # Number of train images.
    nb_images_valid = comp_range(args.path)[6]  # Number of validation images.
    Nb_step = int(args.nb_epoch_vae * nb_images / args.b_size)

    print(nb_images, nb_images_valid, Nb_step)

    preprocessing = preprocessing_type(args.type_preprocessing)

    # Train dataset

    img_generator = generator(args.path, type_dataset='Train')  # Depending on the type specified in 'type_dataset', this is our training dataset

    dataset = tf.data.Dataset.from_generator(img_generator, output_types=(tf.float32))  # Loading from generator of dataset : all images are not loaded at the same time
    dataset = dataset.shuffle(args.shuffle_size).map(preprocessing).batch(args.b_size)  # Processing of the dataset

    # Validation dataset -> for validation loop inside training loop

    img_valid_generator = generator(args.path, type_dataset='Validation')  # Validation dataset

    dataset_valid = tf.data.Dataset.from_generator(img_valid_generator, output_types=(tf.float32))
    dataset_valid = dataset_valid.shuffle(args.shuffle_size).map(preprocessing).batch(args.b_size)

    print_memory_usage('after dataset')

    ''' Training Loop '''

    print("Type of loss is", args.type_loss)

    train_loop(args.nb_epoch_vae, dataset, dataset_valid,
               args.latent_dim, args.Beta, args.init_lr_vae,
               Nb_step, args.decay, args.save,
               args.loop_run, args.type_loss, args.type_preprocessing,
               args.R, args.alpha, args.zeta, args.intensity,
               args.FWi, args.validation_every, args.test_on, args.num_run)

    ''' Flow '''

    flow(args.path_encod, args.n_run, args.tag,
         args.gen, args.test_on, args.new,
         args.type_loss, args.num_run, args.nb_epoch_flow,
         args.nb_made, args.latent_dim, args.b_size,
         args.init_lr_flow, args.end_lr, args.decay,
         args.power, args.nb_sample, args.nb_to_plot,
         args.nb_dim_to_plot, args.plot_every, args.validation_every)


if __name__ == "__main__":
    args = parse_args()
    main(args)

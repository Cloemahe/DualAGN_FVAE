#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of models

from architecture.flow_training_loop import flow
import argparse
import yaml

# Tensorflow
import tensorflow_probability as tfp

# Special functions
tfd = tfp.distributions
tfb = tfp.bijectors


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "True"):
        return True
    if v.lower() in ("no", "false", "f", "0", "False"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    # Workload parameters
    parser.add_argument("--path_encod", type=str, default="../data/encoding")  # Path for encodings

    parser.add_argument("--n_run", type=int, default=None)

    parser.add_argument("--gen", type=bool, default=True)
    parser.add_argument("--test_on", type=bool, default=True)  # Visualization of the images

    # Workload parameters from VAE run
    parser.add_argument("--new", type=bool, default=False)  # New VAE run
    parser.add_argument("--type_loss", type=str, default='Fourier')
    parser.add_argument("--num_run", type=int, default=220)  # LT = 32, beta = 0.1, n-epoch = 400

    # Hyperparameters

    # Training
    parser.add_argument("--nb_epoch", type=int, default=100)
    parser.add_argument("--nb_made", type=int, default=8)  # Number of MADE generated to create mu and sig (MADE = Variationnal Encoder with MASKED Dense layers) ## Size of MAF

    # Model hyperparameters
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--end_lr", type=float, default=1e-6)
    parser.add_argument("--decay", type=float, default=2e-3)
    parser.add_argument("--power", type=float, default=0.5)

    # Test hyperparameters
    parser.add_argument("--nb_sample", type=int, default=10000)
    parser.add_argument("--nb_to_plot", type=int, default=50)
    parser.add_argument("--nb_dim_to_plot", type=int, default=5)

    parser.add_argument("--plot_every", type=int, default=10)
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
    '''
    Model
    '''

    flow(args.path_encod, args.n_run, args.gen,
         args.test_on, args.new, args.type_loss,
         args.num_run, args.nb_epoch, args.nb_made,
         args.latent_dim, args.b_size, args.init_lr,
         args.end_lr, args.decay, args.power,
         args.nb_sample, args.nb_to_plot,
         args.nb_dim_to_plot, args.plot_every,
         args.validation_every)


if __name__ == "__main__":
    args = parse_args()
    main(args)

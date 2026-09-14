#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


# Importation of modules
from utils.diagnostics_flow_utils import test_flow
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
    # Workload parameter
    parser.add_argument("--path", type=str, default='../data/encoding')
    parser.add_argument("--old_run", type=int, default=53)
    parser.add_argument("--gen", type=bool, default=True)

    # Workload hyperparameter from VAE run
    parser.add_argument("--type_loss", type=str, default='Fourier')
    parser.add_argument("--num_run", type=int, default=220)

    # Hyperparameters
    parser.add_argument("--nb_made", type=int, default=8)
    parser.add_argument("--nb_sample", type=int, default=10000)
    parser.add_argument("--nb_to_plot", type=int, default=50)
    parser.add_argument("--nb_dim_to_plot", type=int, default=5)

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
    test_flow(args.nb_made, args.latent_dim, args.nb_sample, args.path, args.nb_to_plot, args.nb_dim_to_plot, args.gen, args.old_run, args.num_run, args.type_loss)


if __name__ == "__main__":
    args = parse_args()
    main(args)

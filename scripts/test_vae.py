#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


''' Modules '''
import matplotlib.pyplot as plt
import argparse
import yaml

from utils.diagnostics_vae_utils import test_model_vae

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
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser()

    ''' Hyperparameters '''

    parser.add_argument("--config", type=str)

    parser.add_argument("--num_run", type=int, default=220)

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--type_loss", type=str, default='Fourier')  # ou 'Composite'
    parser.add_argument("--type_preprocessing", type=str, default='Arcsin hyp')  # ou 'Z-scale'

    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--zeta", type=int, default=40)
    parser.add_argument("--intensity", type=int, default=1)

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
    test_model_vae(args.latent_dim, args.type_loss, args.type_preprocessing, args.num_run, args.R, args.alpha, args.zeta, args.intensity)


if __name__ == "__main__":
    args = parse_args()
    main(args)

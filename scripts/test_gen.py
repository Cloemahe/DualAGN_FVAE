#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 19-12-2025
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France

### /!\ Without conditionning for now ###

## Importation of modules
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time

import argparse
import yaml

import tensorflow_probability as tfp

from architecture.flow_architecture import model_flow
from architecture.vae_architecture import decoder

from utils.save_flow_class import FlowModel
from utils.score_gen_utils import score_gen_for_sample, score_normalised, score_power_spectrum
from utils.plot_utils import score_gen_multiple

tfd = tfp.distributions
tfb = tfp.bijectors

def str2bool(v) :
    if isinstance(v, bool) :
        return v
    if v.lower() in ("yes", "true", "t", "1", "True") :
        return True
    if v.lower() in ("no", "false", "f", "0", "False") :
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default = None)

    parser.add_argument("--nb_made", type = int, default = 8)
    parser.add_argument("--latent_dim", type = int, default = 32)
    parser.add_argument("--nb_sample", type = int, default = 10000)
    parser.add_argument("--nb_plot", type = int, default = 50)
    parser.add_argument("--run", type = int, default = 53)

    parser.add_argument("--save", type = bool, default = False)

    parser.add_argument("--type_loss", type = str, default = 'Fourier')
    parser.add_argument("--num_run", type = int, default = 220)

    parser.add_argument("--path", type = str, default = '/workspace/cmahe/pickle_files/')

    # parse CLI to get the config file
    args = parser.parse_args()
    
    # load config file
    if hasattr(args, "config"):
        with open(args.config) as f :
            cfg = yaml.safe_load(f)
            
        # overwrite defaults with config
        for k, v in cfg.items() :
            setattr(args, k, v)

    else :
        raise ValueError("You must specify a config file with the command line argument --config your_config.yml")

    return args

def main(args) :

    # Load of encodings and reconstruction scores 
    with open(args.path + f'encod_list.pkl', 'rb') as f :
        dico_load = pickle.load(f)

        encod_np = dico_load['encod_np'] 
        #print(encod_np)

    with open(args.path + f'scores_list.pkl', 'rb') as f :
        dico_load = pickle.load(f)

        scores_np = dico_load['scores_np']

    ### Model ###

    maf = model_flow(args.nb_made, args.latent_dim)
    flow_model = FlowModel(maf)
    flow_model.load_weights(f'./checkpoints/flow_model_{args.run}_end')

    rdm_encod = np.random.choice(np.arange(encod_np.shape[0]), 100, replace=False)
    vae_samples = encod_np[rdm_encod]
            
    encod_MSE, density_encod = score_gen_for_sample(maf, vae_samples, encod_np, scores_np, args.latent_dim, radius=0.5)    

    samples = maf.sample(args.nb_sample)
    plot_samples = maf.sample(args.nb_plot)

    ### Generations of new images ###

    Decoder = decoder(args.latent_dim, generation_mode = True)
    Decoder.load_weights(f'./checkpoints/{args.type_loss}_decoder_{args.num_run}_end')

    st = time.time()

    decoded_img = Decoder(samples)

    et = time.time()

    ds = (et - st)
    delta = args.nb_sample/ds
    print('Number of image', args.nb_sample, 'Total : Time = ', ds, 'secondes, nbs  = ',  delta, 'images per second.')

    score_img = score_normalised(maf, samples, encod_np, encod_MSE, density_encod, scores_np, args.latent_dim, radius=0.5)[0]

    if args.save == True :
        print('Saving the generated images with associated scores')
        for i in range(args.nb_sample) :

            generated_img = decoded_img.numpy()[i, :, :, 0]

            plt.figure()
            plt.imshow(generated_img)
            plt.title(rf'Generated image with score $S_g$ = {score_img[i]}')
            plt.savefig(f'/home/cmahe/dual_agn/generated_images/gen_im_{i}.png')
            plt.close()

    ### Visualization ###

    gen_img = Decoder(plot_samples)

    score_sample = score_normalised(maf, plot_samples, encod_np, encod_MSE, density_encod, scores_np, args.latent_dim, radius=0.5)[0]
    score_pp = score_power_spectrum(gen_img, low_limits=[1, 5], high_limits=[50, 80], score=True, log=False)


    fig = score_gen_multiple(gen_img, score_sample, score_pp, args.nb_plot)
    if args.save == True :
        print('Saving sample of generated images with scores')
        plt.savefig('generation_with_scores.png')

    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
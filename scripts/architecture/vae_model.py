#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


from tensorflow import keras as tfk
from architecture.vae_architecture import encoder, decoder

''' Model : Link between encoder, decoder, input and output to create the global model '''


def model_VAE(latent_dim):
    Encoder, _ = encoder(latent_dim)
    Decoder = decoder(latent_dim)

    VAE_input = tfk.Input(shape=(256, 256, 1))
    encoded_distrib_params = Encoder(VAE_input)
    decoded_img = Decoder(encoded_distrib_params)

    VAE_output = [decoded_img, encoded_distrib_params]

    model = tfk.Model(inputs=VAE_input, outputs=VAE_output, name='vae_dual_AGN')
    return model, Encoder, Decoder

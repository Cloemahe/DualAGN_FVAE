#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Mahé C., Bretonnière H., Slezak E.
# 03-06-2026
# Observatoire de la Côte d'Azur, Laboratoire Lagrange, UMR 7293, Nice, France


import tensorflow as tf


class FlowModel(tf.keras.Model):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution  # MUST be tracked attributes

    @tf.function
    def log_prob(self, x):
        # return log_prob as the model output
        return self.distribution.log_prob(x)

    @tf.function
    def sample(self, n):
        return self.distribution.sample(n)

    @tf.function
    def bijector_inverse(self, x):
        return self.distribution.bijector.inverse(x)

#!/usr/bin/env python3
""" This module defines the WGAN_CLIP class. """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """ This class defines a simple WGAN model. """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """
        This method initializes the WGAN_CLIP class.
        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            latent_generator: A function that generates latent vectors.
            real_examples: A tensor with real examples.
            batch_size: The batch size for training.
            disc_iter: The number of iterations to train the discriminator.
            learning_rate: The learning rate for the optimizer.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Define the generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # Define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        This method generates fake samples using the generator model.
        Args:
            size: The number of samples to generate.
            training: A boolean indicating if the model is training.
        Returns:
            A tensor with the generated samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        This method returns real samples from the dataset.
        Args:
            size: The number of samples to return.
        Returns:
            A tensor with the real samples.
        """
        if not size:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, _):
        """
        This method trains the WGAN model for one step.
        Args:
            _: A useless argument.
        Returns:
            A dictionary with the losses of the generator and discriminator.
        """
        for _ in range(self.disc_iter):
            # compute the loss for the discriminator in a tape watching the
            # discriminator's weights
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                discr_loss = self.discriminator.loss(
                    self.discriminator(real_samples),
                    self.discriminator(fake_samples))
            grads = tape.gradient(discr_loss,
                                  self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

            # clip the weights (of the discriminator) between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # compute the loss for the generator in a tape watching the generator's
        # weights
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            gen_loss = self.generator.loss(self.discriminator(fake_samples))
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

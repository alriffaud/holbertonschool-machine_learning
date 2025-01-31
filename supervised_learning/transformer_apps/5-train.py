#!/usr/bin/env python3
"""
This module defines the class CustomSchedule for a custom learning rate
schedule and the function train_transformer for training a Transformer
model.
"""
import tensorflow as tf
import numpy as np
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ This class creates a learning rate schedule. """
    def __init__(self, dm, warmup_steps=4000):
        """
        This method initializes the CustomSchedule instance.
        Args:
            dm: Model dimensionality.
            warmup_steps: Number of warmup steps.
        """
        super(CustomSchedule, self).__init__()

        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        This method calculates the learning rate for a given step.
        Args:
            step: Current training step.
        Returns:
            The learning rate for the current step.
        """
        step = tf.cast(step, tf.float32)  # Convert to float32
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    This method trains a Transformer model on a dataset.
    Args:
        N: Number of Encoder and Decoder blocks.
        dm: Model dimensionality.
        h: Number of attention heads.
        hidden: Number of hidden units in the FFNN.
        max_len: Maximum sequence length.
        batch_size: Batch size.
        epochs: Number of epochs to train for.
    Returns:
        The trained model.
    """
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    learning_rate = CustomSchedule(dm, warmup_steps=1000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len, max_len, drop_rate=0.1)

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for (batch, (input, target)) in enumerate(data.data_train):
            target_input = target[:, :-1]
            target_real = target[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                input, target_input)

            with tf.GradientTape() as tape:
                predictions = transformer(input, target_input, training=True,
                                          encoder_mask=enc_padding_mask,
                                          look_ahead_mask=combined_mask,
                                          decoder_mask=dec_padding_mask)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)(target_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy.update_state(target_real, predictions)

            if batch % 50 == 0:
                print('Epoch {}, Batch {}: Loss {:.4f}, Accuracy {:.4f}'
                      .format(epoch + 1, batch, train_loss.result(),
                              train_accuracy.result()))

        print('Epoch {}: Loss {:.4f}, Accuracy {:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))

    return transformer

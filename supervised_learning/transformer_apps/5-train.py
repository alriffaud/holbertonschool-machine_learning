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
    """ This class defines a custom learning rate schedule. """
    def __init__(self, dm, warmup_steps=4000):
        """ This method initializes the CustomSchedule class.
        Args:
            dm: Integer representing the model dimensionality.
            warmup_steps: Integer representing the number of warmup steps.
        """
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        This method calculates the learning rate for a given step.
        Args:
            step: Integer representing the step number.
        Returns:
            A tensor representing the learning rate for the given step.
        """
        step = tf.cast(step, tf.float32)
        l_rate = (self.dm ** -0.5) * tf.math.minimum(
            step ** -0.5, step * self.warmup_steps ** -1.5)
        return l_rate


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """ Trains a Transformer model for Portuguese-English translation. """
    dataset = Dataset(batch_size, max_len)
    
    # Get vocabulary sizes
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2
    
    # Initialize Transformer model
    transformer = Transformer(N, dm, h, hidden, input_vocab_size,
                              target_vocab_size, max_len, max_len)
    
    # Learning rate scheduling
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Loss function: SparseCategoricalCrossentropy (ignoring padding)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # Ignore padding
        loss_ = loss_object(real, pred)
        loss_ *= tf.cast(mask, dtype=loss_.dtype)
        return tf.reduce_mean(loss_)

    # Accuracy function
    def accuracy_function(real, pred):
        predictions = tf.argmax(pred, axis=2)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        acc = tf.equal(real, predictions)
        acc = tf.math.logical_and(mask, acc)
        return tf.reduce_sum(tf.cast(acc, dtype=tf.float32)) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

    # Training step function
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]  # Remove last token for teacher forcing
        tar_real = tar[:, 1:]  # Shift target for actual predictions
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, True, enc_padding_mask, look_ahead_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        acc = accuracy_function(tar_real, predictions)
        return loss, acc

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} starting...")
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for batch, (inp, tar) in enumerate(dataset.data_train):
            loss, acc = train_step(inp, tar)
            total_loss += loss
            total_acc += acc
            num_batches += 1
            
            if batch % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}: Loss {loss.numpy():.4f}, Accuracy {acc.numpy():.4f}")
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Epoch {epoch+1}: Loss {avg_loss.numpy():.4f}, Accuracy {avg_acc.numpy():.4f}")
    
    return transformer

#!/usr/bin/env python3
""" This module defines the gensim_to_keras function. """
from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """
    This function converts a gensim Word2Vec model to a Keras Embedding layer.
    Args:
        model (gensim.models.Word2Vec): A trained Word2Vec model.
    Returns:
        keras.layers.Embedding: A trainable Keras Embedding layer.
    """
    # Extract the weights (word vectors) from the gensim model
    weights = model.wv.vectors

    # Get the vocabulary size and embedding dimension
    vocab_size, embedding_dim = weights.shape

    # Create the Keras Embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True  # Allowing further training of embeddings in Keras
    )

    return embedding_layer

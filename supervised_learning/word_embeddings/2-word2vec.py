#!/usr/bin/env python3
""" This module defines the word2vec_model function. """
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    This function creates and trains a Word2Vec model.
    Args:
        sentences (list of str): A list of sentences to be trained on.
        vector_size (int): The dimensionality of the embedding layer.
        min_count (int): The minimum number of occurrences of a word for use
                        in training.
        window (int): The maximum distance between the current and predicted
                      word within a sentence.
        negative (int): The size of the negative sampling.
        cbow (bool): A boolean to determine the training type; True for CBOW,
                     False for Skip-gram.
        epochs (int): The number of iterations to train over the corpus.
        seed (int): The seed for the random number generator.
        workers (int): The number of worker threads to train the model.
    Returns:
        gensim.models.word2vec.Word2Vec: The trained Word2Vec model.
    """
    # Set training algorithm: CBOW or Skip-gram
    sg = 0 if cbow else 1

    # Initialize the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # Prepare the model's vocabulary
    model.build_vocab(sentences)
    # Train the model for the specified number of epochs
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model

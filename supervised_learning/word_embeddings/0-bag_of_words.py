#!/usr/bin/env python3
""" This module implements the Bag of Words model for text representation. """
import numpy as np
import string


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    Args:
        sentences (list of str): A list of sentences to analyze.
        vocab (list of str, optional): A list of vocabulary words to use.
        Defaults to None.
    Returns:
        tuple: A tuple containing:
            - embeddings (numpy.ndarray): The bag of words matrix of
            shape (s, f).
            - features (list of str): The vocabulary words used as features.
    """
    # Normalize and tokenize the sentences
    norm_sentences = [
        sentence.lower().translate(str.maketrans('', '', string.punctuation))
        for sentence in sentences
    ]
    tok_sentence = [sentence.split() for sentence in norm_sentences]

    # Build the vocabulary if not provided
    if vocab is None:
        vocab_set = set(word for sentence in tok_sentence for word in sentence)
        vocab = sorted(vocab_set)

    # Map each word in the vocabulary to an index
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    # Create the embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(tok_sentence):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, vocab

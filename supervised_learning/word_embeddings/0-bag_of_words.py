#!/usr/bin/env python3
""" This module implements the Bag of Words model for text representation. """
import numpy as np
import re


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
    # Preprocess sentences
    processed_sentences = []
    for sentence in sentences:
        # Remove non-alphabetic characters and convert to lowercase
        sentence = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        processed_sentences.append(sentence)

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(
            word for sentence in processed_sentences for word in sentence))

    # Create embeddings matrix
    s = len(processed_sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    # Convert vocabulary to required format
    vocab = np.array(vocab)

    return embeddings, vocab

#!/usr/bin/env python3
""" This module defines the tf_idf function. """
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.
    Args:
        sentences (list of str): A list of sentences to analyze.
        vocab (list of str, optional): A list of vocabulary words to use.
                                       Defaults to None.
    Returns:
        tuple: A tuple containing:
            - embeddings (numpy.ndarray): The TF-IDF matrix of shape (s, f).
            - features (list of str): The vocabulary words used as features.
    """
    # Preprocess sentences
    prep_sentences = [re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
                      for sentence in sentences]

    # Build the vocabulary if not provided
    if vocab is None:
        words = []
        for sentence in prep_sentences:
            words = re.findall(r'\w+', sentence)
            words.extend(words)
        vocab = sorted(set(words))

    # Compute TF-IDF for each sentence and term
    tf_idf_vect = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to produce the TF-IDF matrix
    tfidf_matrix = tf_idf_vect.fit_transform(sentences)

    # Extract the features (vocabulary used)
    features = tf_idf_vect.get_feature_names_out()

    # Transform tfidf_matrix to array
    embeddings = tfidf_matrix.toarray()

    return embeddings, features

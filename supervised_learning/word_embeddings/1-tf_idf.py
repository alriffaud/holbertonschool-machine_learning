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
    # Preprocess sentences: Remove non-alphabetic characters and
    # convert to lowercase
    processed_sentences = []
    for sentence in sentences:
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        # Join words back into a sentence
        processed_sentences.append(" ".join(words))

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(
            word for sentence in processed_sentences
            for word in sentence.split()))

    # Compute TF-IDF for each sentence and term
    tf_idf_vect = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to produce the TF-IDF matrix
    tfidf_matrix = tf_idf_vect.fit_transform(processed_sentences)

    # Extract the features (vocabulary used)
    features = tf_idf_vect.get_feature_names_out()

    # Transform tfidf_matrix to array
    embeddings = tfidf_matrix.toarray()

    return embeddings, features

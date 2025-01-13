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
    # Preprocess sentences: lowercase and remove special characters
    def preprocess(sentence):
        return re.sub(r'[^a-z\s]', '', sentence.lower())

    processed_sentences = [preprocess(sentence) for sentence in sentences]

    # Build the vocabulary if not provided
    if vocab is None:
        words = set()
        for sentence in processed_sentences:
            words.update(sentence.split())
        vocab = sorted(words)

    # Compute TF-IDF for each sentence and term
    tf_idf_vect = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to produce the TF-IDF matrix
    tfidf_matrix = tf_idf_vect.fit_transform(processed_sentences)

    # Extract the features (vocabulary used)
    features = tf_idf_vect.get_feature_names_out()

    # Transform tfidf_matrix to array
    embeddings = tfidf_matrix.toarray()

    return embeddings, features

#!/usr/bin/env python3
"""
Semantic Search using Universal Sentence Encoder.
This function performs semantic search on a corpus of reference documents by
finding the document most semantically similar to a given query sentence.
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    This function performs semantic search on a corpus of reference documents.
    Args:
        corpus_path (str): The path to the corpus of reference documents.
        sentence (str): The query sentence used for semantic search.
    Returns:
        str: The content of the document most similar to the query sentence.
    """
    # Load the Universal Sentence Encoder
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Load the reference documents
    files = os.listdir(corpus_path)
    docs = [open(corpus_path + '/' + f, 'rb').read().decode(
        'utf-8') for f in files]

    # Compute the embeddings
    corpus = [doc for doc in docs]
    corpus_embeddings = embed(corpus)
    query = [sentence]
    query_embedding = embed(query)[0]

    # Perform semantic search
    distances = np.inner(query_embedding, corpus_embeddings)
    closest = np.argmax(distances)

    return docs[closest]

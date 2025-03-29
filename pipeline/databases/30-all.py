#!/usr/bin/env python3
""" This module defines the list_all function """


def list_all(mongo_collection):
    """
    This function lists all documents in a collection.
    Args:
        mongo_collection: the pymongo collection object.
    Returns:
        A list of documents.
    """
    documents = mongo_collection.find()
    return list(documents)

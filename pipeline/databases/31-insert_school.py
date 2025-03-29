#!/usr/bin/env python3
""" This module defines the insert_school function """


def insert_school(mongo_collection, **kwargs):
    """
    This function inserts a new document in a collection based on kwargs
    Args:
        mongo_collection: the pymongo collection object.
        kwargs: dictionary with data to insert.
    Returns:
        The new _id.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id

#!/usr/bin/env python3
""" This module defines the update_topics function """


def update_topics(mongo_collection, name, topics):
    """
    This function changes all topics of a school document based on the name.
    Args:
        mongo_collection: the pymongo collection object.
        name: the school name to update.
        topics: the list of topics approached in the school.
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )

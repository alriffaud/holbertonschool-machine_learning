#!/usr/bin/env python3
""" This module defines the schools_by_topic function """


def schools_by_topic(mongo_collection, topic):
    """
    This function returns the list of schools having a specific topic.
    Args:
        mongo_collection: the pymongo collection object.
        topic: the topic searched.
    Returns:
        A list of schools.
    """
    schools = mongo_collection.find({"topics": {"$in": [topic]}})
    return list(schools)

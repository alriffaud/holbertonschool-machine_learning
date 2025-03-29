#!/usr/bin/env python3
""" This module provides some stats about Nginx logs stored in MongoDB """
from pymongo import MongoClient


METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def get_logs_stats(mongo_collection):
    """
    This function gets statistics about Nginx logs stored in MongoDB.
    Args:
        mongo_collection: the pymongo collection object.
        option: the method to search for.
    """
    total_logs = mongo_collection.count_documents({})
    print(f"{total_logs} logs")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = mongo_collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_check_count = mongo_collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print(f"{status_check_count} status check")


if __name__ == "__main__":
    nginx_collection = MongoClient('mongodb://127.0.0.1:27017').logs.nginx
    get_logs_stats(nginx_collection)

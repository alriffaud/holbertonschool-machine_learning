#!/usr/bin/env python3
""" This module defines Random_Forest class. """
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """ This is the Random_Forest class."""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ This is the __init__ method. """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None  # Prediction functions list
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """ This is the predict method. """
        # Aggregate predictions from all trees
        tree_predictions = np.array([tree_predict(explanatory)
                                     for tree_predict in self.numpy_preds])
        # Use mode to get the most common prediction
        return np.array([np.bincount(tree_preds).argmax()
                         for tree_preds in tree_predictions.T])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """ This method trains the decision tree with the given data. """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }
    - Mean accuracy on training data : { np.array(accuracies).mean()  }
    - Accuracy of the forest on td   : {
        self.accuracy(self.explanatory,self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        This method calculates the accuracy of the model. It Compares the
        predictions with the target and calculate the proportion of successes.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size

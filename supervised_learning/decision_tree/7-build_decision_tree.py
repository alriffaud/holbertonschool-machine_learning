#!/usr/bin/env python3
""" This module defines the Node, Leaf and Decision_Tree classes. """
import numpy as np


class Node:
    """ This is the Node class. """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """ This is the __init__ method. """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ This is the max_depth_below method. """
        # If the node is a leaf
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = 0
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = 0
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """ This is the max_depth_below method. """
        if self.is_leaf:
            return 1
        if only_leaves:
            if self.left_child:
                left_leaves = self.left_child.count_nodes_below(True)
            else:
                left_leaves = 0
            if self.right_child:
                right_leaves = self.right_child.count_nodes_below(True)
            else:
                right_leaves = 0
            return left_leaves + right_leaves
        else:
            if self.left_child:
                left_leaves = self.left_child.count_nodes_below()
            else:
                left_leaves = 0
            if self.right_child:
                right_leaves = self.right_child.count_nodes_below()
            else:
                right_leaves = 0
            return left_leaves + right_leaves + 1

    def left_child_add_prefix(self, text):
        """ This is the left_child_add_prefix function. """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """ This is the right_child_add_prefix function. """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """ This is the __str__ method. """
        result = f"{'root' if self.is_root else '-> node'} \
[feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__().strip())
        if self.right_child:
            result += self.right_child_add_prefix(
                self.right_child.__str__().strip())
        return result

    def get_leaves_below(self):
        """ This is the get_leaves_below method. """
        result = []
        if self.left_child:
            result += self.left_child.get_leaves_below()
        if self.right_child:
            result += self.right_child.get_leaves_below()
        return result

    def update_bounds_below(self):
        """ This is the update_bounds_below method. """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        This method computes the indicator function from the Node.lower and
        Node.upper dictionaries and stores it in an attribute Node.indicator.
        """
        def is_large_enough(x):
            """
            This function returns a 1D numpy array of size
            `n_individuals` so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its features > the lower bounds.
            """
            return np.array([np.greater(x[:, key], self.lower[key])
                             for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            """
            This function returns a 1D numpy array of size
            `n_individuals` so that the `i`-th element of the later is `True`
            if the `i`-th individual has all its features <= the lower bounds.
            """
            return np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """ This is the pred method. """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """ This is the Leaf class. """
    def __init__(self, value, depth=None):
        """ This is the __init__ method. """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ This is the max_depth_below method. """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ This is the count_nodes_below method. """
        return 1

    def __str__(self):
        """ This is the __str__ method. """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """ This is the get_leaves_below method. """
        return [self]

    def update_bounds_below(self):
        """ This is the update_bounds_below method. """
        pass

    def pred(self, x):
        """ This is the pred method. """
        return self.value


class Decision_Tree():
    """ This is the Decision_Tree class. """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ This is the __init__ method. """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ This is the depth method. """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ This is the count_nodes method. """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ This is the __str__ method. """
        return self.root.__str__()

    def get_leaves(self):
        """ This is the get_leaves method. """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ This is the update_bounds method. """
        self.root.update_bounds_below()

    def update_predict(self):
        """ This method computes the prediction function. """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum([leaf.indicator(A) * leaf.value
                                         for leaf in leaves], axis=0)

    def pred(self, x):
        """ This is the pred method. """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """ This method trains the decision tree with the given data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory,
                                    self.target)    }""")

    def np_extrema(self, arr):
        """ This method returns the minimum and maximum value of an array. """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ This method is responsible for generating a random division
        criterion. """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def fit_node(self, node):
        """ This method builds the decision tree recursively. """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold)

        left_filter = self.explanatory[:, node.feature][left_population]

        # Is left node a leaf ?
        is_left_leaf = ((left_filter.size < self.min_pop)
                        or (node.depth + 1 == self.max_depth)
                        or (np.unique(self.target[left_population]).size == 1))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        right_filter = self.explanatory[:, node.feature][right_population]
        is_right_leaf = ((right_filter.size < self.min_pop)
                         or (node.depth + 1 == self.max_depth)
                         or (np.unique(
                             self.target[right_population]).size == 1))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """ This method creates a leaf node. """
        # Find the most frequent class in the subpopulation.
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth+1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ This method creates a non-terminal (internal) node. """
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        This method calculates the accuracy of the model. It Compares the
        predictions with the target and calculate the proportion of successes.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size

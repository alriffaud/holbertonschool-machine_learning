#!/usr/bin/env python3
""" This module defines the Node, Leaf and Decision_Tree classes. """
import numpy as np


class Node:
    """ This class represents a node in the decision tree, which can be a
    decision node or a leaf. """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        This is the __init__ method.
        Args:
            feature (int): the index of the feature to split on.
            threshold (float): the threshold to split the feature.
            left_child (Node): the left child of the node.
            right_child (Node): the right child of the node.
            is_root (bool): a boolean indicating if the node is the root.
            depth (int): the depth of the node in the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ This method calculates the maximum depth from the current node
        to the deepest leaves. """
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
        """
        This method returns the number of nodes or leaves in the subtree
        from the current node.
        Args:
            only_leaves (bool): a boolean indicating if we should count only
                the leaves.
        """
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


class Leaf(Node):
    """ This class inherits from Node and represents a leaf in the decision
    tree. """
    def __init__(self, value, depth=None):
        """ This is the __init__ method. """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Overrides the method of the Node class. It simply returns the depth
        of the leaf. """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        This method overrides the method of the Node class. Returns 1
        because the leaf is itself a node.
        Args:
            only_leaves (bool): a boolean indicating if we should count only
                the leaves.
        """
        return 1


class Decision_Tree():
    """ This class is the main implementation of the decision tree. """
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
        """ This method returns the maximum depth of the tree. """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        This method returns the number of nodes in the tree. If only_leaves
        is True, count only leaves.
        Args:
            only_leaves (bool): a boolean indicating if we should count only
                the leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

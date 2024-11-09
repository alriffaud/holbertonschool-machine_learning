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

    def left_child_add_prefix(self, text):
        """
        This method adds visual prefixes to show the tree hierarchy when
        it is printed.
        Args:
            text (str): the text to add the prefix to.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        This method adds visual prefixes to show the tree hierarchy when
        it is printed.
        Args:
            text (str): the text to add the prefix to.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """ This method returns a string representation of the node and its
        subtree for easy viewing. """
        result = f"{'root' if self.is_root else '-> node'} \
[feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__().strip())
        if self.right_child:
            result += self.right_child_add_prefix(
                self.right_child.__str__().strip())
        return result


class Leaf(Node):
    """ This class inherits from Node and represents a leaf in the decision
    tree. """
    def __init__(self, value, depth=None):
        """
        This is the __init__ method.
        Args:
            value (int): the value of the leaf.
            depth (int): the depth of the leaf in the tree.
        """
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

    def __str__(self):
        """ This method overrides the __str__ method of the Node class,
        returning a string representation of the sheet. """
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """ This class is the main implementation of the decision tree. """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        This is the __init__ method.
        Args:
            max_depth (int): the maximum depth of the tree.
            min_pop (int): the minimum population of a node to split.
            seed (int): the seed for the random number generator.
            split_criterion (str): the criterion to split the tree.
            root (Node): the root node of the tree.
        """
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

    def __str__(self):
        """ This method returns a string representation of the entire tree. """
        return self.root.__str__()

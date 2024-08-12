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

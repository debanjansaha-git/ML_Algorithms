# -*- coding: utf-8 -*-
""" Decision Tree using Numpy
    Author: Debanjan Saha
    Affiliation: College of Engineering, Northeastern University, Boston
    Course: IE 7300 - Statistical Learning in Engineering
    Term:   Spring 2023
    Release Date: 05/12/2023
    License: MIT License
"""
# Import Libraries
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set_style('darkgrid')

class DecisionTree:
    '''
    Implements Decision Tree Classifier from scratch
    '''
    def __init__(self, max_depth=None):
        self.max_depth = max_depth    # Maximum depth of the tree
        self.root = None              # Root node of the tree

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # If only one class remains or max depth is reached, return leaf node
        if n_labels == 1 or depth == self.max_depth:
            return Node(value=self._most_common_label(y))

        # Find best split by iterating over all features and thresholds
        best_feature, best_threshold = self._best_split(X, y, n_samples, n_features)

        # Split data and create sub-nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)

        # Return internal node with best split
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Iterate over all features and thresholds to find best split
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # Split data into left and right based on feature and threshold
                left_idxs = X[:, feature] < threshold
                right_idxs = ~left_idxs
                left_labels, right_labels = y[left_idxs], y[right_idxs]

                # Skip if either side is empty
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                # Calculate information gain for this split
                gain = self._information_gain(y, left_labels, right_labels)

                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left, right):
        n = len(parent)
        nl, nr = len(left), len(right)
        pl, pr = nl/n, nr/n

        # Calculate entropy of parent and children
        entropy_parent = self._entropy(parent)
        entropy_children = pl * self._entropy(left) + pr * self._entropy(right)

        # Calculate information gain
        return entropy_parent - entropy_children


    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return unique[index]

    def _traverse_tree(self, x, node):
        # If node is a leaf, return its value
        if node.value is not None:
            return node.value

        # Traverse left or right depending on feature and threshold
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

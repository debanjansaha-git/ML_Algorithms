# -*- coding: utf-8 -*-
""" Random Forest using Numpy
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
import decisionTree

class RandomForestClf:
    '''
    Implements Random Forest Classifier from scratch
    '''
    
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees                    # Number of trees in the forest
        self.max_depth = max_depth                # Maximum depth of each tree
        self.min_samples_split = min_samples_split # Minimum number of samples to split a node
        self.max_features = max_features          # Maximum number of features to consider for each split
        self.trees = []                           # List of decision trees in the forest

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Determine number of features to consider for each split
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Build decision trees for the forest
        for i in range(self.n_trees):
            # Select random subset of samples and features for this tree
            sample_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_idxs = np.random.choice(n_features, size=self.max_features, replace=False)
            X_subset = X[sample_idxs][:, feature_idxs]
            y_subset = y[sample_idxs]

            # Build decision tree and add to forest
            tree = decisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y_subset)
            self.trees.append((tree, feature_idxs))

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            votes = []
            for tree, feature_idxs in self.trees:
                # Use only selected features for this tree
                X_subset = X[i][feature_idxs].reshape(1, -1)
                vote = tree.predict(X_subset)[0]
                votes.append(vote)
            # Tally votes and choose most common label
            label_counts = np.bincount(votes)
            most_common_label = np.argmax(label_counts)
            predictions.append(most_common_label)
        return np.array(predictions)

# -*- coding: utf-8 -*-
""" Logistic Regression using Numpy
    Author: Debanjan Saha
    Affiliation: College of Engineering, Northeastern University, Boston
    Course: IE 7300 - Statistical Learning in Engineering
    Term:   Spring 2023
    Release Date: 02/23/2023
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

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, alpha=0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        z = np.dot(X, self.w)
        h = self._sigmoid(z)
        J = -1 / len(y) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        return J

    def _gradient(self, X, y):
        z = np.dot(X, self.w)
        h = self._sigmoid(z)
        grad = 1 / len(y) * np.dot(X.T, (h - y)).reshape(self.w.shape)
        return grad.reshape(-1, 1)

    def _ridge_gradient(self, X, y):
        z = np.dot(X, self.w)
        h = self._sigmoid(z)
        grad = 1 / len(y) * (np.dot(X.T, (h - y)) + self.alpha * self.w).reshape(self.w.shape)
        return grad.reshape(-1, 1)

    def fit(self, X, y):
        # Normalize data
        self.train_mean = np.mean(X, axis=0)
        self.train_std = np.std(X, axis=0)
        X = (X - self.train_mean) / self.train_std

        self.w = np.zeros((X.shape[1], 1))
        for i in range(self.max_iter):
            if self.alpha == 0:
                grad = self._gradient(X, y)
            else:
                grad = self._ridge_gradient(X, y)
            self.w -= self.learning_rate * grad
            if np.linalg.norm(grad) < self.tol:
                break

    def predict(self, X):
        # Normalize data with train mean and std
        X = (X - self.train_mean) / self.train_std
        z = np.dot(X, self.w)
        return self._sigmoid(z)


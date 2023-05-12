# -*- coding: utf-8 -*-
""" KNN using Numpy
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


class KNN:

    def __init__(self, k):
        self.k = k

    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)):
            distance = distance + (row1[i] - row2[i])**2
        return np.sqrt(distance)

    def get_neighbors(self, test_row):
        distance = list()
        for train_row in self.X_train.values:
            dist = self.euclidean_distance(test_row, train_row)
            distance.append(dist)
        zipped = list(zip(self.X_train,self.y_train,distance))
        distances = pd.DataFrame(zipped, columns=['X_train','y_train','Distance'])
        distances = distances.sort_values(by=['Distance'])
        distances = distances.head(self.k)
        neighbors = distances['y_train']
        return neighbors

    def predict(self, test_row):
        neighbors = self.get_neighbors(test_row)
        count = [0,0,0,0,0,0]
        neighbors = list(neighbors)
        for i in neighbors:
            index = i - 1
            count[index] +=1
        return (np.argmax(count) + 1)
    
    def fit(self, X_trian, X_test, y_train):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        y_hat = list()
        for row in self.X_test:
            output = self.predict(row)
            y_hat.append(output)
        #accuracy = self.evaluate(y_hat)
        #print("Accuracy = ",accuracy)
        return(np.array(y_hat))
    
  
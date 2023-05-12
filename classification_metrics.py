# -*- coding: utf-8 -*-
""" Classification Metrics w/o sklearn
    Author: Debanjan Saha
    Affiliation: College of Engineering, Northeastern University, Boston
    Course: IE 7300 - Statistical Learning in Engineering
    Term:   Spring 2023
    Release Date: 05/12/2023
    License: MIT License
"""

import numpy as np

class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = np.unique(y_true)
        
    def confusion_matrix(self):
        n_classes = len(self.classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] = np.sum((self.y_true == self.classes[i]) & (self.y_pred == self.classes[j]))
        return cm
    
    def accuracy(self):
        cm = self.confusion_matrix()
        tp = np.sum(np.diag(cm))
        total = np.sum(cm)
        return tp / total
    
    def precision(self, average='macro'):
        cm = self.confusion_matrix()
        if average == 'macro':
            return np.mean([cm[i, i] / np.sum(cm[:, i]) for i in range(len(self.classes))])
        elif average == 'micro':
            tp = np.sum(np.diag(cm))
            total = np.sum(cm)
            return tp / total
        elif average == 'weighted':
            class_counts = np.sum(cm, axis=1)
            weights = class_counts / np.sum(class_counts)
            return np.sum([cm[i, i] / np.sum(cm[:, i]) * weights[i] for i in range(len(self.classes))])
    
    def recall(self, average='macro'):
        cm = self.confusion_matrix()
        if average == 'macro':
            return np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(len(self.classes))])
        elif average == 'micro':
            tp = np.sum(np.diag(cm))
            total = np.sum(cm)
            return tp / total
        elif average == 'weighted':
            class_counts = np.sum(cm, axis=1)
            weights = class_counts / np.sum(class_counts)
            return np.sum([cm[i, i] / np.sum(cm[i, :]) * weights[i] for i in range(len(self.classes))])
    
    def f1_score(self, average='macro'):
        p = self.precision(average=average)
        r = self.recall(average=average)
        return 2 * (p * r) / (p + r)

    def display_clf_metrics(metrics):
        print('Confusion Matrix')
        print(metrics.confusion_matrix())
        print()
        print('Accuracy: {:.4f}'.format(metrics.accuracy()))
        print('Precision: {:.4f}'.format(metrics.precision()))
        print('Recall: {:.4f}'.format(metrics.recall()))
        print('F1 Score: {:.4f}'.format(metrics.f1_score()))
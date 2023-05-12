# -*- coding: utf-8 -*-
""" Soft Margin SVM using Numpy
    Author: Debanjan Saha
    Affiliation: College of Engineering, Northeastern University, Boston
    Course: IE 7300 - Statistical Learning in Engineering
    Term:   Spring 2023
    Release Date: 05/12/2023
    License: MIT License
"""
# Import Libraries
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize,Bounds
from dataclasses import dataclass
@dataclass
class SoftMarginSVM:
  C : float
  rbf : bool

  def dualSVM(self, gramXy, alphas):
    return np.sum(alphas) - 0.5 * alphas.dot(alphas.dot(gramXy))

  def jacobian_dualSVM(self,gramXy,alphas):
    return np.ones(len(alphas)) - alphas.dot(gramXy)

  def fit(self,X,y):
    N,n_features = X.shape
    y_ = np.where(y<=0,-1,1)
    self.alphas = np.ones(N)
    bounds = Bounds(np.zeros(N),np.full(N,self.C))
    constraints = ({'type':'eq','fun':lambda a : -np.dot(a, y_),'jac':lambda a:-y_})

    Xy = X * y_[:,np.newaxis]
    if self.rbf:
      gramXY = np.exp(-cdist(Xy,Xy)/2)
    else:
      gramXy = Xy.dot(Xy.T)

    slsqp = minimize(fun = lambda a : -self.dualSVM(gramXy,a),
                     x0=self.alphas,
                     jac = lambda a : -self.jacobian_dualSVM(gramXy,a),
                     bounds=bounds,
                     constraints=constraints,
                     method='SLSQP')
    
    self.alphas = slsqp.x
    self.w = np.sum((self.alphas[:,np.newaxis] * Xy),axis=0)
    epsilon = 1e-4
    self.supportVectors = X[ self.alphas > epsilon]
    self.supportLabels = y[ self.alphas > epsilon]

    b = []
    for i in range(len(self.supportLabels)):
      b_i = self.supportLabels[i] - np.matmul(self.supportVectors[i].T, self.w)
      b.append(b_i)
        
    self.intercept = sum(b)/len(b)

  def sigmoid(self,z):
    sig = 1 / (1 + np.exp(-z))
    return sig

  def predict(self,X):
    sig = self.sigmoid(X.dot(self.w)+self.intercept)
    return sig

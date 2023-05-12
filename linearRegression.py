# -*- coding: utf-8 -*-
""" Linear Regression using Numpy
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

"""# Linear Regression:

## Steps Involved in Linear Regression

- Split Data: Split the input and output data into training and testing sets
- Normalization - Normalize the training and testing input matrices and add the bias column
- Invertibility Check - Check if the input matrix is full rank or low rank
- Cost Function - $J(\theta)$ is linear with respect to $\theta$. \\
We are tasked with learning the optimal parameters using either the Closed Form equation or using Gradient Descent if the closed form matrix is not invertible. If the matrix is full rank and not lower rank, and the data size is less than or equal to 10000 we use Closed Form or Normal Equation to calculate the coefficients, else we use gradient descent
    - Closed Form equation: $\theta^* = (X^TX)^{-1}X^TY$
    - Gradient Descent: $\theta^t = \theta^{t-1} - \alpha \frac{\partial J}{\partial \theta^{t-1}}$ \\
- Weights Update - If using gradient descent, iterate through the data to update weights by multiplying the learning rate with computed gradients until convergence is achieved or the maximum number of iterations is reached
- Plot Residuals - Compute the errors for each iteration and plot the change in errors curve for the training errors.
"""

class LinearRegression:
    '''
    Attributes:
    - X: The input matrix
    - y: The output matrix
    - alpha: Learning rate
    - max_iter: Maximum number of iterations
    - epsilon: Small value used for convergence check
    - method: ['normal', 'gradient', 'sgd'] 
      String indicating the method used to compute gradients. Default is 'gradient'.
    - cost_func: ['RMSE', 'SSE']
      String indicating the cost function used for optimization. Default is 'RMSE'.
    - w: Coefficients of the linear regression model
    - is_full_rank: Boolean flag to indicate whether X_train is full rank
    - is_low_rank: Boolean flag to indicate whether X_train is lower rank than full rank

    Methods:
    - splitData(): Splits the input and output data into training and testing sets
    - normalize(X, mean=None, std=None): Normalizes the input matrix and adds the bias column
    - rank(X): Checks if the input matrix is full rank or lower rank than full rank
    - normalEquation(X, y): Calculates the coefficients using the closed-form solution
    - cost_RMSE(X, y, reg_param): Computes the root mean squared errors (RMSE) for the linear regression model.
    - cost_SSE(X, y, reg_param): Computes the sum of squared errors (SSE) for the linear regression model.
    - costDerivative(X, y): Computes the derivative of the cost function with respect to the model coefficients.
    - gradientDescent(X, y, reg_param): Uses gradient descent by computing the gradients, 
        and updating weights by multiplying learning rate with computed gradient
    - stochasticGradientDescent(X, y, reg_param): Uses stochastic gradient descent by computing the gradient for each record, 
        and updating weights by multiplying learning rate with computed gradient
    - fit(X, y, reg_param=0): Fits the linear regression model to the training data
    - evaluate_test(X, y, reg_param): Compute SSE or RMSE for the test data
    - predict(X): Predicts the output values for the input matrix X
    - plot_errors(train_errors, test_errors, error_type, method): Plots the change in errors curve for the training errors
    - plot_residuals(y_true, y_pred): Plots residuals for the model 
    '''

    def __init__(self, X, y, alpha, max_iter, epsilon, method='gradient', cost_func='RMSE'):
        super().__init__()
        self.alpha = alpha              # learning rate
        self.max_iter = max_iter        # maximum iterations
        self.epsilon = epsilon          # tolerance
        self.method = method            # evaluation model
        self.cost_func = cost_func      # cost function
        self.w = None                   # parameters (weights) matrix
        self.is_full_rank = False       # flag for full rank
        self.is_low_rank = False        # flag for low rank
        self.mean = None                # store training mean
        self.std = None                 # store training standard deviation

    def splitData(self, X, y):
        '''
        Splits the data into training and test sets using the `train_test_split` function from scikit-learn.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=0,
                                                            shuffle=True)
        return X_train, X_test, y_train, y_test

    def normalize(self, X, mean=None, std=None):
        '''
        Normalizes the input matrix X by subtracting the mean and dividing by the standard deviation.
        Adds a column of ones to the input matrix X to account for the intercept term in the linear regression model.
        '''
        if mean is None or std is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        X_norm = (X - mean) / std
        # add intercept column to the beginning of normalized design matrix
        X_norm = np.column_stack([np.ones(X_norm.shape[0]), X_norm])
        return X_norm, mean, std

    def rank(self, X):
        '''
        Computes the rank of the input matrix X using the `np.linalg.matrix_rank` function.
        '''
        rank = np.linalg.matrix_rank(X)
        # v, s, u = np.linalg.svd(X)
        # rank = sum([1 if abs(x) > 0 else 0 for x in s])
        return rank

    def normalEquation(self, X, y):
        '''
        Computes the optimal coefficients for linear regression using the normal equation method.
        '''
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def cost_RMSE(self, X, y, reg_param):
        '''
        Computes the root mean squared errors (RMSE) for the linear regression model.
        '''
        m = len(y)
        rmse = np.sum((X.dot(self.w) - y) ** 2)
        reg_term = reg_param * np.sum(self.w ** 2)
        # add regularization to cost function
        J = np.sqrt((rmse + reg_term) / m)
        return J

    def cost_SSE(self, X, y, reg_param):
        '''
        Computes the sum of squared errors (SSE) for the linear regression model.
        '''
        mse = np.sum((X.dot(self.w) - y) ** 2)
        reg_term = reg_param * np.sum(self.w ** 2)
        # add regularization to cost functiokn
        J = mse + reg_term
        return J

    def costDerivative(self, X, y):
        '''
        Computes the derivative of the cost function with respect to the model coefficients.
        '''
        return (X.dot(self.w) - y).dot(X)

    def gradientDescent(self, X, y, reg_param, reg_type):
        '''
        Uses gradient descent by computing the gradients, and updating weights 
        by multiplying learning rate with computed gradient
        '''
        errors = []
        error_prev = np.infty
        for i in tqdm(range(self.max_iter)):
            # Compute the gradient
            gradient = self.costDerivative(X, y)
            # add regularization to gradient
            if reg_type == 'l1':
                gradient += reg_param * np.sign(self.w)
            elif reg_type == 'l2':
                gradient += reg_param * 2 * self.w
            # update the coefficients
            self.w -= self.alpha * gradient
            # compute the errors
            if self.cost_func == 'SSE':
                current_error = self.cost_SSE(X, y, reg_param)
            else:
                current_error = self.cost_RMSE(X, y, reg_param)
            ## Check for convergence
            if np.abs(current_error - error_prev) < self.epsilon:
                print(f"\n\nModel Stopped Learning...")
                break
            error_prev = current_error
            # Append current error to errors list
            errors.append(current_error)
        if i+1 == self.max_iter:
            print(f"\nMaximum iterations reached. Algorithm failed to converge")
        else:
            print(f"\nThe algorithm converged in:  {i+1} iterations") 
        return errors

    def stochasticGradientDescent(self, X, y, reg_param, reg_type):
        '''
        Uses stochastic gradient descent by computing the gradients for each sample in X and y, 
        and updating weights by multiplying learning rate with computed gradient
        '''
        errors = []
        error_prev = np.infty
        for i in tqdm(range(self.max_iter)):
            # shuffle the data
            permutation = np.random.permutation(X.shape[0])
            X, y = X[permutation], y[permutation]
            # Iterate over each training sample
            for j in range(X.shape[0]):
                xi = X[j, :]
                yi = y[j]
                # Compute the gradient for this sample
                gradient = (xi.dot(self.w) - yi) * xi
                if reg_type == 'l1':
                    gradient += reg_param * np.sign(self.w)
                elif reg_type == 'l2':
                    gradient += reg_param * 2 * self.w
                # Update the coefficients
                self.w -= self.alpha * gradient
            # Compute the errors
            if self.cost_func == 'SSE':
                current_error = self.cost_SSE(X, y, reg_param)
            else:
                current_error = self.cost_RMSE(X, y, reg_param)
            # Check for convergence
            if np.abs(current_error - error_prev) < self.epsilon:
                print("\n\nModel Stopped Learning...")
                break
            error_prev = current_error
            # Append current error to errors list
            errors.append(current_error)
        if i+1 == self.max_iter:
            print(f"\nMaximum iterations reached. Algorithm failed to converge")
        else:
            print(f"\nThe algorithm converged in:  {i+1} iterations") 
        return errors


    def fit(self, X, y, reg_param=0, reg_type='l2'):
        '''
        Fit linear regression model to the data and computes gradient
        Gradients can be computed using both gradient descent and normal form
        @param reg_param: Parameter can be provided to include regularization
        '''
        # Split Data into train & test sets
        X_train, X_test, y_train, y_test = self.splitData(X, y)
        X_train, self.mean, self.std = self.normalize(X_train)
        X_test, _, _ = self.normalize(X_test, self.mean, self.std)

        # Invertibility Check
        self.is_full_rank = self.rank(X_train) == min(X_train.shape)
        self.is_low_rank = X_train.shape[0] < X_train.shape[1]
        train_errors = []
        test_errors = []

        # Normal Equation
        if (self.method == 'normal') and (self.is_full_rank) and (not self.is_low_rank) and (X.shape[0] <= 10000):
            self.w = self.normalEquation(X_train, y_train)
            print("Using Normal Equation for Linear Regression")
            # train_error = X_train.dot(self.w) - y_train
            tr_error = self.cost_RMSE(X_train, y_train, reg_param)
            ts_error = self.cost_RMSE(X_test, y_test, reg_param)
            #print(tr_error.shape, ts_error.shape)
            # train_errors.append(train_error)
            # test_errors.append(test_error)
        else:
            # Gradient Descent
            if self.method == 'gradient':
                self.w = np.zeros(X_train.shape[1])
                print("Using Gradient Descent on Linear Regression")
                train_errors = self.gradientDescent(X_train, y_train, reg_param, reg_type)
                test_errors = self.evaluate_test(X_test, y_test, reg_param)
            
            # Stochastic Gradient Descent
            elif self.method == 'sgd':
                self.w = np.zeros(X_train.shape[1])
                print("Using Stochastic Gradient Descent on Linear Regression")
                train_errors = self.stochasticGradientDescent(X_train, y_train, reg_param, reg_type)
                test_errors = self.evaluate_test(X_test, y_test, reg_param)

            else:
                raise ValueError("Incorrect Method Selected")
        # print final weights
        print("Final Weights Matrix: \n", self.w)

        # Make predictions on train and test sets
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        #Calculate Errors for training and testing
        rmse_training = self.cost_RMSE(X_train, y_train, reg_param)
        rmse_testing = self.cost_RMSE(X_test, y_test, reg_param)
        sse_training = self.cost_SSE(X_train, y_train, reg_param)
        sse_testing = self.cost_SSE(X_train, y_train, reg_param)
        
        # Reporting RMSE and SSE
        print("For method: ", self.method)
        print("\nRMSE for training: ", rmse_training)
        print("\nRMSE for testing: ", rmse_testing)
        print("\nSSE for training: ", sse_training)
        print("\nSSE for testing: ", sse_testing)
        # Plot residuals and errors
        self.plot_residuals(y_train, y_train_pred)
        if self.method == 'gradient' or self.method == 'sgd':
            if self.cost_func == 'SSE':
                self.plot_errors(train_errors, test_errors, 'SSE', self.method)
            else:
                self.plot_errors(train_errors, test_errors, 'RMSE', self.method)
        # print("Processing Successful!! Exiting...")
        
    def evaluate_test(self, X, y, reg_param):
        '''
        Compute SSE or RMSE for the given data and weights
        '''
        test_errors = []
        for i in range(X.shape[0]):
            if self.cost_func == 'SSE':
                error = self.cost_SSE(X, y, reg_param)
            else:
                error = self.cost_RMSE(X, y, reg_param)
            test_errors.append(error)
        return test_errors
    
    def predict(self, X):
        '''
        Compute the output of input matrix and weights
        '''
        return X.dot(self.w)

    def plot_errors(self, train_errors, test_errors, error_type, method):
        '''
        Plots the training and testing errors of the model
        '''
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        ax.plot(train_errors, label='Training Error')
        ax.plot(test_errors, label='Testing Error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(error_type)
        ax.set_title(f'{error_type} vs Iteration for {method}')
        ax.grid()
        ax.legend()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        '''
        Plots the residuals of the model
        '''
        residuals = y_true - y_pred
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        ax.scatter(y_pred, residuals, c=y_pred, cmap='Blues', alpha=1, label='Predicted')
        ax.scatter(y_pred, residuals, c=residuals, cmap='Greens', alpha=0.5, label='Residuals')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        plt.legend()
        plt.show()

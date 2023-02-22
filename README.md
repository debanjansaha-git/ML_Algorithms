# Linear Regression using Numpy

This repository contains a Python implementation of linear regression using the NumPy library. Linear regression is a machine learning algorithm used to predict a continuous output variable based on one or more input variables. This implementation offers three methods for computing the gradients: normal equation, gradient descent, and stochastic gradient descent.

## Steps Involved in Linear Regression

  * Split Data: Split the input and output data into training and testing sets.
  * Normalization: Normalize the training and testing input matrices and add the bias column.
  * Invertibility Check: Check if the input matrix is full rank or low rank.
  * Cost Function: J(θ) is linear with respect to θ. We are tasked with learning the optimal parameters using either the Closed Form equation or using Gradient Descent if the closed-form matrix is not invertible. If the matrix is full rank and not lower rank, and the data size is less than or equal to 10000, we use Closed Form or Normal Equation to calculate the coefficients. Else, we cab use gradient descent or stochastic gradient descent.
      * Closed Form equation: $\theta^* = (X^TX)^{-1}X^TY$
      * Gradient Descent: $\theta^t = \theta^{t-1} - \alpha \frac{\partial J}{\partial \theta^{t-1}}$
  * Weights Update: If using gradient descent, iterate through the data to update weights by multiplying the learning rate with computed gradients until convergence is achieved or the maximum number of iterations is reached.
  * Plot Residuals: Compute the errors for each iteration and plot the change in errors curve for the training errors.

## Dependencies

    numpy
    pandas
    matplotlib
    tqdm
    scikit-learn

## Usage

To use this code, simply clone this repository and import the LinearRegression class into your project. An example of how to use this class is provided in the main function.

    from linearRegression import LinearRegression
    import pandas as pd

    if __name__ == "__main__":
        # load the dataset
        df = pd.read_csv("data/Advertising.csv")

        # create the input and output matrices
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)

        # create an instance of the LinearRegression class
        lr = LinearRegression(X, y, alpha=0.01, max_iter=1000, epsilon=0.0001, method='gradient', cost_func='RMSE')

        # fit the model to the training data
        lr.fit(X_train, y_train)

        # evaluate the model on the test data
        test_errors = lr.evaluate_test(X_test, y_test, reg_param=0)

        # plot the errors curve
        lr.plot_errors(train_errors, test_errors, 'RMSE', 'gradient')

        # plot the residuals
        y_pred = lr.predict(X_test)
        lr.plot_residuals(y_test, y_pred)
        
## Credits

This code was written by Debanjan Saha and Ritika Rao as part of the IE 7300 Statistical Learning in Engineering course at Northeastern University. If you use this code or any part of it, please consider citing this repository.

    Saha, D. and Rao, R. (2023, February 22). Linear Regression Using Numpy. GitHub. from https://github.com/debanjansaha-git/ML_Algorithms 

## License

This code is available under the MIT License.

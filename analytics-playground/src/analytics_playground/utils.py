


################################################################################
# XGBoost CPU Utilization (Number of estimators) vs. Number of Estimators and Threads
# https://xgboosting.com/xgboost-cpu-usage-below-100-during-training/
################################################################################

import psutil
import pandas as pd
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Generate a synthetic dataset for binary classification
X, y = make_classification(n_samples=1000000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Define the range of threads and estimators to test
threads_range = range(1, 5)
estimators_range = [10, 50, 100, 200, 300, 400, 500]

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(columns=['threads', 'estimators', 'cpu_utilization'])

# Iterate over the number of threads and estimators
for threads in threads_range:
   for estimators in estimators_range:
       # Initialize an XGBClassifier with the specified parameters
       model = XGBClassifier(n_jobs=threads, n_estimators=estimators, random_state=42)

       # Measure CPU utilization before training
       _ = psutil.cpu_percent()

       # Train the model
       model.fit(X, y)

       # Measure CPU utilization since last call
       cpu_percent_during = psutil.cpu_percent()

       result = pd.DataFrame([{
                           'threads': threads,
                           'estimators': estimators,
                           'cpu_utilization': cpu_percent_during
                       }])
       # Report progress
       print(result)

       # Append the results to the DataFrame
       results_df = pd.concat([results_df, result], ignore_index=True)

# Pivot the DataFrame to create a matrix suitable for plotting
plot_df_cpu = results_df.pivot(index='estimators', columns='threads', values='cpu_utilization')

# Create a line plot
plt.figure(figsize=(10, 6))
for threads in threads_range:
   plt.plot(plot_df_cpu.index, plot_df_cpu[threads], marker='o', label=f'{threads} threads')

plt.xlabel('Number of Estimators')
plt.ylabel('CPU Utilization (%)')
plt.title('XGBoost CPU Utilization vs. Number of Estimators and Threads')
plt.legend(title='Threads')
plt.grid(True)
plt.xticks(estimators_range)
plt.show()

################################################################################
# Tune XGBoost "alpha" Parameter
# Applying matplotlib.pyplot.semilogx to make a plot with log scaling on the x-axis
# Applying matplotlib.pyplot.fill_between to plot confidence interval
# https://xgboosting.com/tune-xgboost-alpha-parameter/
################################################################################
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score

# Create a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42)

# Configure cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid
param_grid = {
    'alpha': [0, 0.01, 0.1, 1, 10, 100]
}

# Set up XGBoost regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Get results
print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV R^2 score: {grid_search.best_score_:.4f}")

# Plot alpha vs. R^2 score
import matplotlib.pyplot as plt
results = grid_search.cv_results_

plt.figure(figsize=(10, 6))
plt.semilogx(param_grid['alpha'], results['mean_test_score'], marker='o', linestyle='-', color='b')
plt.fill_between(param_grid['alpha'], results['mean_test_score'] - results['std_test_score'],
                 results['mean_test_score'] + results['std_test_score'], alpha=0.1, color='b')
plt.title('Alpha vs. R^2 Score')
plt.xlabel('Alpha (log scale)')
plt.ylabel('CV Average R^2 Score')
plt.grid(True)
plt.show()




# from abc import ABC # Abstract Base Classes
from typing import Any

import numpy as np
from numpy.typing import NDArray
# from pydantic import BaseModel

class LinearRegression():
   
   def __init__(self) -> None:
      """Initialize the LinearRegression"""
      self.intercept = None
      self.coefficients = None
   
   def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
      """
      Fit a linear regression model.

      Parameters
      ----------
      X : np.ndarray
         an array containing feature values for training.
      y : np.ndarray
         an array containing the response values.
      
      Returns
      -------
      Any
         a trained model.
      """
      
      ones = np.ones((len(X), 1)) # y-intercept (or bias)
      X = np.hstack((ones, X))
      # (X^T*X)^(-1)*X^T*y
      XT = X.T # Transpose of X
      XTX = XT.dot(X)   # X^T*X
      # NumPy linear algebra functions to inverse a matrix.
      # also scipy.linalg.inv(a) in SciPy library 
      XTX_inv = np.linalg.inv(XTX)  # (X^T*X)^(-1), inverse
      XTy = XT.dot(y)   # X^T*y
      self.coefficients = XTX_inv.dot(XTy)   # Calculate the coefficients
 
      
   def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
      """
      Make predictions from a model.

      Parameters
      ----------
      X : np.ndarray
         an array containing feature values for prediction.
      
      Returns
      -------
      np.array
         a numpy array of predictions.
      """
      ones = np.ones((len(X), 1)) # Match the train data structure
      X = np.hstack((ones, X))
      # Apply the coefficients to make predictions. numpy.matmul(A, B) or A @ B or A.dot(B). the @ operator is preferable.
      # return X.dot(self.coefficients)
      return X @ self.coefficients
   
   # Calculate R-squared (the coefficient of determination)
   def Rsquared(self, X:NDArray[np.float64], y:NDArray[np.float64]) -> float:
      """
      Make predictions from a model.

      Parameters
      ----------
      X : np.ndarray
         an array containing feature values for prediction.
      y : np.ndarray
         an array containing the response values.
      
      Returns
      -------
      float
         the calculated R-squared value.
      """
      ypred = self.predict(X)
      ss_total = np.sum((y - np.mean(y))**2) # Total sum of squares
      ss_residual = np.sum((y - ypred)**2)   # Residual sum of squares
      return 1 - ss_residual / ss_total   # R-squared
   

from sklearn import datasets

# Generate some toy data
X, y = datasets.make_regression(
        n_samples=500,
        n_features=1, 
        # the code is generalized for any number (n > 1) of features
        noise=15,
        random_state=4
        )

# Initialize and fit a model
model = LinearRegression()
model.fit(X, y)
#print(model.coefficients)

#print(model.intercept)
dir(model)

# Make prediction
y_pred = model.predict(X)

# R-squared value
print(model.Rsquared(X, y))

# one-D plot to illustrate how well the model fits the data
import matplotlib.pyplot as plt
# select one feature only for one-D plot
feature_index = 0

intercept = model.coefficients[0]

fig, ax = plt.subplots(figsize=(8,6))

# generate x values and y = a*x + b for line plot
x_lp = np.linspace(X[:,feature_index].min(), X[:,feature_index].max(), 100)
y_lp = model.coefficients[feature_index+1]*x_lp + intercept

ax.scatter(X[:,feature_index], y, color='blue')
ax.scatter(X[:,feature_index], y_pred, color='red')
ax.plot(x_lp, y_lp, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Linear Regression')
plt.show() # plt.savefig()


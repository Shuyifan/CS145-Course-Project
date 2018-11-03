import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

"""
The function to perform a linear regression based on a training data. Then calculate the training error and testing 
    error.

:param train_X: the training dataset
:param train_y: the ground truth label of training dataset
:param validate_X: the testing dataset
:param validate_y: the ground truth label of testing dataset

:type train_X: pandas.DataFrame
:type train_y: pandas.Series
:type validate_X: pandas.DataFrame
:type validate_y: pandas.Series

:returns train_error: the trainig error of the model
:returns test_error: the trainig error of the model using the validation dataset

:rtype train_error: float
:rtype test_error: float
"""
def runLinearRegression(train_X, train_y, validate_X, validate_y):
    linreg = LinearRegression().fit(train_X, train_y)
    train_error = mse(train_y, linreg.predict(train_X))**0.5
    test_error = mse(validate_y, linreg.predict(validate_X))**0.5
    print (train_error)
    print (test_error)
    return train_error, test_error
    #print (linreg.score(validate_X, validate_y))

"""
The function to perform a logistic regression based on a training data. Then calculate the training error and testing 
    error.

:param train_X: the training dataset
:param train_y: the ground truth label of training dataset
:param validate_X: the testing dataset
:param validate_y: the ground truth label of testing dataset

:type train_X: pandas.DataFrame
:type train_y: pandas.Series
:type validate_X: pandas.DataFrame
:type validate_y: pandas.Series

:returns train_error: the trainig error of the model
:returns test_error: the trainig error of the model using the validation dataset

:rtype train_error: float
:rtype test_error: float
"""
def runLogisticRegression(train_X, train_y, validate_X, validate_y):
    logreg = LogisticRegression().fit(train_X, train_y)
    train_error = logreg.score(train_X, train_y)
    test_error = logreg.score(validate_X, validate_y)
    print (train_error)
    print (test_error)

"""
The function to perform a random forest based on a training data. Then calculate the training error and testing 
    error.

:param train_X: the training dataset
:param train_y: the ground truth label of training dataset
:param validate_X: the testing dataset
:param validate_y: the ground truth label of testing dataset

:type train_X: pandas.DataFrame
:type train_y: pandas.Series
:type validate_X: pandas.DataFrame
:type validate_y: pandas.Series

:returns train_error: the trainig error of the model
:returns test_error: the trainig error of the model using the validation dataset

:rtype train_error: float
:rtype test_error: float
"""
#RandomForest Regression
def runRandomForest(train_X, train_y, validate_X, validate_y):
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(train_X, train_y)
    train_error = mse(train_y, clf.predict(train_X))**0.5
    test_error = mse(validate_y, clf.predict(validate_X))**0.5
    print (train_error)
    print (test_error)
    return train_error, test_error
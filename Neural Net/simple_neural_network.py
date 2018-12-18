import sklearn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

path = os.path.dirname(os.getcwd())
df = pd.read_csv(path + '/Data/After Processing/join_train_queries.csv')
train_y = df['stars_review']
train_X = df.drop('stars_review', axis = 1)
df = pd.read_csv(path + '/Data/After Processing/join_validate_queries.csv')
validate_y = df['stars_review']
validate_X = df.drop('stars_review', axis = 1)

#drop some columns
columns = train_X.columns.tolist()
print (len(columns))
good_bye_list = []
for col in columns:
    if 'categories' in col:
        good_bye_list.append(col)
train_X.drop(good_bye_list, axis = 1, inplace=True)
validate_X.drop(good_bye_list, axis = 1, inplace=True)
print (len(train_X.columns.tolist()))

"""
The function to perform a neural net model (using a multi-layer Perceptron regressor) based on a training 
    data. Then calculate the training error and testing error.

:param train_X: the training dataset
:param train_y: the ground truth label of training dataset
:param validate_X: the testing dataset
:param validate_y: the ground truth label of testing dataset
:param act_func: Activation function for the hidden layer
:param unit_n: The ith element represents the number of neurons in the ith hidden layer

:type train_X: pandas.DataFrame
:type train_y: pandas.Series
:type validate_X: pandas.DataFrame
:type validate_y: pandas.Series
:type act_func: str
:type unit_n: int

:returns train_error: the trainig error of the neural net model
:returns test_error: the trainig error of the neural net model using the validation dataset

:rtype train_error: float
:rtype test_error: float
"""
def rmse(train_X, train_y, validate_X, validate_y, act_func, unit_n):
    reg = MLPRegressor(hidden_layer_sizes=unit_n, activation=act_func, random_state=42)
    reg.fit(train_X, train_y)
    train_y_pred = reg.predict(train_X)
    validate_y_pred = reg.predict(validate_X)
    train_error = np.mean((train_y_pred - train_y)**2)
    test_error = np.mean((validate_y_pred - validate_y)**2)
    return train_error, test_error

"""
The function to select the best hyperparameter from the previous test result. And plot the RMSE versus the 
    ith element represents the number of neurons in the ith hidden layer.

:param unit_nums: an iterable, which contains the number of units_num we have tested
:param RMSE: the RMSE corresponding to the specific units_num
:param state: specify whether it is a training RMSE or testsing RMSE
:param act: specify which activation function is used to train the model and calculate the current RMSE

:type unit_nums: range
:type RMSE: list
:type state: str
:type act: str
"""
def select_best_unit(unit_nums, RMSE, state='Testing', act='relu'):
    min_idx = np.argmin(RMSE)
    print('--------------------------------------------')
    print('For ' + state + ', the best unit number is %i' % unit_nums[min_idx])
    print('The best RMSE is ' + str(RMSE[min_idx]))
    plt.plot(unit_nums, RMSE, label=state)
    plt.title(state + ' RMSE (' + act + ')')
    plt.xlabel('Unit Number')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

unit_nums = range(50, 201, 10)
act_func = ['relu', 'logistic', 'tanh']
for act in act_func:
    train_RMSE = []
    val_RMSE = []
    print('*** For '+act+' ***')
    for n in unit_nums:
        un = tuple([n])
        train_err, val_err = rmse(train_X, train_y, validate_X, validate_y, act, un)
        train_RMSE.append(train_err)
        val_RMSE.append(val_err)
    select_best_unit(unit_nums, train_RMSE, 'Training', act)
    select_best_unit(unit_nums, val_RMSE, 'Validating', act)

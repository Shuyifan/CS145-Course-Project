import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('join_train_queries.csv')
train_y = df['stars_review']
train_X = df.drop('stars_review', axis = 1)
df = pd.read_csv('join_validate_queries.csv')
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

def rmse(train_X, train_y, validate_X, validate_y, act_func, unit_n):
    reg = MLPRegressor(hidden_layer_sizes=unit_n, activation=act_func, random_state=42)
    reg.fit(train_X, train_y)
    train_y_pred = reg.predict(train_X)
    validate_y_pred = reg.predict(validate_X)
    np.mean((train_y_pred - train_y)**2)
    return np.mean((train_y_pred - train_y)**2), np.mean((validate_y_pred - validate_y)**2)

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

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import datetime
from sklearn.metrics import mean_squared_error as mse
import csv

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
test_X.drop(good_bye_list, axis = 1, inplace=True)
print (len(train_X.columns.tolist()))
      
#Linear Regression
starttime = datetime.datetime.now()
linreg = LinearRegression().fit(train_X, train_y)
print ('training rmse is '+str(mse(train_y, linreg.predict(train_X))**0.5))
print ('validating rmse is '+str(mse(validate_y, linreg.predict(validate_X))**0.5))
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')

starttime = datetime.datetime.now()
logreg = LogisticRegression().fit(train_X, train_y)
print (logreg.score(train_X, train_y))
print (logreg.score(validate_X, validate_y))
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')

'''
results
training rmse is 1.0034983409007958
validating rmse is 1.0526230022791017
takes 2 seconds
0.4230124074764364
0.4678594963755816
takes 736 seconds
'''
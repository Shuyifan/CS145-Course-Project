import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

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

#standardization, commentted because the number of samples changes after standardization, which makes following prediction unusable.
# print (train_X.shape)
# scaler = StandardScaler()
# train_X = scaler.fit_transform(train_X)
# train_X = scaler.fit_transform(validate_X)
# print (train_X.shape)
# print (validate_X.shape)

#Linear Regression
starttime = datetime.datetime.now()
linreg = LinearRegression().fit(train_X, train_y)
print (mse(train_y, linreg.predict(train_X))**0.5)
print (mse(validate_y, linreg.predict(validate_X))**0.5)
#print (linreg.score(validate_X, validate_y))
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')

# starttime = datetime.datetime.now()
# logreg = LogisticRegression().fit(train_X, train_y)
# print (logreg.score(train_X, train_y))
# print (logreg.score(validate_X, validate_y))
# endtime = datetime.datetime.now()
# print ('takes '+str((endtime - starttime).seconds)+' seconds')

#RandomForest Regression
starttime = datetime.datetime.now()
clf = RandomForestRegressor(n_estimators=100)
clf.fit(train_X, train_y)
print (mse(train_y, clf.predict(train_X))**0.5)
print (mse(validate_y, clf.predict(validate_X))**0.5)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')

'''
results
1.0034983409007958
1.0526230022791017
takes 2 seconds
0.3798323897591327
1.0746892677354853
takes 611 seconds
'''
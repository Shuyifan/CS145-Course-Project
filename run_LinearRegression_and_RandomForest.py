import pandas as pd
import numpy as np
import datetime
import modelsFunc as models

df = pd.read_csv(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_train_queries.csv')
train_y = df['stars_review']
train_X = df.drop('stars_review', axis = 1)
print(type(train_y))
print(type(train_X))
print("***************")
df = pd.read_csv(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_validate_queries.csv')
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

#Run Linear Regression
starttime = datetime.datetime.now()
models.runLinearRegression(train_X, train_y, validate_X, validate_y)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')

#Run Random Forest
starttime = datetime.datetime.now()
models.runRandomForest(train_X, train_y, validate_X, validate_y)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')
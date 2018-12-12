import pandas as pd
import numpy as np
import datetime
import modelsFunc as models

df = pd.read_csv(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/train_queries_compacted_more.csv')
train_y = df['stars_review']
train_X = df.drop('stars_review', axis = 1)
df = pd.read_csv(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/validation_queries_compacted_more.csv')
validate_y = df['stars_review']
validate_X = df.drop('stars_review', axis = 1)
df = pd.read_csv(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/test_queries_compacted_more.csv')
test_X = df

#drop some columns
"""columns = train_X.columns.tolist()
print (len(columns))
good_bye_list = []
for col in columns:
    if 'categories' in col:
        good_bye_list.append(col)
train_X.drop(good_bye_list, axis = 1, inplace=True)
validate_X.drop(good_bye_list, axis = 1, inplace=True)
test_X.drop(good_bye_list, axis = 1, inplace=True)
print (len(train_X.columns.tolist()))"""


#Run Linear Regression
starttime = datetime.datetime.now()
models.runLinearRegression(train_X, train_y, validate_X, validate_y)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')


#Run Logistic Regression
starttime = datetime.datetime.now()
models.runLogisticRegression(train_X, train_y, validate_X, validate_y)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')


#Run Random Forest Regression
starttime = datetime.datetime.now()
train_error, test_error, clf = models.runRandomForest(train_X, train_y, validate_X, validate_y, max_depth = 6)
result = clf.predict(test_X)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')
models.writeOutputSubmission(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/submission14.csv', result)

#Run Random Forest Classifiers
starttime = datetime.datetime.now()
train_error, test_error, clf = models.runRandomForestClassifier(train_X, train_y, validate_X, validate_y, max_depth = 5)
result = clf.predict(test_X)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')
models.writeOutputSubmission(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/submission14.csv', result)

#Run Gradient Boosting Regression
starttime = datetime.datetime.now()
train_error, test_error, clf = models.runGradientBoostingRegressor(train_X, train_y, validate_X, validate_y, max_depth = 3)
result = clf.predict(test_X)
endtime = datetime.datetime.now()
print ('takes '+str((endtime - starttime).seconds)+' seconds')
models.writeOutputSubmission(r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/submission14.csv', result)
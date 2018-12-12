import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import datetime
import modelsFunc as models

train_queries = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_train_queries_compacted.csv'
validate_queries = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_validate_queries_compacted.csv'

train_data = pd.read_csv(train_queries)
validate_data = pd.read_csv(validate_queries)
print("Successfully load the data")

train_y = train_data['stars_review']
validate_y = validate_data['stars_review']

train_data.drop('stars_review', axis = 1, inplace = True)
validate_data.drop('stars_review', axis = 1, inplace = True)

train_errors = list()
test_errors = list()
cross_train_errors = list()
cross_test_errors = list()

##Run random forest using different number of components of PCA
"""for i in range(510, 900, 50):
    selected_column = list(range(1, i))
    train_X = train_data[selected_column]
    validate_X = validate_data[selected_column]
    train_error, test_error = models.runRandomForest(train_X, train_y, validate_X, validate_y)
    train_errors.append(train_error)
    test_errors.append(test_error)"""

##Run random forest using different max_depth

#selected_column = list(range(1, 460))
train_X = train_data#[selected_column]
validate_X = validate_data#[selected_column]
X = train_X.values
y = train_y.values

for max_depth in range(1, 50):
    train_error, test_error, __ = models.runRandomForestClassifier(train_X, train_y, validate_X, validate_y, max_depth)
    cross_train_error, cross_test_error = 0, 0
    kf = KFold(n_splits = 5)
    for train_index, test_index in kf.split(X):
        temp_train_error, temp_test_error, _ = models.runRandomForestClassifier(X[train_index, : ], y[train_index], \
                                                                                X[test_index, : ], y[test_index], max_depth)
        cross_train_error += temp_train_error
        cross_test_error += temp_test_error

    cross_train_error /= 5.0
    cross_test_error /= 5.0

    print(max_depth, "*****", cross_train_error)
    print(max_depth, "*****", cross_test_error)
    print(max_depth, "*****", train_error)
    print(max_depth, "*****", test_error)
    cross_train_errors.append(cross_train_error)
    cross_test_errors.append(cross_test_error)
    train_errors.append(train_error)
    test_errors.append(test_error)

print(cross_train_errors)
print(cross_test_errors)
print(train_errors)
print(test_errors)
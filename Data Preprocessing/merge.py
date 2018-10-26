import pandas as pd

reviews_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/train_reviews.csv'
bussiness_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/business_transformed.csv'
users_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/users_transformed.csv'
output_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_train_queries.csv'

reviews = pd.read_csv(reviews_location)
bussiness = pd.read_csv(bussiness_location)
users = pd.read_csv(users_location)

"""
Delete the features in train_reviews.csv that the validate_queries.csv do not have.
----------------------------------------------------------------
Later we may see how to use these feature
----------------------------------------------------------------
"""
deleted_columns = ['cool', 'date', 'funny', 'text', 'useful']
for column in deleted_columns:
        reviews.drop(column, axis = 1, inplace = True)

###Merger the data
output = pd.merge(reviews, bussiness, on = ['business_id'], suffixes = ['_review', '_bussiness'])
output = pd.merge(output, users, on = ['user_id'], suffixes = ['', '_users'])

###Delete the useless attribute
useless_columns = ['business_id', 'review_id', 'user_id']
for column in useless_columns:
    output.drop(column, axis = 1, inplace = True)

output.to_csv(output_location, index = False)
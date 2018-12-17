import pandas as pd
import os

path = os.path.dirname(os.getcwd())
if not os.path.exists(path + "/Data/After Processing"):
    os.makedirs(path + "/Data/After Processing") 
reviews_location = path + r'/Data/test_queries.csv'

#Please customize this by changing the input and output path location
bussiness_location = path + r'/Data/After Processing/business_transformed_compacted.csv'
users_location = path + r'/Data/After Processing/users_transformed.csv'
output_location = path + r'/Data/After Processing/join_test_queries_compacted.csv'

reviews = pd.read_csv(reviews_location)
bussiness = pd.read_csv(bussiness_location)
users = pd.read_csv(users_location)

"""
Delete the features in train_reviews.csv that the validate_queries.csv do not have.
"""
#deleted_columns = ['cool', 'date', 'funny', 'text', 'useful']
#for column in deleted_columns:
#        reviews.drop(column, axis = 1, inplace = True)

###Merger the data
output = pd.merge(reviews, bussiness, on = ['business_id'], suffixes = ['_review', '_bussiness'])
output = pd.merge(output, users, on = ['user_id'], suffixes = ['', '_users'])

###Delete the useless attribute
#useless_columns = ['business_id', 'review_id', 'user_id']
useless_columns = ['business_id', 'user_id']
for column in useless_columns:
    output.drop(column, axis = 1, inplace = True)

output.to_csv(output_location, index = False)
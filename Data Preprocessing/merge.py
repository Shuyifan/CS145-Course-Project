import pandas as pd

reviews_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/train_reviews.csv'
bussiness_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/business.csv'
users_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/users_transformed.csv'
output_location = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_train_queries.csv'

reviews = pd.read_csv(reviews_location)
bussiness = pd.read_csv(bussiness_location)
users = pd.read_csv(users_location)

output = pd.merge(reviews, bussiness, on = ['business_id'])
output = pd.merge(output, users, on = ['user_id'])

output.to_csv(output_location)
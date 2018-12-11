#knn_test.py
import numpy as np
import pandas as pd
from numpy import linalg
import scipy 
import sklearn
from sklearn.decomposition import TruncatedSVD
import csv
import string
import time
from sklearn.metrics import mean_squared_error as mse
start_time = time.time()
def getDataframe(filePath):
    data = pd.read_csv(filePath)
    return data

numNeighbors = 9

#Idea: compute the knn test using the cosine distance as our metric. It makes no sense to compare the "distance" between
#user_id and business_id, so I want to concatenate the attributes of users.csv and business.csv. Thus, each user/business pair
#is assigned a string of integers, which is compared to that of another user/business pair to determine how "similar" they are.
#currently, all attributes are weighted evenly.


#parameters: x_1, x_2, two input variables [user_id, business_id]
#returns: cosine distance between the two based on user attributes and business attributes
def distance_metric(x_1, x_2):
    #use cosine_distances based on attributes
    #x_1, x_2 are ['user_id', 'business_id']
    #return sklearn.metrics.pairwise.cosine_distances(x_1, x_2)
    x_1_norm=np.sum(x_1*x_1)
    x_2_norm=np.sum(x_2*x_2)
    
    cosine_similarity=(np.sum( np.dot(x_1,x_2) ))/np.sqrt(x_1_norm*x_2_norm)
    return 1-cosine_similarity



#use cosine distance to compute K closest neighbors
#initializes the knn algorithm and tests the validate_queries elements
def knn_test(numNeighbors):    

    
    #store training target data
    #train_queries_compacted.csv
    #join_training_queries.csv
    train_csv = "train_queries_compacted.csv"


    target_values = pd.read_csv(train_csv, header=0)
    target_values = target_values['stars_review']

    #load training data
    training_input = pd.read_csv(train_csv)
    #print("training_input: ", training_input)
    training_input.drop('stars_review', axis = 1, inplace = True)


    #num_training=150231
    #fit the data
    user_neighbors = sklearn.neighbors.KNeighborsClassifier(n_neighbors = numNeighbors, weights='uniform', metric=distance_metric)
    #print("target_values: ", target_values)
    user_neighbors.fit(training_input, np.ravel(target_values))


    #load validating input data
    #validation_queries_compacted.csv
    #join_validate_queries.csv
    validate_csv = "validation_queries_compacted.csv"
    validating_input = pd.read_csv(validate_csv, header=0)
    #validating_input.rename(columns=validating_input.iloc[0], inplace=True) #set column headers
    #validating_input.drop(validating_input.index[0], inplace=True)

    validating_class = validating_input["stars_review"]


    #Test on validating data

    #print("validating_input: ", validating_input)
    validating_input.drop('stars_review', axis = 1, inplace = True)

    #test using validate_queries.csv
    RMSE=0
    num_validate = validating_input.shape[0]
   
    print("Predicting....")
    test_this = validating_input
    predicted_y = user_neighbors.predict(test_this)
    
    print("Predicting Time: --- %s seconds ---" %(time.time()-start_time))
    np.savetxt("knn_validate_predictions9.csv", predicted_y, delimiter=",")

    num_test = len(test_this) #num_validate
    print("Calculating MSE...")
    RMSE = mse(predicted_y, validating_class.head(num_test)) **0.5
    print("RMSE=", RMSE)

####################################################################################################

    #Run on actual test data
    #test_queries_compacted.csv
    test_csv = "test_queries_compacted.csv"
    test_input = pd.read_csv(test_csv, header=0)

    #test using test_queries_compacted.csv   
    print("Predicting Test queries....")
    test_this = test_input
    predicted_y = user_neighbors.predict(test_this)
    
    print("Predicting Time: --- %s seconds ---" %(time.time()-start_time))
    np.savetxt("knn_test_predictions9.csv", predicted_y, delimiter=",")

knn_test(numNeighbors)
print("Total Time: --- %s seconds ---" %(time.time()-start_time))
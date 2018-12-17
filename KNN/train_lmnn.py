#train_LMNN.py
import numpy as np
import pandas as pd
from numpy import linalg
import scipy 
import sklearn
from sklearn.decomposition import TruncatedSVD
import csv
import string
import time
from metric_learn import lmnn
from pylmnn import LargeMarginNearestNeighbor as LMNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error as mse
import os

start_time = time.time()

path = os.path.dirname(os.getcwd())
if not os.path.exists(path + "/Data/Submission"):
    os.makedirs(path + "/Data/Submission") 

#try columsn A-S aka 1-19 (inclusive)
train_csv = path + "/Data/After Processing/train_queries_compacted.csv"
numNeighbors = 35 #change this

#Idea: train the matrix M for use in mahalanobis metric function in other programs
def train_LMNN():    

    
    #store training target data
    print("Loading Training Data...")
    training_input = pd.read_csv(train_csv, header=0)
    target_values = training_input['stars_review']
    remove_these_cols = []
    #load training data
    training_input.drop('stars_review', axis = 1, inplace = True)
    training_input = training_input.iloc[:,0:20]


    print("Loading Validating Data...")
    #load validating input data
    #validation_queries_compacted.csv
    #join_validate_queries.csv
    validate_csv = path + "/Data/After Processing/validation_queries_compacted.csv"
    validating_input = pd.read_csv(validate_csv, header=0)
    validating_class = validating_input["stars_review"]
    validating_input.drop('stars_review', axis = 1, inplace = True)
    validating_input = validating_input.iloc[:,0:20]



    #num_training=150231
    # Instantiate the metric learner
    lmnn = LMNN(n_neighbors=numNeighbors, n_components=training_input.shape[1]) #number of neighbors for LMNN !!!
    print("Training LMNN....")
    # Train the metric learner
    lmnn.fit(training_input.values, np.ravel(target_values))

    # Fit the nearest neighbors classifier
    print("Training KNN....")
    knn = KNeighborsClassifier(n_neighbors=numNeighbors) #number of neighbors for KNN !!!
    knn.fit(lmnn.transform(training_input.values), np.ravel(target_values))

    #predict
    print("Predicting....")
    test_this = lmnn.transform(validating_input.values)
    predicted_y = knn.predict(test_this)
    
    #print("Predicting Time: --- %s seconds ---" %(time.time()-start_time))
        
    #np.savetxt("knn_validate_predictions_lmnn.csv", predicted_y, delimiter=",")

    # RMSE
    print("Calculating MSE...")
    RMSE = mse(predicted_y, validating_class) **0.5
    print("RMSE=", RMSE)


    ###########################
    print("Loading Testing Data...")
    test_csv = path + "/Data/After Processing/test_queries_compacted.csv"
    test_input = pd.read_csv(test_csv, header=0).iloc[:,0:20]
    
    #predict
    print("Predicting TESTING....")
    test_this = lmnn.transform(test_input.values)
    predicted_y = knn.predict(test_this)
            
    np.savetxt(path + "/Data/Submission/knn_test_predictions_lmnn.csv", predicted_y, delimiter=",")

    

train_LMNN()
print("Total Time: --- %s seconds ---" %(time.time()-start_time))

import pandas as pd
import numpy as np
import sys
from pprint import pprint
from sklearn.cluster import KMeans
import csv

#with open('business.csv', "r") as infile:
#    with open('train_review_transformed.csv', "w") as outfile:
#        reader = csv.reader(infile)
#        writer = csv.writer(outfile)
#        add_dataset = getDataframe(

# Reads the data from CSV files, each attribute column can be obtained via its name, e.g., y = data['y']
def getDataframe(filePath):
    data = pd.read_csv(filePath)
    return data

def cluster_by_category(filePath, numClusters, output):
    categories = getDataframe(filePath)
    X = categories.drop('index', axis=1)
    i = categories['index']
    
    kmeans = KMeans(n_clusters=numClusters, random_state=0)
    kmeans.fit(X)
    
    if output is True:
        n = categories.shape[0]
        output = np.zeros((n,2))
        output[:,0] = i
        output[:,1] = kmeans.labels_
        np.savetxt('clustered_by_category.csv', output, delimiter = ',', newline = '\n')
    
    return kmeans

file = 'business_categories.csv'
numClusters = 8
cluster_by_category(file, numClusters, True)
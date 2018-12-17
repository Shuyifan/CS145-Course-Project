import pandas as pd
import numpy as np
import os
from pprint import pprint
from sklearn.cluster import KMeans
import csv

"""
Reads the data from CSV files, each attribute column can be obtained via its name

:param filePath: the file path of the input data frame

:type filePath: str

:returns: return the dataframe after read the data from the .csv file

:rtype: pandas.DataFrame
"""
def getDataframe(filePath):
    data = pd.read_csv(filePath)
    return data

"""
Perform the K-means algorithm to cluster the data.

:param filePath: the file path of the input data frame
:param numClusters: number of clusters, which is a hyper-parameter for K-means algorithm.
:param output: whether we will result the result

:type filePath: str
:type numClusters: int
:type outfilename: str

:returns kmeans: return the Kk-means model after we fit the data from the filePath

:rtype kmeans: sklearn.cluster.KMeans
"""
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
        path = os.path.dirname(os.getcwd())
        if not os.path.exists(path + "/Data/After Processing"):
            os.makedirs(path + "/Data/After Processing") 
        np.savetxt(path + '/Data/After Processing/clustered_by_category.csv', output, delimiter = ',', newline = '\n')
    
    return kmeans

path = os.path.dirname(os.getcwd())
file = r'/Data/business_categories.csv'
numClusters = 8
cluster_by_category(path + file, numClusters, True)
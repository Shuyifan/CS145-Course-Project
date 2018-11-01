import pandas as pd
import numpy as np
import sys
from pprint import pprint
from sklearn.decomposition import PCA
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

def perform_PCA(filePath, limitComponents, numComponents, output, outfilename):
    data = getDataframe(filePath)
    y = data['stars_review']
    X = data.drop('stars_review', axis=1)
    
    if limitComponents is True:
        pca = PCA(n_components=numComponents).fit(X)
    else:
        pca = PCA().fit(X)
    
    if output is True:
        n = data.shape[0]
        m = pca.components_.shape[0] + 1
        output = np.zeros((n,m))
        output[:,0] = y
        output[:,1:] = X.dot(pca.components_.T)
        np.savetxt(outfilename, output, delimiter = ',', newline = '\n')
    
    print(pca.explained_variance_ratio_)
    
    return pca

infile = 'join_train_queries.csv'
numComponents = 10
outfile = 'queries_under_pca.csv'
perform_PCA(infile, True, numComponents, True, outfile)
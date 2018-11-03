import pandas as pd
import numpy as np
import sys
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv

#with open('business.csv', "r") as infile:
#    with open('train_review_transformed.csv', "w") as outfile:
#        reader = csv.reader(infile)
#        writer = csv.writer(outfile)
#        add_dataset = getDataframe(

# , , e.g., y = data['y']
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
Perform the PCA, fit the PCA based on the dataframe from "filePath" and perform the transformation to that dataframe, write
    the result to the "outfilename" if needed

:param filePath: the file path of the input data frame
:param limitComponents: determine whether there is a limiation on the components we selected after PCA; if no, we just selected every
                        components after the PCA
:param numComponents: the number of components we selected after the PCA
:param output: whether we will result the result
:param outfilename: the place where we will write the data after PCA transformation

:type filePath: str
:type limitComponents: bool
:type numComponents: int
:type output: bool
:type outfilename: str

:returns pca: return the PCA model after we fit the data from the filePath
:returns standard_scaler: return the standard scaler after we fit the data from the filePath

:rtype pca: sklearn.decomposition.PCA
:rtype standard_scaler: sklearn.preprocessing.StandardScaler
"""
def perform_PCA(filePath, limitComponents, numComponents, output, outfilename):
    data = getDataframe(filePath)
    y = data['stars_review']
    X = data.drop('stars_review', axis=1)
    
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    X = standard_scaler.transform(X)

    if limitComponents is True:
        pca = PCA(n_components=numComponents).fit(X)
    else:
        pca = PCA().fit(X)
    
    if output is True:
        n = data.shape[0]
        m = pca.components_.shape[0] + 1
        output = np.zeros((n,m))
        output[ : , 0] = y
        output[ : , 1 :] = pca.transform(X)
        np.savetxt(outfilename, output, delimiter = ',', newline = '\n')

    print(pca.explained_variance_ratio_.cumsum()) #Print the variance contribution (cumulatively) for each components
    return pca, standard_scaler

"""
Perform the normalization transforamtion and PCA transformation based on already fitted model. Perform the PCA tranforamtion 
    on the dataframe from "filePath" and write the result to the "outfilename"

:param filePath: the file path of the input data frame
:param pca: the already fitted PCA model, which we can use to transforma the data
:param standard_scaler: the already fitted scaler, which we can use to transforma the data
:param outfilename: the place where we will write the data after PCA transformation

:type filePath: str
:type pca: sklearn.decomposition.PCA
:type standard_scaler: sklearn.preprocessing.StandardScaler
:type outfilename: str
"""
def transform_PCA(filePath, pca, standard_scaler, outfilename):
    data = getDataframe(filePath)
    y = data['stars_review']
    X = data.drop('stars_review', axis=1)
    X = standard_scaler.transform(X)

    n = data.shape[0]
    m = pca.components_.shape[0] + 1
    output = np.zeros((n,m))
    output[ : , 0] = y
    output[ : , 1 :] = pca.transform(X)
    np.savetxt(outfilename, output, delimiter = ',', newline = '\n')

join_train_queries = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_train_queries.csv'
train_queries_pca = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/train_queries_pca.csv'

join_validate_queries = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/join_validate_queries.csv'
validate_queries_pca = r'/mnt/c/Users/shuyi/OneDrive/CS145/Data/after preprocess/validate_queries_pca.csv'

pca_model, scalar = perform_PCA(join_train_queries, False, 0, True, train_queries_pca)
transform_PCA(join_validate_queries, pca_model, scalar, validate_queries_pca)
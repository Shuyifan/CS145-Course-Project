import pandas as pd
import numpy as np
import sys
import os
from pprint import pprint
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error as mse
import csv

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
from sklearn.metrics import confusion_matrix

"""
The class implements a Support Vector Classification. The class contains the function with initialization, training, and prediction.

:attribute data: the training data
:attribute kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
                   ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used.
:attribute max_iter: Hard limit on iterations within solver, or -1 for no limit.
:attribute degree: Degree of the polynomial kernel function (‘poly’).
:attribute C: Penalty parameter C of the error term.

:type data: pandas.DataFrame
:type kernel: str
:type max_iter: int
:type degree: int
:type C: float
"""
class YelpSVM(object):
    """
    The init function for the class. Initialize the need attribute.
    
    :param kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
                       ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used.
    :param max_iter: Hard limit on iterations within solver, or -1 for no limit.
    :param degree: Degree of the polynomial kernel function (‘poly’).
    :param C: Penalty parameter C of the error term.

    :type kernel: str
    :type max_iter: int
    :type degree: int
    :type C: float
    """
    def __init__(self, kernel = 'linear', degree = 3, max_iter = -1, C = 1.0):
        self.data = None
        self.kernel = kernel
        self.max_iter = max_iter
        self.degree = degree
        self.C = C
        
        self.__mean = 0
        self.__std = 0
        self.__X_sub = [1, 2, 3, 4, 5]
        self.__y_sub = [1, 2, 3, 4, 5]
        self.__svm_array = [[], [21], [31, 32], [41, 42, 43], [51, 52, 53, 54]]
    
    """
    Train the SVC model given the training data.
    
    :param infile: the location of the training dataset.

    :type infile: str
    """
    def create_SVMs(self, infile):
        print("retrieving data")
        self.data = self.__getDataframe(infile)
        y = self.data['stars_review']
        X = self.data.drop('stars_review', axis=1)
        self.__mean = X.mean(0)
        self.__std = X.std(0)
        
        for i in range(0, 5):
            print("obtaining data for class ", i+1)
            X, y = self.__getPointsOfClass(self.data, i + 1)
            self.__X_sub[i] = (X - self.__mean).div(self.__std).replace([np.inf, -np.inf, np.nan], 0)
            self.__y_sub[i] = y
        
        for i in range(0, 5):
            for j in range(0, i):
                print("fitting svm ", i + 1, j + 1)
                X = self.__X_sub[i].append(self.__X_sub[j])
                y = self.__y_sub[i].append(self.__y_sub[j])
                
                self.__svm_array[i][j] = SVC(kernel = self.kernel, degree = self.degree, max_iter = self.max_iter, C = self.C)
                self.__svm_array[i][j].fit(X, y)
    
    """
    Use the trained SVC model to give a prediction given an input unseen data.
    
    :param infile: the location of the test data or validation data.
    :param validate: tell whether the input data is validation data or test data. If True, the input data is validation data, and
                     the function will draw a confusion matrix. If false, the function will just return the prediction.

    :type infile: str
    :type validate: bool

    :return y: the prediction predicted by the trained SVC model given the input data.

    :rtype y: numpy.ndarray
    """
    def predict(self, infile, validate = False):
        print("retrieving data")
        data = self.__getDataframe(infile)

        if validate is True:
            X = data.drop('stars_review', axis=1)
            validate_y = data['stars_review']
        else:
            X = data
        
        X = (X - self.__mean).div(self.__std).replace([np.inf, -np.inf, np.nan], 0)
        
        predictions = [[], [21], [31, 32], [41, 42, 43], [51, 52, 53, 54]]
        
        for i in range(0, 5):
            for j in range(0, i):
                print("running svm ", i+1, j+1)
                predictions[i][j] = self.__svm_array[i][j].predict(X)
        
        print("concluding")
        y = np.zeros(predictions[1][0].shape)
        predict_counts = np.zeros((y.shape[0], 5))
        for i in range(0, 5):
            for j in range(0, i):
                for p in range(0, predictions[i][j].shape[0]):
                    predict_counts[p][int(predictions[i][j][p]) - 1] += 1
        
        for p in range(0, predictions[1][0].shape[0]):
            bestpick = 1
            bestpickcount = predict_counts[p][0]
            for i in range(1, 5):
                if predict_counts[p][i] > bestpickcount:
                    bestpick = i+1
                    bestpickcount = predict_counts[p][i]
            y[p] = bestpick
        
        if validate is True:
            print(mse(y, validate_y) ** 0.5)
            mat = confusion_matrix(y, validate_y)
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
            plt.xlabel('true label')
            plt.ylabel('predicted label')
            plt.tight_layout()
            plt.savefig('./conf_matrix.png')
        return y
    
    def __getDataframe(self, filePath):
        data = pd.read_csv(filePath)
        return data

    def __getPointsOfClass(self, dataframe, classToUse):
        data = dataframe.loc[dataframe['stars_review'] == classToUse]
        y = data['stars_review']
        X = data.drop('stars_review', axis = 1)
        return X, y

path = os.path.dirname(os.getcwd())
if not os.path.exists(path + r"/Data/Submission"):
    os.makedirs(path + r"/Data/Submission") 

trainfile = path + r'/Data/After Processing/train_queries_compacted.csv'
validatefile = path + r'/Data/After Processing/validation_queries_compacted.csv'
testfile = path + r'/Data/After Processing/test_queries_compacted.csv'

ysvm = YelpSVM(max_iter = 300000, C = 1.0, kernel = 'rbf')
ysvm.create_SVMs(trainfile)
y = ysvm.predict(validatefile, validate = True)
print(y)

y = ysvm.predict(testfile, validate = False)
print(y)

df = pd.DataFrame(y, columns=['stars'])
df.to_csv(path + r"/Data/Submission/results.csv", index = True, index_label = 'index')
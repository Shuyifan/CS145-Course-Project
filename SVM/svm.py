import pandas as pd
import numpy as np
import sys
from pprint import pprint
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error as mse
import csv

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
from sklearn.metrics import confusion_matrix

class YelpSVM(object):
    def __init__(self, kernel='linear', degree=3, max_iter=-1, C=1.0):
        self.kernel = kernel
        self.max_iter = max_iter
        self.degree = degree
        self.C = C
        self.X_sub = [1, 2, 3, 4, 5]
        self.y_sub = [1, 2, 3, 4, 5]
        self.svm_array = [[], [21], [31, 32], [41, 42, 43], [51, 52, 53, 54]]
    
    def getDataframe(self, filePath):
        data = pd.read_csv(filePath)
        return data
    
    def getPointsOfClass(self, dataframe, classToUse):
        data = dataframe.loc[dataframe['stars_review'] == classToUse]
        y = data['stars_review']
        X = data.drop('stars_review', axis=1)
        return X, y
    
    def create_SVMs(self, infile):
        print("retrieving data")
        self.data = self.getDataframe(infile)
        y = self.data['stars_review']
        X = self.data.drop('stars_review', axis=1)
        self.mean = X.mean(0)
        self.std = X.std(0)
        
        for i in range(0, 5):
            print("obtaining data for class ", i+1)
            X, y = self.getPointsOfClass(self.data, i+1)
            self.X_sub[i] = (X - self.mean).div(self.std).replace([np.inf, -np.inf, np.nan], 0)
            self.y_sub[i] = y
        
        for i in range(0, 5):
            for j in range(0, i):
                print("fitting svm ", i+1, j+1)
                X = self.X_sub[i].append(self.X_sub[j])
                y = self.y_sub[i].append(self.y_sub[j])
                
                self.svm_array[i][j] = SVC(kernel=self.kernel, degree=self.degree, max_iter=self.max_iter, C=self.C)
                self.svm_array[i][j].fit(X, y)
    
    def predict(self, infile, validate=False):
        print("retrieving data")
        data = self.getDataframe(infile)

        if validate is True:
            X = data.drop('stars_review', axis=1)
            validate_y = data['stars_review']
        else:
            X = data
        
        X = (X - self.mean).div(self.std).replace([np.inf, -np.inf, np.nan], 0)
        
        predictions = [[], [21], [31, 32], [41, 42, 43], [51, 52, 53, 54]]
        
        for i in range(0, 5):
            for j in range(0, i):
                print("running svm ", i+1, j+1)
                predictions[i][j] = self.svm_array[i][j].predict(X)
        
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
            print(mse(y, validate_y)**0.5)
            mat = confusion_matrix(y, validate_y)
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
            plt.xlabel('true label')
            plt.ylabel('predicted label')
            plt.tight_layout()
            plt.savefig('./conf_matrix.png')

        
        return y

trainfile = 'train_queries_compacted.csv'
validatefile = 'validation_queries_compacted.csv'
testfile= 'test_queries_compacted.csv'

ysvm = YelpSVM(max_iter=300000, C=1.0, kernel='rbf')
ysvm.create_SVMs(trainfile)
y = ysvm.predict(validatefile, validate=True)
print(y)

y = ysvm.predict(testfile, validate=False)
print(y)

df = pd.DataFrame(y, columns=['stars'])
df.to_csv("./results.csv", index = True, index_label = 'index')

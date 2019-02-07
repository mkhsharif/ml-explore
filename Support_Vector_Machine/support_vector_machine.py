import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd


'''
Support Vector Machine:
an application of the k nearest neighbors 
algorithm to detect breast cancer based on
the classification of cells by their attributes

Dataset Source:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
'''

# read in dataset
df = pd.read_csv('../Data/breast-cancer-wisconsin.data.txt')
# treat unknowns as outliers
df.replace('?', -99999, inplace=True)
# drop useless data like id 
# (has nothing to do with identifing if a tumor is cancerous or not)
df.drop(['id'], 1, inplace=True)

# features, everything but class
X = np.array(df.drop(['class'],1))
# labels, class
y = np.array(df['class'])

# cross validation 
# (shuffle data and separate it into training and testing chucks) 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

# fit data using K Nearest Neighbors classifier 
clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# custom prediction of new value
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
# e.g. X.reshape(-1, 1) if data has single feature
# e.g. X.reshape(1, -1) if data has single sample
example_measures = example_measures.reshape(2, -1)
prediction = clf.predict(example_measures)
print(prediction)
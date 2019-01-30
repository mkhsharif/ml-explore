import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

# read in dataset
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
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
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
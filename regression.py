import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# Forecast stock prices with linear regression
# parsed from Quandl

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# high - low percent (volatility)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# label 
forecast_col = 'Adj. Close'
# fill nan(not a number) data instead of getting rid of column
# i.e -99999 will be treated as an outlier in the dataset
df.fillna(-99999, inplace=True)
#  get percentage of dataframe (1%)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
# shift up the label column to forecast the price
df['label'] = df[forecast_col].shift(-forecast_out)
# features X, labels y
# drop labels from dataframe and return features stored into x
X = np.array(df.drop(['label'],1))
y =  np.array(df['label'])
# scale X before feeding into classifier
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
# remove missing values
df.dropna(inplace=True)
y = np.array(df['label'])
# create training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# classifier
clf = LinearRegression()
# fit(train) features, labels
clf.fit(X_train, y_train) 
# test(score) features, labels 
accuracy = clf.score(X_test, y_test)
#print(accuracy)
# make prediction on each value in array (or single value)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style 
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

# define two classes and their features
dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            # list of lists where first item is distances and second is the group within the distance
            distances.append([euclidean_distance, group])

    # populate votes in list up to k
    votes = [i[1] for i in sorted(distances) [:k]]
    # take the most common vote 
    vote_result = Counter(votes).most_common(1)[0][0]
    print(Counter(votes).most_common(1))

    return vote_result

'''
Algorithm Test 

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# plot data points on  graph
for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0], j[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], color=result)

# show visual representation of dataset
plt.show()
'''

# Algorithm Application with real Data

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
# splice data 
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
# training data (up to last 20%)
train_data = full_data[:-int(test_size*len(full_data))]
# testing data (last 20%)
test_data = full_data[-int(test_size*len(full_data)):]

# populate train/test set dictionary
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        # if group from test set equals vote from knn classifier mark as correct
        if group == vote:
            correct += 1
        total += 1 

print('Accuracy:', correct/total)
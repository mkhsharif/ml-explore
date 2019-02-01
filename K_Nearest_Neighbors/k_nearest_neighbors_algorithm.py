import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style 
from collections import Counter
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

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# plot data points on  graph
for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0], j[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], color=result)

# show visual representation of dataset
plt.show()

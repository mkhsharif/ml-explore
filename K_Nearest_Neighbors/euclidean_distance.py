from math import sqrt

'''
Euclidean Distance:
is the straight-line distance between two points
in Euclidean space

sqrt(nSi=1(Qi - Pi)^2)
'''

# define points
plot1 = [1, 3]
plot2 = [2, 5]

# calculate euclidean distance
euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
print(euclidean_distance)
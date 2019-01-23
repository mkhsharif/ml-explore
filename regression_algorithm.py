from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import random 
style.use('fivethirtyeight')


'''
Linear Regression Algorithm 
Steps:
1. Gather Data
2. Fit Model to Data 
3. Make prediction on Data
'''
#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,4,6,7], dtype=np.float64)

# creates pseudo random dataset 
def create_dataset(num_of_pts, variance, step=2, correlation=False):
    val = 1
    ys = []
    # interate through range and append new y 
    for i in range(num_of_pts):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
    # add step to value
    #  pos = ascending data pts, neg = descending data pts
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    # add xs for each data point
    xs = [i for i in range(len(ys))]
    # return coordinates
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# generate the best fit slope and intercept
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b

# difference between ys original and ys line squared
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

# r squared formula: r^2 = SE Y-Hat / SE mean(Y)
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

# create dataset
xs, ys = create_dataset(40, 40, 2, correlation='pos')

# generate Y-Hat for new dataset
m, b  = best_fit_slope_and_intercept(xs,ys)
print(m,b)

# create regression line
regression_line = [(m*x)+b for x in xs]

# make prediction
predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()
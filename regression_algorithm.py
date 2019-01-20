from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,4,6,7], dtype=np.float64)

# generate the best fit slope and intercept
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b  = best_fit_slope_and_intercept(xs,ys)
print(m,b)

# create regression line
regression_line = [(m*x)+b for x in xs]

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualiztion=True):
        self.visualiztion = visualiztion
        self.colors = {1: 'r', -1:'b'}
        if self.visualiztion:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                       [-1,1],
                       [-1,-1],
                       [1,-1]]
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # find global min
        # take bigger steps to find val
        # then take smaller steps
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense (threading? multi-process?)
                      self.max_feature_value * 0.001]

        # extremely expensive
        # b doesn't need to be as precise as w
        b_range_multiple = 5
        # b takes bigger steps than w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # because convex we can know when to stop
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), 
                                       self.max_feature_value*b_range_multiple, 
                                       step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in svm
                        # smo(sequential min optimation) fixes slightly
                        # have to run on each sample
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    # if one sample doesn't fit the def
                                    # the whole dataset is invalid
                                    found_option = False 

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b] 

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            
            # sorted list of all the magnitudes
            norms = sorted([n for n in opt_dict])
            # optimal choice is the smallest norm
            opt_choice = opt_dict[norms[0]]
            # ||w|| : [w,b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice





    def predict(self, data):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        return classification


data_dict = {-1:np.array([[1,7],
                           [2,8],
                           [3,8]]),

                1:np.array([[5,1],
                           [6,-1],
                           [7,3]])}


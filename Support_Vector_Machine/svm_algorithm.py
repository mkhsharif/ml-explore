import matplotlib.pyplot as plt 
from matplotlib import style
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
        pass

    def predict(self, data):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        return classification


data_dict = -{-1:np.array([[1,7],
                           [2,8]
                           [3,8]]),

               1:np.array([[5,1],
                           [6,-1],
                           [7,3]])}


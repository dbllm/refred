import numpy

class DefaultModel():
    
    def fit(self, x, y):
        #  do nothing
        pass
    
    def predict(self, x):
        # distance is always 0
        # print(x)
        # print(type(x), x.shape)
        # exit()
        return numpy.zeros([x.shape[0],])
    

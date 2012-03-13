
import numpy
from apgl.util.Parameter import Parameter


class FeatureGenerator(object):
    """
    A class to compute new features from exisiting ones.
    """
    def __init__(self):
        pass


    def categoricalToIndicator(self, X, indices):
        """
        Convert a set of categorical variables to indicator ones.
        """
        Parameter.checkList(indices, Parameter.checkIndex, (0, X.shape[1]))

        X2 = numpy.zeros((X.shape[0], 0)) 

        for i in range(X.shape[1]):
            if i in indices:
                categories = numpy.unique(X[:, i])
                Z = numpy.zeros((X.shape[0], categories.shape[0]))

                for j in range(categories.shape[0]):
                    Z[:, j] = X[:, i] == categories[j]

                X2 = numpy.c_[X2, Z]
            else:
                X2 = numpy.c_[X2, X[:, i]]

        return X2

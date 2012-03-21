from apgl.predictors.AbstractPredictor import AbstractPredictor
import numpy

class BinomialClassifier(AbstractPredictor):
    """
    A classifier for binary labelled data which randomly selects the label according
    to the binomial distribution.
    """
    def __init__(self, p):
        """
        Initialise with probability of a positive label, p.
        """
        self.p = p 

    def learnModel(self, X, y):
        """
        No learning required.
        """
        pass

    def classify(self, X):
        numExamples = X.shape[0]
        y = (numpy.random.rand(numExamples, 1) < self.p)*2 - 1

        return y


    p = None 
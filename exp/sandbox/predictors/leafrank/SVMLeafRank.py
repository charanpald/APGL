import numpy
import logging
from apgl.util.Sampling import Sampling
from apgl.predictors.LibSVM import LibSVM

class SVMLeafRank(LibSVM):
    """
    This is a subclass of LibSVM which will do model selection before learning. 
    """
    def __init__(self, paramDict, folds):
        super(SVMLeafRank, self).__init__()
        self.paramDict = paramDict
        self.folds = folds 
        self.chunkSize = 2
        self.setMetricMethod("auc")          
            
    def generateLearner(self, X, y):
        """
        Train using the given examples and labels, and use model selection to
        find the best parameters.
        """
        if numpy.unique(y).shape[0] != 2:
            print(y)
            raise ValueError("Can only operate on binary data")

        #Do model selection first 
        idx = Sampling.crossValidation(self.folds, X.shape[0])
        learner, meanErrors = self.parallelModelSelect(X, y, idx, self.paramDict)
        
        return learner

    def getBestLearner(self, meanErrors, paramDict, X, y, idx=None):
        """
        As we are using AUC we will look for the max value. 
        """
        return super(SVMLeafRank, self).getBestLearner(meanErrors, paramDict, X, y, idx, best="max")
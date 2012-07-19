
"""
A wrapper for the Decision Tree learner in scikits.learn with model selection 
functionality. 
""" 
import numpy
import logging 

from apgl.util.Evaluator import Evaluator 
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util 
from apgl.predictors.AbstractWeightedPredictor import AbstractWeightedPredictor


class RandomForest(AbstractWeightedPredictor):
    def __init__(self, numTrees=10, maxFeatures="auto", criterion="gini", maxDepth=10, minSplit=30, type="class"):
        try: 
            from sklearn import ensemble
        except ImportError as error:
            logging.debug(error)
            return 
        super(RandomForest, self).__init__()
        
        self.numTrees = numTrees
        self.maxFeatures = maxFeatures 
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.type = type
        
        self.maxDepths = numpy.arange(1, 10)
        self.minSplits = numpy.arange(10, 51, 10)

    def setMinSplit(self, minSplit):
        Parameter.checkInt(minSplit, 0, float('inf'))
        self.minSplit = minSplit

    def getMinSplit(self):
        return self.minSplit 
        
    def setNumTrees(self, numTrees):
        Parameter.checkInt(numTrees, 0, float('inf'))
        self.numTrees = numTrees
        
    def getNumTrees(self): 
        return self.numTrees

    def getMinSplit(self):
        return self.minSplit 
        
        
    def getMinSplits(self): 
        return self.minSplits
    
    def getMaxDepths(self): 
        return self.maxDepths

    def setMaxDepth(self, maxDepth):
        Parameter.checkInt(maxDepth, 1, float('inf'))
        self.maxDepth = maxDepth

    def getMaxDepth(self):
        return self.maxDepth 

    def learnModel(self, X, y):
        try: 
            from sklearn import ensemble
        except ImportError as error:
            logging.debug(error)
            return 

        classes = numpy.unique(y)
        if classes.shape[0] == 2: 
            self.worstResponse = classes[classes!=self.bestResponse][0]
        
        if self.type == "class": 
            self.learner = ensemble.RandomForestClassifier(n_estimators=self.numTrees, max_features=self.maxFeatures, criterion=self.criterion, max_depth=self.maxDepth, min_samples_split=self.minSplit, random_state=21)
        else: 
            self.learner = ensemble.RandomForestRegressor(n_estimators=self.numTrees, max_features=self.maxFeatures, criterion=self.criterion, max_depth=self.maxDepth, min_samples_split=self.minSplit, random_state=21)          
            
        self.learner = self.learner.fit(X, y)

    def getLearner(self):
        return self.learner

    def getClassifier(self): 
        return self.learner 
        

    def predict(self, X):
        predY = self.learner.predict(X)
        return predY
        
    def copy(self): 
        randomForest = RandomForest(numTrees=self.numTrees, maxFeatures=self.maxFeatures, criterion=self.criterion, maxDepth=self.maxDepth, minSplit=self.minSplit, type=self.type)
        return randomForest 
        
    @staticmethod
    def generate(maxDepth=10, minSplit=30):
        def generatorFunc():
            randomForest = RandomForest()
            randomForest.setMaxDepth(maxDepth)
            randomForest.setMinSplit(minSplit)
            return randomForest
        return generatorFunc


        
    def getMetricMethod(self): 
        if self.type == "class": 
            return Evaluator.binaryError
        else:
            return Evaluator.rootMeanSqError 
        
    def __str__(self): 
        outputStr = self.type 
        outputStr += " maxDepth=" + str(self.maxDepth)
        outputStr += " minSplit=" + str(self.minSplit)
        outputStr += " criterion=" + str(self.criterion)
        outputStr += " numTrees=" + str(self.numTrees)
        outputStr += " maxFeatures=" + str(self.maxFeatures)
        return outputStr 

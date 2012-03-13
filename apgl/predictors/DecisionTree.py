
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


class DecisionTree(AbstractWeightedPredictor):
    def __init__(self, criterion="gini", maxDepth=10, minSplit=30, type="class"):
        try: 
            from sklearn import tree
        except ImportError as error:
            logging.debug(error)
            return 
        super(DecisionTree, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        
        self.maxDepths = numpy.arange(1, 10)
        self.minSplits = numpy.arange(10, 51, 10)

    def setMinSplit(self, minSplit):
        Parameter.checkInt(minSplit, 0, float('inf'))
        self.minSplit = minSplit

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
            from sklearn import tree
        except ImportError as error:
            logging.debug(error)
            return 

        classes = numpy.unique(y)
        if classes.shape[0] == 2: 
            self.worstResponse = classes[classes!=self.bestResponse][0]
        
        if self.type == "class": 
            self.learner = tree.DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, min_split=self.minSplit)
        else: 
            self.learner = tree.DecisionTreeRegressor(criterion=self.criterion, max_depth=self.maxDepth, min_split=self.minSplit)          
            
        self.learner = self.learner.fit(X, y)

    def getLearner(self):
        return self.learner

    def getClassifier(self): 
        return self.learner 
        
    def getTree(self): 
        return self.learner.tree_

    def predict(self, X):
        predY = self.learner.predict(X)
        return predY
        
    def copy(self): 
        try: 
            from sklearn import tree
        except ImportError as error:
            logging.debug(error)
            return 
        decisionTree = DecisionTree(criterion=self.criterion, maxDepth=self.maxDepth, minSplit=self.minSplit, type=self.type)
        return decisionTree 
        
    @staticmethod
    def generate(maxDepth=10, minSplit=30):
        def generatorFunc():
            decisionTree = DecisionTree()
            decisionTree.setMaxDepth(maxDepth)
            decisionTree.setMinSplit(minSplit)
            return decisionTree
        return generatorFunc

    def parallelVfcv(self, X, y, idx, type="gini"):
        """
        Perform v fold penalisation model selection using the decision tree learner
        and then pick the best one. Using the best set of parameters train using
        the whole dataset. 

        :param X: The examples as rows
        :type X: :class:`numpy.ndarray`

        :param y: The binary -1/+1 labels
        :type y: :class:`numpy.ndarray`

        :param idx: A list of train/test splits
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)
        
        paramDict = {} 
        paramDict["setMinSplit"] = self.minSplits
        paramDict["setMaxDepth"] = self.maxDepths

        return self.parallelModelSelect(X, y, idx, paramDict)
        
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
        return outputStr 

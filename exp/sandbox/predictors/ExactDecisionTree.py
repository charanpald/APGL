"""
A class to test the penalty bound of Nobel by searching all possible trees. 
"""

import numpy 
from apgl.graph.DictTree import DictTree 
from apgl.util.Util import Util 
from apgl.util.Evaluator import Evaluator 
from apgl.util.Parameter import Parameter 
from exp.sandbox.predictors.DecisionNode import DecisionNode
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2, findBestSplitRisk
from apgl.predictors.AbstractPredictor import AbstractPredictor

class ExactDecisionTree(AbstractPredictor): 
    def __init__(self, criterion="gain", maxDepth=10, minSplit=30, learnType="reg", gamma=0.01):
        """
        Learn a decision tree with penalty proportional to the root of the size 
        of the tree as in Nobel 2002. We use a stochastic approach in which we 
        learn a set of trees randomly and choose the best one. 

        :param criterion: The splitting criterion which is only informaiton gain currently 

        :param maxDepth: The maximum depth of the tree 
        :type maxDepth: `int`

        :param minSplit: The minimum size of a node for it to be split. 
        :type minSplit: `int`
        
        :param learnType: The type of learning to perform. Currently only regression 
        
        :param gamma: The weight on the penalty factor between 0 and 1
        :type gamma: `float`
        
        """
        super(ExactDecisionTree, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.learnType = learnType
        self.setGamma(gamma)
        self.alphaThreshold = 0.0
        
    def learnModel(self): 
        if numpy.unique(y).shape[0] != 2: 
            raise ValueError("Must provide binary labels")
        if y.dtype != numpy.int: 
            raise ValueError("Labels must be integers")
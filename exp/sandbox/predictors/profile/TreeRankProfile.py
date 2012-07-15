import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.predictors.leafrank.SVMLeafRank import SVMLeafRank
from exp.sandbox.predictors.TreeRank import TreeRank

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class TreeRankProfile(object):
    def __init__(self):
        self.folds = 3 
        self.paramDict = {} 
        self.paramDict["setC"] = 2**numpy.arange(-5, 5, dtype=numpy.float)  
        self.leafRanklearner = SVMLeafRank(self.paramDict, self.folds)

    def profileLearnModel(self):
        treeRank = TreeRank(self.leafRanklearner)
        treeRank.setMaxDepth(2)
        treeRank.setMinSplit(50)

        numExamples = 1000
        numFeatures = 950

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)*2-1

        def run():
            for i in range(5):
                print("Iteration " + str(i))
                treeRank.learnModel(X, Y)
                #print(treeRank.getTreeSize())
                #print(treeRank.getTreeDepth())

        ProfileUtils.profile('run()', globals(), locals())

profiler = TreeRankProfile()
profiler.profileLearnModel()

#Takes 2.34 s versus 45 seconds for TreeRankR 
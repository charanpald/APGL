import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from exp.metabolomics.TreeRankForestR import TreeRankForestR

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class TreeRankForestRProfile(object):
    def __init__(self):
        pass

    def profileLearnModel(self):
        treeRankForest = TreeRankForestR()
        treeRankForest.printMemStats = True
        treeRankForest.setMaxDepth(2)
        treeRankForest.setNumTrees(5)

        numExamples = 650
        numFeatures = 950

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)

        def run():
            for i in range(10):
                print("Iteration " + str(i))
                treeRankForest.learnModel(X, Y)
                #print(treeRank.getTreeSize())
                #print(treeRank.getTreeDepth())

        ProfileUtils.profile('run()', globals(), locals())

profiler = TreeRankForestRProfile()
profiler.profileLearnModel()

#Limiting the depth improves memory
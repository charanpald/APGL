import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from apgl.metabolomics.leafrank.LinearSVM import LinearSVM
from apgl.metabolomics.TreeRank import TreeRank

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class TreeRankProfile(object):
    def __init__(self):
        pass

    def profileLearnModel(self):
        treeRank = TreeRank(LinearSVM.generate(10.0))
        treeRank.setMaxDepth(2)
        treeRank.setMinSplit(50)

        numExamples = 650
        numFeatures = 950

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)

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
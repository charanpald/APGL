
import numpy
import unittest
import logging 
import sys 
from exp.sandbox.predictors.TreeRankForest import TreeRankForest
from exp.sandbox.predictors.RankNode import RankNode
from exp.sandbox.predictors.leafrank.SVMLeafRank import SVMLeafRank
from exp.sandbox.predictors.leafrank.DecisionTree import DecisionTree
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Evaluator import Evaluator
from apgl.data.Standardiser import Standardiser

class TreeRankForestTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr(all="raise")

        self.folds = 5 
        self.paramDict = {} 
        self.paramDict["setC"] = 2**numpy.arange(-10, 10, dtype=numpy.float)  
        #self.paramDict["setC"] = numpy.array([10], dtype=numpy.float)  
        self.leafRanklearner = SVMLeafRank(self.paramDict, self.folds)

        numExamples = 500
        numFeatures = 10

        self.X = numpy.random.rand(numExamples, numFeatures)
        self.y = numpy.array(numpy.sign(numpy.random.rand(numExamples)-0.5), numpy.int)
        
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testInit(self):
        treeRank = TreeRankForest(self.leafRanklearner)

    #@unittest.skip("")
    def testLearnModel(self):
        maxDepth = 2
        treeRankForest = TreeRankForest(self.leafRanklearner)
        treeRankForest.setMaxDepth(maxDepth)
        treeRankForest.learnModel(self.X, self.y)

        forest = treeRankForest.getForest()

        self.assertEquals(len(forest), treeRankForest.getNumTrees())

        for treeRank in forest:
            tree = treeRank.getTree()
            self.assertTrue(tree.depth() <= maxDepth)

    #@unittest.skip("")
    def testPredict(self):
        maxDepth = 2
        treeRankForest = TreeRankForest(self.leafRanklearner)
        treeRankForest.setMaxDepth(maxDepth)
        treeRankForest.learnModel(self.X, self.y)

        scores = treeRankForest.predict(self.X)
        scores2 = numpy.zeros(self.X.shape[0])
        forest = treeRankForest.getForest()

        for i in range(len(forest)):
            scores2 += forest[i].predict(self.X)

        scores2 /= treeRankForest.getNumTrees()

        self.assertTrue((scores==scores2).all())

    def testPredict2(self):
        #Test on Gauss2D dataset
        dataDir = PathDefaults.getDataDir()

        fileName = dataDir + "Gauss2D_learn.csv"
        XY = numpy.loadtxt(fileName, skiprows=1, usecols=(1,2,3), delimiter=",")
        X = XY[:, 0:2]
        y = XY[:, 2]
        
        y = y*2 - 1 

        fileName = dataDir + "Gauss2D_test.csv"
        testXY = numpy.loadtxt(fileName, skiprows=1, usecols=(1,2,3), delimiter=",")
        testX = testXY[:, 0:2]
        testY = testXY[:, 2]
        
        testY = testY*2-1

        X = Standardiser().standardiseArray(X)
        testX = Standardiser().standardiseArray(testX)

        numTrees = 5
        minSplit = 50 
        maxDepths = range(3, 10)
        trainAucs = numpy.array([0.7252582, 0.7323278, 0.7350289, 0.7372529, 0.7399985, 0.7382176, 0.7395104, 0.7386347])
        testAucs = numpy.array([0.6806122, 0.6851614, 0.6886183, 0.6904147, 0.6897266, 0.6874600, 0.6875980, 0.6878801])

        i = 0
        
        #The results are approximately the same, but not exactly 
        for maxDepth in maxDepths:
            treeRankForest = TreeRankForest(self.leafRanklearner)
            treeRankForest.setMaxDepth(maxDepth)
            treeRankForest.setMinSplit(minSplit)
            treeRankForest.setNumTrees(numTrees)
            treeRankForest.learnModel(X, y)
            trainScores = treeRankForest.predict(X)
            testScores = treeRankForest.predict(testX)

            print(Evaluator.auc(trainScores, y), Evaluator.auc(testScores, testY))

            self.assertAlmostEquals(Evaluator.auc(trainScores, y), trainAucs[i], 1)
            self.assertAlmostEquals(Evaluator.auc(testScores, testY), testAucs[i], 1)
            i+=1

    #@unittest.skip("")
    def testEvaluateCvOuter(self):
        maxDepth = 10
        treeRankForest = TreeRankForest(self.leafRanklearner)
        treeRankForest.setMaxDepth(maxDepth)

        folds = 3
        (bestParams, allMetrics, bestMetaDicts) = treeRankForest.evaluateCvOuter(self.X, self.y, folds)

        #print(allMetrics)

    def testStr(self):
        treeRankForest = TreeRankForest(self.leafRanklearner)

if __name__ == '__main__':
    unittest.main()



import numpy
import unittest

from apgl.metabolomics.TreeRankForest import TreeRankForest
from apgl.metabolomics.leafrank.LinearSVM import LinearSVM
from apgl.metabolomics.leafrank.DecisionTree import DecisionTree
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Evaluator import Evaluator
from apgl.data.Standardiser import Standardiser

class TreeRankForestTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr(all="raise")

        C = 10.0
        self.generateleafRank = LinearSVM.generate(C)

        numExamples = 500
        numFeatures = 10

        self.X = numpy.random.rand(numExamples, numFeatures)
        self.y = numpy.array(numpy.sign(numpy.random.rand(numExamples)-0.5), numpy.int)

    def testInit(self):
        treeRank = TreeRankForest(self.generateleafRank)

    def testLearnModel(self):
        maxDepth = 2
        treeRankForest = TreeRankForest(self.generateleafRank)
        treeRankForest.setMaxDepth(maxDepth)
        treeRankForest.learnModel(self.X, self.y)

        forest = treeRankForest.getForest()

        self.assertEquals(len(forest), treeRankForest.getNumTrees())

        for treeRank in forest:
            tree = treeRank.getTree()
            self.assertTrue(tree.depth() <= maxDepth)

    def testPredict(self):
        maxDepth = 2
        treeRankForest = TreeRankForest(self.generateleafRank)
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

        fileName = dataDir + "Gauss2D_test.csv"
        testXY = numpy.loadtxt(fileName, skiprows=1, usecols=(1,2,3), delimiter=",")
        testX = testXY[:, 0:2]
        testY = testXY[:, 2]

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
            treeRankForest = TreeRankForest(self.generateleafRank)
            treeRankForest.setMaxDepth(maxDepth)
            treeRankForest.setMinSplit(minSplit)
            treeRankForest.setNumTrees(numTrees)
            treeRankForest.learnModel(X, y)
            trainScores = treeRankForest.predict(X)
            testScores = treeRankForest.predict(testX)

            #print(Evaluator.auc(trainScores, y), Evaluator.auc(testScores, testY))

            self.assertAlmostEquals(Evaluator.auc(trainScores, y), trainAucs[i], 1)
            self.assertAlmostEquals(Evaluator.auc(testScores, testY), testAucs[i], 1)
            i+=1

    def testEvaluateCvOuter(self):
        maxDepth = 10
        treeRankForest = TreeRankForest(self.generateleafRank)
        treeRankForest.setMaxDepth(maxDepth)

        folds = 3
        (bestParams, allMetrics, bestMetaDicts) = treeRankForest.evaluateCvOuter(self.X, self.y, folds)

        #print(allMetrics)

    def testStr(self):
        treeRankForest = TreeRankForest(self.generateleafRank)

if __name__ == '__main__':
    unittest.main()


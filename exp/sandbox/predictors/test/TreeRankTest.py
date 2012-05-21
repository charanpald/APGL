
import numpy 
import unittest

from exp.metabolomics.TreeRank import TreeRank
from exp.metabolomics.RankNode import RankNode
from exp.metabolomics.leafrank.LinearSVM import LinearSVM
from exp.metabolomics.leafrank.DecisionTree import DecisionTree
from apgl.util.PathDefaults import PathDefaults
from apgl.graph.DictTree import DictTree
from apgl.util.Evaluator import Evaluator
from apgl.data.Standardiser import Standardiser

class TreeRankTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr(all="raise")

        self.generateleafRank = LinearSVM

        numExamples = 200
        numFeatures = 10

        self.X = numpy.random.rand(numExamples, numFeatures)
        self.y = numpy.array(numpy.sign(numpy.random.rand(numExamples)-0.5), numpy.int)

    def testInit(self):
        treeRank = TreeRank(self.generateleafRank)

    def testLearnModel(self):
        maxDepth = 2
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        treeRank.learnModel(self.X, self.y)
        tree = treeRank.getTree()

        self.assertTrue(tree.depth() <= maxDepth)

    def testSplitNode(self):
        d = 0
        k = 0
        maxDepth = 1 
        inds = numpy.arange(self.y.shape[0])
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        
        node = RankNode(inds, numpy.arange(self.X.shape[1]))
        
        tree = DictTree()
        tree.setVertex((0, 0), node)
        tree = treeRank.splitNode(tree, self.X, self.y, d, k)

        self.assertEquals(tree.getNumVertices(), 3)
        self.assertEquals(tree.getNumEdges(), 2)
        self.assertEquals(tree.getRootId(), (0,0))
        self.assertTrue(not tree.getVertex((0, 0)).isLeafNode())
        self.assertTrue(tree.getVertex((1, 0)).isLeafNode())
        self.assertTrue(tree.getVertex((1, 1)).isLeafNode())

        self.assertTrue(tree.depth() <= maxDepth)

    def testClassifyNode(self):
        #Try on a single split tree
        maxDepth = 1
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)

        treeRank.learnModel(self.X, self.y)

        tree = treeRank.getTree()
        root = tree.getVertex(tree.getRootId())
        root.setTestInds(numpy.arange(self.X.shape[0]))
        treeRank.classifyNode(tree, self.X, 0, 0)

        self.assertTrue((tree.getVertex((0,0)).getTrainInds() == tree.getVertex((0,0)).getTestInds()).all())
        self.assertTrue((tree.getVertex((1,0)).getTrainInds() ==  tree.getVertex((1,0)).getTestInds()).all())
        self.assertTrue((tree.getVertex((1,1)).getTrainInds() ==  tree.getVertex((1,1)).getTestInds()).all())

    def testPredict(self):
        maxDepth = 2
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        treeRank.learnModel(self.X, self.y)

        scores = treeRank.predict(self.X)
        scores2 = numpy.zeros(self.X.shape[0])

        #Test if train and test indices are the same
        tree = treeRank.getTree()

        vertexIds = tree.getAllVertexIds()

        for vertexId in vertexIds:
            node = tree.getVertex(vertexId)
            self.assertTrue((node.getTrainInds() == node.getTestInds()).all())

            (d, k) = vertexId

            if node.isLeafNode():
                self.assertEquals(node.getScore(), (1 - float(k)/2**d)*2**maxDepth)
                scores2[node.getTestInds()] = node.getScore()

        self.assertTrue((scores==scores2).all())

    def testMaxDepth(self):
        maxDepth = 10 

        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        treeRank.learnModel(self.X, self.y)

        depth = treeRank.getTree().depth()
        self.assertTrue(depth <= maxDepth)

        maxDepth = 1
        treeRank.setMaxDepth(maxDepth)
        treeRank.learnModel(self.X, self.y)
        depth2 = treeRank.getTree().depth()

        self.assertTrue(depth2 <= maxDepth)
        self.assertTrue(depth > depth2)

    def testMinSplit(self):
        maxDepth = 10
        minSplit = 100
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        treeRank.setMinSplit(minSplit)
        treeRank.learnModel(self.X, self.y)

        tree = treeRank.getTree()

        vertexIds = tree.getAllVertexIds()
        for vertexId in vertexIds:
            node = tree.getVertex(vertexId)
            if not node.isLeafNode():
                self.assertTrue(node.getTrainInds().shape[0] >= minSplit)

    def testFeaturesSize(self):
        treeRank = TreeRank(self.generateleafRank)
        featureSize = 0.5
        treeRank.setFeatureSize(featureSize)
        treeRank.learnModel(self.X, self.y)
        scores = treeRank.predict(self.X)

        tree = treeRank.getTree()

        vertexIds = tree.getAllVertexIds()
        for vertexId in vertexIds:
            node = tree.getVertex(vertexId)

            #print(node.getFeatureInds())
            self.assertEquals(node.getFeatureInds().shape[0], numpy.round(featureSize*self.X.shape[1]))
            self.assertEquals(numpy.unique(node.getFeatureInds()).shape[0], node.getFeatureInds().shape[0])

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

        maxDepths = range(3, 10)
        trainAucs = numpy.array([0.7194734, 0.7284824, 0.7332185, 0.7348198, 0.7366152, 0.7367508, 0.7367508, 0.7367508])
        testAucs = numpy.array([0.6789078, 0.6844632, 0.6867918, 0.6873420, 0.6874820, 0.6874400, 0.6874400, 0.6874400])
        i = 0
        
        #The results are approximately the same, but not exactly 
        for maxDepth in maxDepths:
            treeRank = TreeRank(self.generateleafRank)
            treeRank.setMaxDepth(maxDepth)
            treeRank.learnModel(X, y)
            trainScores = treeRank.predict(X)
            testScores = treeRank.predict(testX)

            self.assertAlmostEquals(Evaluator.auc(trainScores, y), trainAucs[i], 2)
            self.assertAlmostEquals(Evaluator.auc(testScores, testY), testAucs[i], 1)
            i+=1 

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

        #X = Standardiser().standardiseArray(X)
        #testX = Standardiser().standardiseArray(testX)

        maxDepths = range(3, 10)
        trainAucs = numpy.array([0.7194734, 0.7284824, 0.7332185, 0.7348198, 0.7366152, 0.7367508, 0.7367508, 0.7367508])
        testAucs = numpy.array([0.6789078, 0.6844632, 0.6867918, 0.6873420, 0.6874820, 0.6874400, 0.6874400, 0.6874400])
        i = 0

        #The results are approximately the same, but not exactly
        for maxDepth in maxDepths:
            treeRank = TreeRank(DecisionTree)
            treeRank.setMaxDepth(maxDepth)
            treeRank.learnModel(X, y)
            trainScores = treeRank.predict(X)
            testScores = treeRank.predict(testX)

            #print(Evaluator.auc(trainScores, y), Evaluator.auc(testScores, testY))

            #self.assertAlmostEquals(Evaluator.auc(trainScores, y), trainAucs[i], 2)
            #self.assertAlmostEquals(Evaluator.auc(testScores, testY), testAucs[i], 1)
            i+=1

        #Compare tree to that of R version 
        tree = treeRank.getTree()

    def testEvaluateCvOuter(self):
        maxDepth = 10
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)

        folds = 3
        (bestParams, allMetrics, bestMetaDicts) = treeRank.evaluateCvOuter(self.X, self.y, folds)

        #print(allMetrics)

    def testCut(self):
        maxDepth = 10
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        treeRank.learnModel(self.X, self.y)

        tree = treeRank.getTree()
        depth = tree.depth()

        cutTree = TreeRank.cut(tree, depth-1)
        self.assertEquals(cutTree.depth(), depth-1)

        #Check if leaves are marked correctly
        for leaf in cutTree.leaves():
            self.assertTrue(cutTree.getVertex(leaf).isLeafNode())

        cutTree = TreeRank.cut(tree, 1)

        for leaf in cutTree.leaves():
            self.assertTrue(cutTree.getVertex(leaf).isLeafNode())

    def testLearnModelCut(self):
        maxDepth = 5
        minSplit = 10 
        treeRank = TreeRank(self.generateleafRank)
        treeRank.setMaxDepth(maxDepth)
        treeRank.setMinSplit(minSplit)
        treeRank.learnModelCut(self.X, self.y)
        tree = treeRank.getTree()
        
        self.assertTrue(tree.depth() <= maxDepth)

        #Checked inside the code 

if __name__ == '__main__':
    unittest.main()


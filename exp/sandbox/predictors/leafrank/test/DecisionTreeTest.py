
import unittest
import numpy
import logging
from apgl.metabolomics.leafrank.DecisionTree import DecisionTree
import orngTree
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Evaluator import Evaluator

class  DecisionTreeTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numExamples = 200
        numFeatures = 5

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

    def testLearnModel(self):
        decisionTree = DecisionTree()
        decisionTree.learnModel(self.X, self.y)

        tree = decisionTree.getClassifier()

    def testPredict(self):
        decisionTree = DecisionTree()
        decisionTree.learnModel(self.X, self.y)
        predY = decisionTree.predict(self.X)

        inds = numpy.random.permutation(self.X.shape[0])
        predY2 = decisionTree.predict(self.X[inds, :])

        self.assertTrue((predY[inds] == predY2).all())

        #Let's test on -1, +1 labels
        y2 = (self.y*2)-1
        decisionTree.learnModel(self.X, y2)
        predY2 = decisionTree.predict(self.X)

        self.assertTrue((predY2 == predY*2-1).all())

    def testSetWeight(self):
        decisionTree = DecisionTree()
        decisionTree.setWeight(1.0)
        decisionTree.learnModel(self.X, self.y)

        predY = decisionTree.predict(self.X)
        self.assertTrue((predY == numpy.ones(self.y.shape[0])).all())

        decisionTree.setWeight(0.0)
        decisionTree.learnModel(self.X, self.y)

        predY = decisionTree.predict(self.X)
        self.assertTrue((predY == numpy.zeros(self.y.shape[0])).all())

    def testMinSplit(self):
        decisionTree = DecisionTree()
        decisionTree.setMinSplit(20)
        decisionTree.learnModel(self.X, self.y)

        size = orngTree.countNodes(decisionTree.getClassifier())
        #orngTree.printTree(decisionTree.getClassifier())

        decisionTree.setMinSplit(0)
        decisionTree.learnModel(self.X, self.y)
        size2 = orngTree.countNodes(decisionTree.getClassifier())
        #orngTree.printTree(decisionTree.getClassifier())

        self.assertTrue(size < size2)


    def testGenerate(self):
        generate = DecisionTree.generate(5, 50)

        learner = generate()
        learner.learnModel(self.X, self.y)

        self.assertEquals(learner.getMaxDepth(), 5)
        self.assertEquals(learner.getMinSplit(), 50)

    def testSetWeight(self):
        #Try weight = 0 and weight = 1
        decisionTree = DecisionTree()
        decisionTree.setWeight(0.0)
        decisionTree.learnModel(self.X, self.y)

        predY = decisionTree.predict(self.X)
        self.assertTrue((predY == numpy.zeros(predY.shape[0])).all())

        decisionTree.setWeight(1.0)
        decisionTree.learnModel(self.X, self.y)
        predY = decisionTree.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

    def testSetM(self):
        decisionTree = DecisionTree()
        decisionTree.setMinSplit(5)

        folds = 3
        meanError, varError = decisionTree.evaluateCv(self.X, self.y, folds)
        
        decisionTree.setM(100)
        #decisionTree.setMinSplit(20)
        meanError2, varError = decisionTree.evaluateCv(self.X, self.y, folds)
        self.assertTrue(meanError != meanError2)

    def testPredict2(self):
        #We play around with parameters to maximise AUC on the IGF1_0-Haar data
        dataDir = PathDefaults.getDataDir()
        fileName = dataDir + "IGF1_0-Haar.npy"

        XY = numpy.load(fileName)
        X = XY[:, 0:XY.shape[1]-1]
        y = XY[:, XY.shape[1]-1].ravel()

        weight = numpy.bincount(numpy.array(y, numpy.int))[0]/float(y.shape[0])
        #weight = 0.5
        #weight = 0.9

        folds = 3
        decisionTree = DecisionTree()
        decisionTree.setWeight(weight)
        decisionTree.setMaxDepth(50)
        #decisionTree.setMinSplit(100)
        decisionTree.setM(50)
        mean, var = decisionTree.evaluateCv(X, y, folds, Evaluator.auc)
        logging.debug("AUC = " + str(mean))
        logging.debug("Var = " + str(var))


    def testSetMaxDepth(self):
        maxDepth = 20
        decisionTree = DecisionTree()
        decisionTree.setMaxDepth(maxDepth)
        decisionTree.learnModel(self.X, self.y)

        self.assertTrue(DecisionTree.depth(decisionTree.getClassifier().tree) <= maxDepth+1)

        maxDepth = 5
        decisionTree = DecisionTree()
        decisionTree.setMaxDepth(maxDepth)
        decisionTree.learnModel(self.X, self.y)

        self.assertTrue(DecisionTree.depth(decisionTree.getClassifier().tree) <= maxDepth+1)

if __name__ == '__main__':
    unittest.main()


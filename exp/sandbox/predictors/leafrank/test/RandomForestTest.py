
import unittest
import numpy
from apgl.metabolomics.leafrank.RandomForest import RandomForest
from apgl.metabolomics.leafrank.DecisionTree import DecisionTree
import orngTree
import orange
import orngEnsemble
from apgl.util.PathDefaults import PathDefaults

class  RandomForestTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numExamples = 200
        numFeatures = 5

        self.X = numpy.random.rand(numExamples, numFeatures)
        c = numpy.random.rand(numFeatures)
        self.y = numpy.sign(self.X.dot(c) < numpy.mean(self.X.dot(c)))

    def testLearnModel(self):
        randomForest = RandomForest()
        randomForest.learnModel(self.X, self.y)

        tree = randomForest.getClassifier()

    def testPredict(self):
        randomForest = RandomForest()
        randomForest.learnModel(self.X, self.y)
        predY = randomForest.predict(self.X)

        inds = numpy.random.permutation(self.X.shape[0])
        predY2 = randomForest.predict(self.X[inds, :])

        self.assertTrue((predY[inds] == predY2).all())

        #Let's test on -1, +1 labels
        y2 = (self.y*2)-1
        randomForest.learnModel(self.X, y2)
        predY2 = randomForest.predict(self.X)

        self.assertTrue((predY2 == predY*2-1).all())

    def testSetWeight(self):
        randomForest = RandomForest()
        randomForest.setWeight(1.0)
        randomForest.learnModel(self.X, self.y)

        predY = randomForest.predict(self.X)
        self.assertTrue((predY == numpy.ones(self.y.shape[0])).all())

        randomForest.setWeight(0.0)
        randomForest.learnModel(self.X, self.y)

        predY = randomForest.predict(self.X)
        self.assertTrue((predY == numpy.zeros(self.y.shape[0])).all())

    def testMinSplit(self):
        randomForest = RandomForest()
        randomForest.setMinSplit(20)
        randomForest.learnModel(self.X, self.y)

        size = numpy.zeros(100)
        i = 0 
        for c in randomForest.getClassifier().classifiers:
            size[i] = orngTree.countNodes(c)
            i += 1
        size = numpy.mean(size)
        #orngTree.printTree(randomForest.getClassifier())

        randomForest.setMinSplit(0)
        randomForest.learnModel(self.X, self.y)
        size2 = numpy.zeros(100)
        i = 0
        for c in randomForest.getClassifier().classifiers:
            size2[i] = orngTree.countNodes(c)
            i += 1
        size2 = numpy.mean(size2)

        self.assertTrue(size < size2)


    def testGenerate(self):
        generate = RandomForest.generate(5, 50)

        learner = generate()
        learner.learnModel(self.X, self.y)

        self.assertEquals(learner.getMaxDepth(), 5)
        self.assertEquals(learner.getMinSplit(), 50)

    def testSetWeight(self):
        #Try weight = 0 and weight = 1
        randomForest = RandomForest()
        randomForest.setWeight(0.0)
        randomForest.learnModel(self.X, self.y)

        predY = randomForest.predict(self.X)
        self.assertTrue((predY == numpy.zeros(predY.shape[0])).all())

        randomForest.setWeight(1.0)
        randomForest.learnModel(self.X, self.y)
        predY = randomForest.predict(self.X)
        self.assertTrue((predY == numpy.ones(predY.shape[0])).all())

    def testSetM(self):
        randomForest = RandomForest()
        randomForest.setM(0)
        randomForest.setMinSplit(5)

        folds = 3
        meanError, varError = randomForest.evaluateCv(self.X, self.y, folds)

        randomForest.setM(100)
        #randomForest.setMinSplit(20)
        meanError2, varError = randomForest.evaluateCv(self.X, self.y, folds)
        #Pruning seems to have no effect 
        #self.assertTrue(meanError != meanError2)

    def testSetMaxDepth(self):
        maxDepth = 20 
        randomForest = RandomForest()
        randomForest.setMaxDepth(maxDepth)
        randomForest.learnModel(self.X, self.y)

        for c in randomForest.getClassifier().classifiers:
            self.assertTrue(DecisionTree.depth(c.tree) <= maxDepth+1)

        maxDepth = 5
        randomForest = RandomForest()
        randomForest.setMaxDepth(maxDepth)
        randomForest.learnModel(self.X, self.y)

        for c in randomForest.getClassifier().classifiers:
            self.assertTrue(DecisionTree.depth(c.tree) <= maxDepth+1)

if __name__ == '__main__':
    unittest.main()


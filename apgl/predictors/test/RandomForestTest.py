import apgl
import unittest
import numpy
import logging
from apgl.predictors.RandomForest import RandomForest
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Evaluator import Evaluator
from apgl.util.Sampling import Sampling

@apgl.skipIf(not apgl.checkImport('sklearn'), 'Module sklearn is required')
class  RandomForestTest(unittest.TestCase):
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



    def testGenerate(self):
        generate = RandomForest.generate(5, 50)

        learner = generate()
        learner.learnModel(self.X, self.y)

        self.assertEquals(learner.getMaxDepth(), 5)
        self.assertEquals(learner.getMinSplit(), 50)

    @unittest.skip("")
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
        randomForest = RandomForest()
        randomForest.setWeight(weight)
        randomForest.setMaxDepth(50)
        #randomForest.setMinSplit(100)
        mean, var = randomForest.evaluateCv(X, y, folds, Evaluator.auc)
        logging.debug("AUC = " + str(mean))
        logging.debug("Var = " + str(var))


    def testSetMaxDepth(self):
        maxDepth = 20
        randomForest = RandomForest()
        randomForest.setMaxDepth(maxDepth)
        randomForest.learnModel(self.X, self.y)

        #self.assertTrue(RandomForest.depth(randomForest.getClassifier().tree) <= maxDepth+1)

        maxDepth = 5
        randomForest = RandomForest()
        randomForest.setMaxDepth(maxDepth)
        randomForest.learnModel(self.X, self.y)

        #self.assertTrue(RandomForest.depth(randomForest.getClassifier().tree) <= maxDepth+1)


        
    def testParallelPenaltyGrid(self): 
        folds = 3
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        randomForest = RandomForest()
        
        trainX = self.X[0:40, :]
        trainY = self.y[0:40]
        
        paramDict = {} 
        paramDict["setMinSplit"] = randomForest.getMinSplits()
        paramDict["setMaxDepth"] = randomForest.getMaxDepths()      

        idealPenalties = randomForest.parallelPenaltyGrid(trainX, trainY, self.X, self.y, paramDict)

if __name__ == '__main__':
    unittest.main()


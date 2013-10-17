


import numpy 
import unittest
import apgl
from apgl.util.Evaluator import Evaluator 


class  EvaluatorTestCase(unittest.TestCase):
    #def setUp(self):
    #    self.foo = Evaluator()
    #


    def testRootMeanSqError(self):
        y = numpy.array([1,2,3])
        predY = numpy.array([1,2,3])

        self.assertEquals(Evaluator.rootMeanSqError(y, predY), 0.0)

        y = numpy.array([1,2,3])
        predY = numpy.array([1,2,2])

        self.assertEquals(Evaluator.rootMeanSqError(y, predY), float(1)/numpy.sqrt(3))

        predY = numpy.array([1,2])
        self.assertRaises(ValueError, Evaluator.rootMeanSqError, y, predY)

    def testWeightedRootMeanSqError(self):

        y = numpy.array([0.1, 0.2, 0.3])
        predY = numpy.array([0.1, 0.2, 0.3])

        self.assertEquals(Evaluator.weightedRootMeanSqError(y, predY), 0.0)

        #Errors on larger ys are weighted more 
        predY = numpy.array([0.0, 0.2, 0.3])
        predY2 = numpy.array([0.1, 0.2, 0.4])

        self.assertTrue(Evaluator.weightedRootMeanSqError(y, predY) < Evaluator.weightedRootMeanSqError(y, predY2))

    def testBinaryError(self):
        testY = numpy.array([1, 1, -1, 1])
        predY = numpy.array([-1, 1, -1, 1])
        predY2 = numpy.array([-1, -1, -1, 1])
        predY3 = numpy.array([-1, -1, 1, -1])

        self.assertTrue(Evaluator.binaryError(testY, predY) == 0.25)
        self.assertTrue(Evaluator.binaryError(testY, testY) == 0.0)
        self.assertTrue(Evaluator.binaryError(predY, predY) == 0.0)

        self.assertTrue(Evaluator.binaryError(testY, predY2) == 0.5)
        self.assertTrue(Evaluator.binaryError(testY, predY3) == 1.0)

    def testBinaryErrorP(self):
        testY = numpy.array([1, 1, -1, 1])
        predY = numpy.array([-1, 1, -1, 1])
        predY2 = numpy.array([-1, -1, -1, 1])
        predY3 = numpy.array([-1, -1, 1, -1])

        self.assertTrue(Evaluator.binaryErrorP(testY, predY) == 1.0/3.0)
        self.assertTrue(Evaluator.binaryErrorP(testY, testY) == 0.0)
        self.assertTrue(Evaluator.binaryErrorP(predY, predY) == 0.0)

        self.assertTrue(Evaluator.binaryErrorP(testY, predY2) == 2.0/3.0)
        self.assertTrue(Evaluator.binaryErrorP(testY, predY3) == 1.0)

        testY = numpy.array([-1, -1, -1, -1])
        predY = numpy.array([-1, 1, -1, 1])

        self.assertTrue(Evaluator.binaryErrorP(testY, predY) == 0.0)

    def testBinaryErrorN(self):
        testY = numpy.array([1, 1, -1, 1])
        predY = numpy.array([-1, 1, -1, 1])
        predY2 = numpy.array([-1, -1, -1, 1])
        predY3 = numpy.array([-1, -1, 1, -1])

        self.assertTrue(Evaluator.binaryErrorN(testY, predY) == 0.0)
        self.assertTrue(Evaluator.binaryErrorN(testY, testY) == 0.0)
        self.assertTrue(Evaluator.binaryErrorN(predY, predY) == 0.0)

        self.assertTrue(Evaluator.binaryErrorN(testY, predY2) == 0.0)
        self.assertTrue(Evaluator.binaryErrorN(testY, predY3) == 1.0)

        testY = numpy.array([-1, -1, -1, -1])
        predY = numpy.array([-1, 1, -1, 1])

        self.assertTrue(Evaluator.binaryErrorN(testY, predY) == 0.5)

        testY = numpy.array([1, 1, 1, 1])
        predY = numpy.array([-1, 1, -1, 1])

        self.assertTrue(Evaluator.binaryErrorN(testY, predY) == 0.0)

    @apgl.skipIf(not apgl.checkImport('sklearn'), 'No module scikits.learn')
    def testAuc(self):
        testY = numpy.array([-1, -1, 1, 1])
        predY = numpy.array([-1, 0, 1, 1])
        predY2 = numpy.array([0.1, 0.2, 0.3, 0.4])

        self.assertEquals(Evaluator.auc(predY, testY), 1.0)
        self.assertEquals(Evaluator.auc(predY2, testY), 1.0)
        self.assertEquals(Evaluator.auc(-predY, testY), 0.0)

        numExamples = 1000
        testY = numpy.array(numpy.random.rand(numExamples)>0.5, numpy.int)
        predY = numpy.random.rand(numExamples)>0.5

        #For a random score the AUC is approximately 0.5 
        self.assertAlmostEquals(Evaluator.auc(predY, testY), 0.5, 1)

    @apgl.skipIf(not apgl.checkImport('sklearn'), 'No module scikits.learn')
    def testLocalAuc(self):
        testY = numpy.array([-1, -1, 1, 1, 1, 1, 1, -1, -1, 1])
        predY = numpy.array([0.987,  0.868,  0.512,  0.114,  0.755,  0.976,  0.05,  0.371, 0.629,  0.819])

        self.assertEquals(Evaluator.localAuc(testY, predY, 1.0), Evaluator.auc(predY, testY))
        self.assertEquals(Evaluator.localAuc(testY, predY, 0.0), 0)

        self.assertEquals(Evaluator.localAuc(testY, testY, 0.2), 1.0)

        #Ask stephan if correct - needs extra tests 

    def testBinaryBootstrapError(self):

        testY = numpy.array([-1, -1, 1, 1, 1])
        predY = 1 - testY

        trainY = numpy.array([-1, -1, 1, 1, 1])
        predTrainY = 1 - trainY

        self.assertEquals(Evaluator.binaryBootstrapError(testY, testY, trainY, trainY, 0.5), 0.0)

        self.assertEquals(Evaluator.binaryBootstrapError(testY, testY, trainY, predTrainY, 0.5), 0.5)
        self.assertEquals(Evaluator.binaryBootstrapError(testY, testY, trainY, predTrainY, 0.1), 0.9)

        self.assertEquals(Evaluator.binaryBootstrapError(testY, predY, trainY, trainY, 0.1), 0.1)
        
    def testMeanAbsError(self): 
        testY = numpy.array([1, 2, 1.5])
        predY = numpy.array([2, 1, 0.5]) 
        
        self.assertEquals(Evaluator.meanAbsError(testY, predY), 1.0)
        self.assertEquals(Evaluator.meanAbsError(testY, testY), 0.0)
        
        testY = numpy.random.rand(10)
        predY = numpy.random.rand(10)
        
        error = numpy.abs(testY - predY).mean()
        self.assertEquals(error, Evaluator.meanAbsError(testY, predY))
    
    @apgl.skipIf(not apgl.checkImport('sklearn'), 'No module scikits.learn')
    def testPrecisionFromIndLists(self): 
        predList  = [4, 2, 10]
        testList = [4, 2]

        self.assertEquals(Evaluator.precisionFromIndLists(testList, predList), 2.0/3)  
        
        testList = [4, 2, 10]
        self.assertEquals(Evaluator.precisionFromIndLists(testList, predList), 1) 
        
        predList  = [10, 2, 4]
        self.assertEquals(Evaluator.precisionFromIndLists(testList, predList), 1)
        
        testList = [1, 9, 11]
        self.assertEquals(Evaluator.precisionFromIndLists(testList, predList), 0)
        
        predList = [1, 2, 3, 4, 5]
        testList = [1, 9, 11]
        
        self.assertEquals(Evaluator.precisionFromIndLists(testList, predList), 1.0/5)
        
    def testAveragePrecisionFromLists(self): 
        predList  = [4, 2, 10]
        testList = [4, 2, 15, 16]
        
        self.assertEquals(Evaluator.averagePrecisionFromLists(testList, predList), 0.5)
        
        predList = [0,1,2,3,4,5]
        testList = [0, 3, 4, 5]
        self.assertAlmostEquals(Evaluator.averagePrecisionFromLists(testList, predList), 0.691666666666)
        
if __name__ == '__main__':
    unittest.main()


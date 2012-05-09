import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2, findBestSplitRand, findBestSplitRisk
from apgl.data.ExamplesGenerator import ExamplesGenerator  

class TreeCriterionPyTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr("raise")

    def testFindBestSplit(self): 
        minSplit = 1 
        
        X = numpy.zeros((20, 10))
        y = numpy.ones(20)
        
        X[0:10, 2] = numpy.arange(10)
        X[10:, 2] = numpy.arange(10)+10 
        y[0:10] = -1 
        
        nodeInds = numpy.arange(X.shape[0])
        argsortX = numpy.zeros(X.shape, numpy.int)      
        
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y, nodeInds, argsortX)
        
        
        self.assertEquals(bestError, 0.0)
        self.assertEquals(bestFeatureInd, 2)
        self.assertEquals(bestThreshold, 9.5)
        
        self.assertTrue((bestLeftInds == numpy.arange(0, 10)).all())
        self.assertTrue((bestRightInds == numpy.arange(10, 20)).all())
        
        #Test case where all values are the same 
        X = numpy.zeros((20, 10))
        
        argsortX = numpy.zeros(X.shape, numpy.int)      
        
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
         
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y, nodeInds, argsortX)
        self.assertTrue(bestLeftInds.shape[0]==0)
        self.assertTrue(bestRightInds.shape[0]==X.shape[0])
        
        #Another simple example 
        X = numpy.random.rand(20, 1)
        y = numpy.random.rand(20)

        inds = [1, 3, 7, 12, 14, 15]
        X[inds, 0] += 10 
        y[inds] += 1 
        
        argsortX = numpy.zeros(X.shape, numpy.int)      
        
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y, nodeInds, argsortX)
        nptst.assert_array_equal(bestRightInds, numpy.array(inds))
        
        #Test minSplit 
        minSplit = 10
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y, nodeInds, argsortX)
        self.assertTrue(bestLeftInds.shape[0] >= minSplit)
        self.assertTrue(bestRightInds.shape[0] >= minSplit)
        
        #Vary nodeInds 
        minSplit = 1 
        nodeInds = numpy.arange(16)
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y, nodeInds, argsortX)
        nptst.assert_array_equal(bestRightInds, numpy.array(inds))
        nptst.assert_array_equal(bestLeftInds, numpy.setdiff1d(nodeInds, numpy.array(inds))) 
        
        nodeInds = numpy.arange(10)
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y, nodeInds, argsortX)
        nptst.assert_array_equal(bestRightInds, numpy.array([1,3,7]))
        nptst.assert_array_equal(bestLeftInds, numpy.setdiff1d(nodeInds, numpy.array([1,3,7]))) 
    
    @unittest.skip("")
    def testFindBestSplit2(self): 
        minSplit = 1 
        X = numpy.zeros((20, 10))
        y = numpy.ones(20)
        
        X[0:10, 2] = numpy.arange(10)
        X[10:, 2] = numpy.arange(10)+10 
        y[0:10] = -1 
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y)
        
        
        self.assertEquals(bestError, 0.0)
        self.assertEquals(bestFeatureInd, 2)
        self.assertEquals(bestThreshold, 9.5)
        
        self.assertTrue((bestLeftInds == numpy.arange(0, 10)).all())
        self.assertTrue((bestRightInds == numpy.arange(10, 20)).all())
        
        #Test case where all values are the same 
        X = numpy.zeros((20, 10))
         
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y)
        self.assertTrue(bestRightInds.shape[0]==0)
        
        #Another simple example 
        X = numpy.random.rand(20, 1)
        y = numpy.random.rand(20)

        inds = [1, 3, 7, 12, 14, 15]
        X[inds, 0] += 10 
        y[inds] += 1   
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit2(minSplit, X, y)
        
        for i in range(10): 
            numExamples = numpy.random.randint(1, 200)
            numFeatures = numpy.random.randint(1, 10)
            
            X = numpy.random.rand(numExamples, numFeatures)
            y = numpy.random.rand(numExamples)
            
            bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit(minSplit, X, y)
            bestError2, bestFeatureInd2, bestThreshold2, bestLeftInds2, bestRightInds2 = findBestSplit2(minSplit, X, y)
            
            self.assertEquals(bestFeatureInd, bestFeatureInd2)
            self.assertAlmostEquals(bestThreshold, bestThreshold2)
            nptst.assert_array_equal(bestLeftInds, bestLeftInds2)
            nptst.assert_array_equal(bestRightInds, bestRightInds2)      
            
    def testFindBestSplitRand(self): 
        minSplit = 1 
        numExamples = 20 
        numFeatures = 10 
        X = numpy.zeros((numExamples, numFeatures))
        y = numpy.ones(numExamples, numpy.int)
        
        X[0:10, 2] = numpy.arange(10)
        X[10:, 2] = numpy.arange(10)+10 
        y[0:10] = -1 
        
        y += 1 
        
        nodeInds = numpy.arange(X.shape[0])
        argsortX = numpy.zeros(X.shape, numpy.int)      
        
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])        
        
        errors, thresholds = findBestSplitRand(minSplit, X, y, nodeInds, argsortX) 
        
        #print(errors, thresholds)
        
    def testFindBestSplitRisk(self): 
        minSplit = 1 
        numExamples = 20 
        numFeatures = 10 
        X = numpy.zeros((numExamples, numFeatures))
        y = numpy.ones(numExamples, numpy.int)
        
        X[0:10, 2] = numpy.arange(10)
        X[10:, 2] = numpy.arange(10)+10 
        y[0:10] = -1 
        
        y += 1 
        
        nodeInds = numpy.arange(X.shape[0])
        argsortX = numpy.zeros(X.shape, numpy.int)      
        
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])        
        
        errors, thresholds = findBestSplitRisk(minSplit, X, y, nodeInds, argsortX) 
        print(errors, thresholds)
        
        X = numpy.random.rand(numExamples, numFeatures)
        errors, thresholds = findBestSplitRisk(minSplit, X, y, nodeInds, argsortX) 
        print(errors, thresholds)
        
        
if __name__ == "__main__":
    unittest.main()
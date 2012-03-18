import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.TreeCriterion import findBestSplit, findBestSplit2
from apgl.data.ExamplesGenerator import ExamplesGenerator  

class TreeCriterionTest(unittest.TestCase):
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
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit(minSplit, X, y)
        
        
        self.assertEquals(bestError, 0.0)
        self.assertEquals(bestFeatureInd, 2)
        self.assertEquals(bestThreshold, 9.5)
        
        self.assertTrue((bestLeftInds == numpy.arange(0, 10)).all())
        self.assertTrue((bestRightInds == numpy.arange(10, 20)).all())
        
        #Test case where all values are the same 
        X = numpy.zeros((20, 10))
         
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit(minSplit, X, y)
        self.assertTrue(bestLeftInds.shape[0]==0)
        self.assertTrue(bestRightInds.shape[0]==0)
        
        #Another simple example 
        X = numpy.random.rand(20, 1)
        y = numpy.random.rand(20)

        inds = [1, 3, 7, 12, 14, 15]
        X[inds, 0] += 10 
        y[inds] += 1   
        
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit(minSplit, X, y)
        nptst.assert_array_equal(bestRightInds, numpy.array(inds))
        
        #Test minSplit 
        minSplit = 10
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit(minSplit, X, y)
        self.assertTrue(bestLeftInds.shape[0] >= minSplit)
        self.assertTrue(bestRightInds.shape[0] >= minSplit)
        
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
            
if __name__ == "__main__":
    unittest.main()
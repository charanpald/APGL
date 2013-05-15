import logging
import unittest
import numpy
import scipy.sparse 
from exp.util.MCEvaluator import MCEvaluator

class  MCEvaluatorTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(21)
    
    def testMeanSqError(self): 
        numExamples = 10
        testX = scipy.sparse.rand(numExamples, numExamples)
        testX = testX.tocsr()
        
        predX = testX.copy() 
        error = MCEvaluator.meanSqError(testX, predX)
        self.assertEquals(error, 0.0)
        
        testX = numpy.random.rand(numExamples, numExamples)
        predX = testX + numpy.random.rand(numExamples, numExamples)*0.5 
        
        error2 = ((testX-predX)**2).sum()/(numExamples**2)
        error = MCEvaluator.meanSqError(scipy.sparse.csr_matrix(testX), scipy.sparse.csr_matrix(predX)) 
        
        self.assertEquals(error, error2)
        
if __name__ == '__main__':
    unittest.main()


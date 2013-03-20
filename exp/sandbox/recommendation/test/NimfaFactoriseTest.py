
import sys
from apgl.util.Util import Util
from apgl.util.Sampling import Sampling 
from exp.sandbox.recommendation.NimfaFactorise import NimfaFactorise 
from apgl.util.MCEvaluator import MCEvaluator 
import numpy
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 

class SoftImputeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        numpy.set_printoptions(precision=3, suppress=True)
        
        numpy.seterr(all="raise")

    
    def testLearnModel(self): 
        numpy.random.seed(21)
        X = scipy.sparse.rand(10, 10, 0.5)
        X = X.tocsr()
                
        method = "lsnmf"
        
        nimfaFactorise = NimfaFactorise(method, maxIter=50)
        predX = nimfaFactorise.learnModel(X)
        
        self.assertEquals(predX.shape, X.shape)
        
        #Test the case where we specify many ranks 
        ranks = numpy.array([10, 8, 5, 2])
        nimfaFactorise = NimfaFactorise(method, ranks)
        predXList = nimfaFactorise.learnModel(X)
        
        #Let's look at the errors 
        for predX in predXList: 
            error = MCEvaluator.meanSqError(X, predX)
            print(error)
            

    def testParallelModelSelect(self): 
        X = scipy.sparse.rand(10, 10, 0.5)
        X = X.tocsr()
          
        numExamples = X.getnnz()
        paramDict = {}
        paramDict["setRank"] = numpy.array([5, 10, 20])
        folds = 3 
        idx = Sampling.randCrossValidation(folds, numExamples)
                
        
        method = "lsnmf"
        nimfaFactorise = NimfaFactorise(method)
        learner, meanErrors = nimfaFactorise.parallelModelSelect(X, idx, paramDict)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
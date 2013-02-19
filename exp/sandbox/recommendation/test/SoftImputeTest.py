
import sys
from apgl.util.Util import Util
from exp.sandbox.recommendation.SoftImpute import SoftImpute 
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

    def testSvdSoft(self): 
        A = scipy.sparse.rand(10, 10, 0.2)
        
        lmbda = 0.1
        k = 6
        U, s, V = SoftImpute.svdSoft(A, lmbda, k)
        ATilde = U.dot(numpy.diag(s)).dot(V.T)        
        
        #Now comput the same matrix using numpy
        #Pick first k singular vectors/values 
        A = A.todense() 
        
        U2, s2, V2 = numpy.linalg.svd(A)
        inds = numpy.flipud(numpy.argsort(s2))[0:k]
        U2, s2, V2 = Util.indSvd(U2, s2, V2, inds)        
        
        s2 = s2 - lmbda 
        s2 = numpy.clip(s, 0, numpy.max(s2))
        

        ATilde2 = U2.dot(numpy.diag(s2)).dot(V2.T)
        
        nptst.assert_array_almost_equal(s, s)
        nptst.assert_array_almost_equal(ATilde, ATilde2)
    
    def testLearnModel(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tocsr()
                
        lmbdas = numpy.array([100, 0.0])
        eps = 0.001         
        k = 8
        
        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel(X)
        
        nptst.assert_array_almost_equal(numpy.zeros(X.shape), ZList[0].todense())
        nptst.assert_array_almost_equal(X.todense(), ZList[1].todense(), 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
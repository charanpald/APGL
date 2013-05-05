
import sys
from apgl.util.Util import Util
from apgl.util.Sampling import Sampling 
from exp.sandbox.recommendation.SoftImpute import SoftImpute 
import numpy
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 
import exp.util.SparseUtils as ExpSU
from sparsesvd import sparsesvd

class SoftImputeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=100)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)
        
    def testLearnModel(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tocsc()        
        lmbdas = numpy.array([10.0, 0.0])
        eps = 0.01         
        k = 9
        
        #Check out singular values 
        U, s, V = sparsesvd(X.tocsc(), k) 

        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel(X)
        
        #Test that when lambda=0 get approx original matrix back 
        X2 = ZList[1].todense()
        nptst.assert_almost_equal(X.todense(), X2)
        
        #When lambda is greater or equal to largest singular value, get 0 
        U, s, V = sparsesvd(X.tocsc(), k) 
        lmbdas = numpy.array([numpy.max(s)]) 
        softImpute = SoftImpute(lmbdas, eps, k)
        Z = softImpute.learnModel(X)
        self.assertEquals(numpy.linalg.norm(Z.todense()), 0)
        
        #Check solution for medium values of lambda 
        eps = 0.1
        lmbdas = numpy.array([0.1, 0.2, 0.5, 1.0])
        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel(X)
        
        for j, Z in enumerate(ZList): 
            Z = Z.todense()
            Zomega = numpy.zeros(X.shape)
            
            rowInds, colInds = X.nonzero()
            for i in range(X.nonzero()[0].shape[0]): 
                Zomega[rowInds[i], colInds[i]] = Z[rowInds[i], colInds[i]]
                
            U, s, V = ExpSU.SparseUtils.svdSoft(numpy.array(X-Zomega+Z), lmbdas[j])      
            
            tol = 0.1
            self.assertTrue(numpy.linalg.norm(Z -(U*s).dot(V.T))**2 < tol)
        
    def testLearnModel2(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tocsc()        
        lmbdas = numpy.array([10.0, 0.0])
        eps = 0.01         
        k = 9
        
        #Check out singular values 
        U, s, V = sparsesvd(X.tocsc(), k) 

        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel2(X)
        
        #Test that when lambda=0 get approx original matrix back 
        X2 = ZList[1].todense()
        nptst.assert_almost_equal(X.todense(), X2)
        
        #When lambda is greater or equal to largest singular value, get 0 
        U, s, V = sparsesvd(X.tocsc(), k) 
        lmbdas = numpy.array([numpy.max(s)]) 
        softImpute = SoftImpute(lmbdas, eps, k)
        Z = softImpute.learnModel2(X)
        self.assertEquals(numpy.linalg.norm(Z.todense()), 0)
        
        #Check solution for medium values of lambda 
        eps = 0.1
        lmbdas = numpy.array([0.1, 0.2, 0.5, 1.0])
        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel2(X)
        
        for j, Z in enumerate(ZList): 
            Z = Z.todense()
            Zomega = numpy.zeros(X.shape)
            
            rowInds, colInds = X.nonzero()
            for i in range(X.nonzero()[0].shape[0]): 
                Zomega[rowInds[i], colInds[i]] = Z[rowInds[i], colInds[i]]
                
            U, s, V = ExpSU.SparseUtils.svdSoft(numpy.array(X-Zomega+Z), lmbdas[j])      
            
            tol = 0.1
            self.assertTrue(numpy.linalg.norm(Z -(U*s).dot(V.T))**2 < tol)

    def testPredict(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tocsc()        
        lmbdas = numpy.array([2.0, 1.5, 1.0, 0.5, 0.2, 0.1])
        eps = 0.001         
        k = 9
        
        #Check out singular values 
        U, s, V = sparsesvd(X.tocsc(), k) 
        
        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel(X, fullMatrices=False)
        
        inds = X.nonzero()
        
        predXList = softImpute.predict(ZList, inds)
        
        U, s, V = ZList[0]

        for predX in predXList: 
            nptst.assert_array_equal(predX.nonzero()[0], inds[0])
            nptst.assert_array_equal(predX.nonzero()[1], inds[1])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    

import sys
from apgl.util.Util import Util
from exp.sandbox.recommendation.SGDNorm2Reg import SGDNorm2Reg 
import numpy
import numpy.random
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 
from exp.sandbox.recommendation.SoftImpute import SoftImpute 

class SoftImputeTest(unittest.TestCase):
    def setUp(self):
#        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        numpy.set_printoptions(precision=3, suppress=True)
        self.n = 20
        self.k = 5
        lmbda = 0.01
        eps = 0.000001         
        tmax = 1000
        gamma = 1
        self.sigma = 0.1# perturbation of the initial decomposition
        
        # with a sparse matrix
        X = scipy.sparse.rand(self.n, self.n, 0.1)
        self.X = X.tocsr()        

        self.omega = X.nonzero()
        self.inds = numpy.ravel_multi_index(self.omega, (self.n,self.n))
                
        # difficulty of that matrix
        self.Xtest = X.todense()
        omega = X.nonzero()
        self.P, self.s, self.Qh = scipy.linalg.svd(X.todense())
        Xtruncated = (self.P[:,0:self.k]*self.s[0:self.k]).dot(self.Qh[0:self.k,:])
        XtruncatedTest = numpy.zeros(X.shape)
        for u,i in zip(omega[0], omega[1]):
            XtruncatedTest[u,i] = Xtruncated[u,i]
        self.simple_svd_error = numpy.linalg.norm(self.Xtest-XtruncatedTest)
        logging.info("simple svd error:" + str(self.simple_svd_error))

        self.algo = SGDNorm2Reg(self.k, lmbda, eps, tmax)
        self.algo.gamma = gamma

    def testPredict(self): 
        ZList = self.algo.learnModel(self.X)
        
        predX = self.algo.predict(ZList, self.inds)
        nptst.assert_array_equal(predX.nonzero()[0], self.omega[0])
        nptst.assert_array_equal(predX.nonzero()[1], self.omega[1])
              
    def testPredictAll(self): 
        ZList = self.algo.learnModel(self.X)
        
        predXList = self.algo.predictAll(ZList, self.inds)
        self.assertEquals(len(predXList), len(ZList))
        
        for predX in predXList: 
            nptst.assert_array_equal(predX.nonzero()[0], self.omega[0])
            nptst.assert_array_equal(predX.nonzero()[1], self.omega[1])

    def testLearnModel(self): 
        # reasonable starting point => almost fine
        ZList = self.algo.learnModel(self.X, self.P[:,0:self.k]*self.s[0:self.k], self.Qh.T[:,0:self.k])

        # noisy reasonable starting point => almost fine
        ZList = self.algo.learnModel(self.X
                                     , self.P[:,0:self.k]*self.s[0:self.k] + self.sigma*numpy.random.randn(self.n, self.k)
                                     , self.Qh.T[:,0:self.k]+ self.sigma*numpy.random.randn(self.n, self.k))

        # random starting point => mostly leads to NaNs
#        ZList = self.algo.learnModel(X)
        
        self.assertTrue(len(ZList)>1, msg="len(ZList)="+str(len(ZList)))

        predXList = self.algo.predictAll(ZList, self.inds)
        for predX in predXList:
            sgd_error = numpy.linalg.norm(self.Xtest-predX)
            logging.info("sgd error:" + str(sgd_error))
            self.assertGreater(self.simple_svd_error, sgd_error)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    

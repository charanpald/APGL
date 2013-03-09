
import sys
from apgl.util.Util import Util
from exp.sandbox.recommendation.SGDNorm2Reg import SGDNorm2Reg 
import numpy
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 
from exp.sandbox.recommendation.SoftImpute import SoftImpute 

class SoftImputeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        numpy.set_printoptions(precision=3, suppress=True)

    def testLearnModel(self): 
        n = 10
        k = 5
        lmbda = 0.001
        eps = 0.00000001         
        tmax = 10000
        gamma = 1
        
        # with a sparse matrix
        X = scipy.sparse.rand(n, n, 0.2)
        X = X.tocsr()        
                
        # difficulty of that matrix
        Xtest = X.todense()
        omega = X.nonzero()
        P, s, Qh = scipy.linalg.svd(X.todense())
        Xtruncated = (P[:,0:k]*s[0:k]).dot(Qh[0:k,:])
        XtruncatedTest = numpy.zeros(X.shape)
        for u,i in zip(omega[0], omega[1]):
            XtruncatedTest[u,i] = Xtruncated[u,i]
        simple_svd_error = numpy.linalg.norm(Xtest-XtruncatedTest)
        logging.info("simple svd error:" + str(simple_svd_error))


        algo = SGDNorm2Reg(k, lmbda, eps, tmax)
        algo.gamma = gamma

        # perfect starting point => no problem
#        softImpute = SoftImpute(numpy.array([100, 0.0]), eps, k)
#        ZList = softImpute.learnModel(X)
#        P, s, Qh = scipy.linalg.svd(ZList[-1].todense())
#        print(s)
#        print(X)
#        print(ZList[-1])
#        nptst.assert_array_almost_equal(X.todense(), (P*s).dot(Qh), 2)
#        nptst.assert_array_almost_equal(X.todense(), (P[:,0:k]*s[0:k]).dot(Qh[0:k,:]), 2)
#        ZList = algo.learnModel(X, P[:,0:k]*s[0:k], Qh.T[:,0:k])
        

        # reasonnable starting point => almost fine
        ZList = algo.learnModel(X, P[:,0:k]*s[0:k], Qh.T[:,0:k])

        # random starting point => mostly leads to NaNs
#        ZList = algo.learnModel(X)
        
        Ztest = numpy.zeros(ZList[-1].shape)
        for u,i in zip(omega[0], omega[1]):
            Ztest[u,i] = ZList[-1][u,i]
        # gain compare to simple svd
        sgd_error = numpy.linalg.norm(Xtest-Ztest)
        logging.info("sgd error:" + str(sgd_error))
        self.assertGreater(simple_svd_error, sgd_error)

               

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    

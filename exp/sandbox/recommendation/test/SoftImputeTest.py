
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

    #@unittest.skip("")
    def testLearnModel(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tocsc()        
        lmbdas = numpy.array([10.0, 0.0])
        eps = 0.01         
        k = 9
        
        #Check out singular values 
        U, s, V = sparsesvd(X.tocsc(), k) 
        #print(s)
        
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
        
        #Compare against learnModel2 against a moderate lambda value 
        lmbdas = numpy.array([0.2])
        softImpute = SoftImpute(lmbdas, eps, k)
        ZList = softImpute.learnModel(X)
        ZList2 = softImpute.learnModel2(X)
        
        #nptst.assert_almost_equal(ZList.todense(), ZList2.todense())

    def testLearnModel2(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tolil()        
        lmbdas = numpy.array([10.0, 0.0])
        eps = 0.001         
        k = 9
        
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

    @unittest.skip("")
    def testParallelModelSelect(self): 
        X = scipy.sparse.rand(10, 10, 0.5)
        X = X.tocsr()
          
        numExamples = X.getnnz()
        paramDict = {}
        paramDict["setK"] = numpy.array([5, 10, 20])
        folds = 3 
        idx = Sampling.randCrossValidation(folds, numExamples)
                
        
        lmbdas = numpy.array([0.1])
        softImpute = SoftImpute(lmbdas, k=10)
        learner, meanErrors = softImpute.parallelModelSelect(X, idx, paramDict)

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
    
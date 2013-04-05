
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
from sparsesvd import sparsesvd

class SoftImputeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=100)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)

    def testSvdSoft(self): 
        A = scipy.sparse.rand(10, 10, 0.2)
        A = A.tocsc()
        
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

    def testSvdSparseLowRank(self): 
        A = scipy.sparse.rand(10, 10, 0.2) 
        A = A.tocsc()
        
        B = numpy.random.rand(10, 10)
        U, s, V = numpy.linalg.svd(B)
        V = V.T         
        
        r = 3
        U = U[:, 0:r]
        s = s[0:r]
        V = V[:, 0:r]
        #B is low rank 
        B = (U*s).dot(V.T)
        
        U2, s2, V2 = SoftImpute.svdSparseLowRank(A, U, s, V)
        
        nptst.assert_array_almost_equal(U2.T.dot(U2), numpy.eye(U2.shape[1]))
        nptst.assert_array_almost_equal(V2.T.dot(V2), numpy.eye(V2.shape[1]))
        #self.assertEquals(s2.shape[0], r)
        
        A2 = (U2*s2).dot(V2.T)
        
        #Compute real SVD 
        C = numpy.array(A.todense()) + B
        U3, s3, V3 = numpy.linalg.svd(C)
        V3 = V3.T  

        A3 = (U3*s3).dot(V3.T)
        
        self.assertAlmostEquals(numpy.linalg.norm(A2 - A3), 0)
        

    #@unittest.skip("")
    def testLearnModel(self): 
        X = scipy.sparse.rand(10, 10, 0.2)
        X = X.tolil()        
        lmbdas = numpy.array([10.0, 0.0])
        eps = 0.001         
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
        ZList2 = softImpute.learnModel(X)
        
        nptst.assert_almost_equal(ZList.todense(), ZList2.todense())

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

    def testPartialReconstruct(self):
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        indices = numpy.nonzero(Y)
        
        X = SoftImpute.partialReconstruct(indices, U, s, V)
        X = X.todense()
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just someIndices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        
        X = SoftImpute.partialReconstruct(inds, U, s, V)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(X[j, k], Y[j, k])  
            
        self.assertTrue(X.nnz == inds[0].shape[0])
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
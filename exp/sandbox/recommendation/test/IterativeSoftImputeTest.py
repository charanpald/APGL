


from apgl.util.Util import Util
from exp.sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from exp.sandbox.recommendation.SoftImpute import SoftImpute 
import sys
import numpy
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 
import exp.util.SparseUtils as ExpSU


class IterativeSoftImputeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=100)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)
        
        #Create a sequence of matrices 
        n = 20 
        m = 10 
        r = 8
        U, s, V = ExpSU.SparseUtils.generateLowRank((n, m), r)
        
        numInds = 100
        inds = numpy.random.randint(0, n*m-1, numInds)
        inds = numpy.unique(inds)
        numpy.random.shuffle(inds)
        numInds = inds.shape[0]
        
        inds1 = inds[0:numInds/3]
        inds2 = inds[0:2*numInds/3]
        inds3 = inds
        
        X1 = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds1)
        X2 = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds2)
        X3 = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds3)
        
        self.matrixList = [X1, X2, X3]
        self.indsList = [inds1, inds2, inds3]

    def testLearnModel(self): 
        lmbda = 0.0 
        eps = 0.1 
        k = 10
        
        matrixIterator = iter(self.matrixList)
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=k, eps=eps, svdAlg="propack")
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Check that ZList is the same as XList 
        for i, Z in enumerate(ZList):
            U, s, V = Z
            Xhat = (U*s).dot(V.T)
            
            nptst.assert_array_almost_equal(Xhat, self.matrixList[i].todense())
        
        #Compare solution with that of SoftImpute class 
        lmbda = 0.5 
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=k, eps=eps, svdAlg="propack", updateAlg="zero")
        
        matrixIterator = iter(self.matrixList)
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        lmbdas = numpy.array([lmbda])
        
        softImpute = SoftImpute(lmbdas, k=k, eps=eps)
        Z1 = softImpute.learnModel(self.matrixList[0])
        Z2 = softImpute.learnModel(self.matrixList[1])
        Z3 = softImpute.learnModel(self.matrixList[2])
        
        ZList2 = [Z1, Z2, Z3]
        
        for i, Z in enumerate(ZList):
            U, s, V = Z 
            Xhat = (U*s).dot(V.T)
            nptst.assert_array_almost_equal(Xhat, ZList2[i].todense())
        
        #Test the SVD updating solution   
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=9, eps=eps, svdAlg="propack", updateAlg="zero")
        iterativeSoftImpute.svdAlg = "svdUpdate"
        matrixIterator = iter(self.matrixList)
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Test using Randomised SVD 
        iterativeSoftImpute.svdAlg = "RSVD"
        matrixIterator = iter(self.matrixList)
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Test on an increasing then decreasing set of solutions 

    def testPredict(self): 
        #Create a set of indices 
        lmbda = 0.0 
        
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=10)
        matrixIterator = iter(self.matrixList)
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        XhatList = iterativeSoftImpute.predict(ZList, self.indsList)
        
        #Check we get the exact matrices returned 
        for i, Xhat in enumerate(XhatList): 
            nptst.assert_array_almost_equal(numpy.array(Xhat.todense()), self.matrixList[i].todense())
            
            self.assertEquals(Xhat.nnz, self.indsList[i].shape[0])

    def testModelSelect(self):
        lmbda = 0.1
        X = self.matrixList[0]
        
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=10)
        lmbdas = numpy.array([1.0, 0.8, 0.5, 0.2, 0.1])
        folds = 5 
        iterativeSoftImpute.modelSelect(X, lmbdas, folds)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
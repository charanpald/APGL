


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

    def testLearnModel(self): 
        
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
        
        matrixList = [X1, X2, X3]
        matrixIterator = iter(matrixList)
        
        lmbda = 0.0 
        
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=10)
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Check that ZList is the same as XList 
        for i, Z in enumerate(ZList):
            U, s, V = Z
            Xhat = (U*s).dot(V.T)
            
            nptst.assert_array_almost_equal(Xhat, matrixList[i].todense())
        
        #Compare solution with that of SoftImpute class 
        lmbda = 0.5 
        iterativeSoftImpute.setLambda(lmbda)
        
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        lmbdas = numpy.array([lmbda])
        softImpute = SoftImpute(lmbdas, k=10)
        Z1 = softImpute.learnModel(X1)
        Z2 = softImpute.learnModel(X2)
        Z3 = softImpute.learnModel(X3)
        
        ZList2 = [Z1, Z2, Z3]
        
        for i, Z in enumerate(ZList):
            nptst.assert_array_almost_equal(Z, ZList2[i].todense())
        
        #Test the SVD updating solution   
        iterativeSoftImpute.svdAlg = "svdUpdate"
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Test using Randomised SVD 
        iterativeSoftImpute.svdAlg = "RSVD"
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Test on an increasing then decreasing set of solutions 
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
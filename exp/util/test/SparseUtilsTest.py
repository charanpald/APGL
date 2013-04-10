


import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from exp.util.SparseUtils import SparseUtils
from apgl.util.Util import Util 

class SparseUtilsCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)

    def testGenerateSparseLowRank(self): 
        shape = (5000, 1000)
        r = 5 
        k = 10 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, True)         
        
        self.assertEquals(U.shape, (shape[0],r))
        self.assertEquals(V.shape, (shape[1], r))
        self.assertTrue(X.nnz <= k)
        
        Y = (U*s).dot(V.T)
        inds = X.nonzero()
        
        for i in range(inds[0].shape[0]):
            self.assertAlmostEquals(X[inds[0][i], inds[1][i]], Y[inds[0][i], inds[1][i]])
 
    def testGenerateLowRank(self): 
        shape = (5000, 1000)
        r = 5  
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        nptst.assert_array_almost_equal(U.T.dot(U), numpy.eye(r))
        nptst.assert_array_almost_equal(V.T.dot(V), numpy.eye(r))
        
        self.assertEquals(U.shape[0], shape[0])
        self.assertEquals(V.shape[0], shape[1])
        self.assertEquals(s.shape[0], r)
        
        #Check the range is not 
        shape = (500, 500)
        r = 100
        U, s, V = SparseUtils.generateLowRank(shape, r)
        X = (U*s).dot(V.T)
        
        self.assertTrue(abs(numpy.max(X) - 1) < 0.5) 
        self.assertTrue(abs(numpy.min(X) + 1) < 0.5) 
       

    def testReconstructLowRank(self): 
        shape = (5000, 1000)
        r = 5
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        inds = numpy.array([0])
        X = SparseUtils.reconstructLowRank(U, s, V, inds)
        
        self.assertEquals(X[0, 0], (U[0, :]*s).dot(V[0, :]))
        
    def testSvdSoft(self): 
        A = scipy.sparse.rand(10, 10, 0.2)
        A = A.tocsc()
        
        lmbda = 0.1
        k = 6
        U, s, V = SparseUtils.svdSoft(A, lmbda, k)
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
        
        U2, s2, V2 = SparseUtils.svdSparseLowRank(A, U, s, V)
        
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
        
if __name__ == '__main__':
    unittest.main()



import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from exp.util.SparseUtils import SparseUtils

class SparseUtilsCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        #numpy.random.seed(21)

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
       

    def testReconstructLowRank(self): 
        shape = (5000, 1000)
        r = 5
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        inds = numpy.array([0])
        X = SparseUtils.reconstructLowRank(U, s, V, inds)
        
        self.assertEquals(X[0, 0], (U[0, :]*s).dot(V[0, :]))
        
        
if __name__ == '__main__':
    unittest.main()
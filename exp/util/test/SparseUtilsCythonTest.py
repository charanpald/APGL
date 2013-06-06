

import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from exp.util.SparseUtilsCython import SparseUtilsCython

class SparseUtilsCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)

    def testPartialReconstructValsPQ(self):
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        indices = numpy.nonzero(Y)  
        vals = SparseUtilsCython.partialReconstructValsPQ(indices[0], indices[1], numpy.ascontiguousarray(U*s), V)
        X = numpy.reshape(vals, Y.shape)
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just some indices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        rowInds = numpy.array(inds[0], numpy.int)
        colInds = numpy.array(inds[1], numpy.int)
        
        vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, numpy.ascontiguousarray(U*s), V)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(vals[i], Y[j, k])  
            
        
        self.assertEquals(A.nnz, inds[0].shape[0])

    def testPartialReconstructValsPQ2(self): 
        numRuns = 10         
        
        for i in range(numRuns): 
            m = numpy.random.randint(5, 50)
            n = numpy.random.randint(5, 50)
            Y = numpy.random.rand(m, n)
            
            U, s, V = numpy.linalg.svd(Y,  full_matrices=0)
            V = V.T 
            
            rowInds, colInds = numpy.nonzero(Y)  
            rowInds = numpy.array(rowInds, numpy.int32)
            colInds = numpy.array(colInds, numpy.int32)
            #print(U.shape, V.shape)
            vals = SparseUtilsCython.partialReconstructValsPQ2(rowInds, colInds, numpy.ascontiguousarray(U*s), V)
            X = numpy.reshape(vals, Y.shape)
            
            nptst.assert_almost_equal(X, Y)
        

    def testPartialOuterProduct(self):
        m = 15        
        n = 10
        
        
        u = numpy.random.rand(m)
        v = numpy.random.rand(n)
        Y = numpy.outer(u, v)
        
        indices = numpy.nonzero(Y)  
        vals = SparseUtilsCython.partialOuterProduct(indices[0], indices[1], u, v)
        X = numpy.reshape(vals, Y.shape)
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just some indices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        rowInds = numpy.array(inds[0], numpy.int)
        colInds = numpy.array(inds[1], numpy.int)
        
        vals = SparseUtilsCython.partialOuterProduct(rowInds, colInds, u, v)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(vals[i], Y[j, k])  
            
        
        self.assertEquals(A.nnz, inds[0].shape[0])

   

if __name__ == '__main__':
    unittest.main()
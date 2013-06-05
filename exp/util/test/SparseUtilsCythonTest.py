

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

    def testPartialReconstructVals(self):
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        indices = numpy.nonzero(Y)  
        vals = SparseUtilsCython.partialReconstructVals(indices[0], indices[1], U, s, V)
        X = numpy.reshape(vals, Y.shape)
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just some indices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        rowInds = numpy.array(inds[0], numpy.int)
        colInds = numpy.array(inds[1], numpy.int)
        
        vals = SparseUtilsCython.partialReconstructVals(rowInds, colInds, U, s, V)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(vals[i], Y[j, k])  
            
        
        self.assertEquals(A.nnz, inds[0].shape[0])

    def testPartialReconstructValsPQ2(self): 
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        indices = numpy.nonzero(Y)  
        rowInds = numpy.array(indices[0], numpy.int32)
        colInds = numpy.array(indices[1], numpy.int32)
        vals = SparseUtilsCython.partialReconstructValsPQ2(rowInds, colInds, numpy.ascontiguousarray(U*s), V)
        print(vals)
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

    def testPartialReconstruct(self):
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        indices = numpy.nonzero(Y)
        
        X = SparseUtilsCython.partialReconstruct(indices, U, s, V)
        X = X.todense()
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just someIndices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        
        X = SparseUtilsCython.partialReconstruct(inds, U, s, V)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(X[j, k], Y[j, k])  
            
        self.assertTrue(X.nnz == inds[0].shape[0])
        
    def testPartialReconstruct2(self):
        n = 10
        Y = numpy.random.rand(n, n)
        
        U, s, V = numpy.linalg.svd(Y)
        V = V.T 
        
        indices = numpy.nonzero(Y)
        
        X = SparseUtilsCython.partialReconstruct2(indices, U, s, V)
        X = X.todense()
        
        nptst.assert_almost_equal(X, Y)
        
        #Try just someIndices 
        density = 0.2
        A = scipy.sparse.rand(n, n, density)
        inds = A.nonzero()
        rowInds = numpy.array(inds[0], numpy.int)
        colInds = numpy.array(inds[1], numpy.int)
        
        X = SparseUtilsCython.partialReconstruct2((rowInds, colInds), U, s, V)
        
        for i in range(inds[0].shape[0]): 
            j = inds[0][i]
            k = inds[1][i]
            
            self.assertAlmostEquals(X[j, k], Y[j, k])  
            
        self.assertTrue(X.nnz == inds[0].shape[0])
        self.assertTrue(scipy.sparse.isspmatrix_csc(X))

if __name__ == '__main__':
    unittest.main()
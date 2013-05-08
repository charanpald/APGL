



import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from exp.util.LinOperatorUtils import LinOperatorUtils
from exp.util.SparseUtils import SparseUtils

class LinOperatorUtilsTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)   
        
    def testParallelSparseOp(self): 
        numRuns = 10         
        
        for i in range(numRuns): 
            m = numpy.random.randint(10, 100)
            n = numpy.random.randint(10, 100)
            p = numpy.random.randint(10, 100)
            density = numpy.random.rand()
            A = scipy.sparse.rand(m, n, density)
            A = A.tocsc()
            
            L = LinOperatorUtils.parallelSparseOp(A)
            
            u = numpy.random.rand(m)
            v = numpy.random.rand(n)
            W = numpy.random.rand(n, p)
            
            nptst.assert_array_almost_equal(L.matvec(v), A.dot(v))
            nptst.assert_array_almost_equal(L.rmatvec(u), A.T.dot(u))
            nptst.assert_array_almost_equal(L.matmat(W), A.dot(W))

    def testSparseLowRankOp(self): 
        numRuns = 10         
        
        for i in range(numRuns): 
            m = numpy.random.randint(10, 100)
            n = numpy.random.randint(10, 100)
            density = numpy.random.rand()
            A = scipy.sparse.rand(m, n, density)
            A = A.tocsc()
            
            r = numpy.random.randint(10, 100)
            U, s, V = SparseUtils.generateLowRank((m, n), r)          
            
            L = LinOperatorUtils.sparseLowRankOp(A, U, s, V)
            
            u = numpy.random.rand(m)
            v = numpy.random.rand(n)
            
            B = numpy.array(A+(U*s).dot(V.T))            
            
            nptst.assert_array_almost_equal(L.matvec(v), B.dot(v))
            nptst.assert_array_almost_equal(L.rmatvec(u), B.T.dot(u))

    def testParallelSparseLowRankOp(self): 
        numRuns = 10         
        
        for i in range(numRuns): 
            m = numpy.random.randint(10, 100)
            n = numpy.random.randint(10, 100)
            density = numpy.random.rand()
            A = scipy.sparse.rand(m, n, density)
            A = A.tocsc()
            
            r = numpy.random.randint(10, 100)
            U, s, V = SparseUtils.generateLowRank((m, n), r)          
            
            L = LinOperatorUtils.parallelSparseLowRankOp(A, U, s, V)
            
            u = numpy.random.rand(m)
            v = numpy.random.rand(n)
            
            B = numpy.array(A+(U*s).dot(V.T))            
            
            nptst.assert_array_almost_equal(L.matvec(v), B.dot(v))
            nptst.assert_array_almost_equal(L.rmatvec(u), B.T.dot(u))
      
if __name__ == '__main__':
    unittest.main()
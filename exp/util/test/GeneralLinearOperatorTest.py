

import unittest
import numpy
import numpy.testing as nptst 
import scipy.sparse 
from exp.util.GeneralLinearOperator import GeneralLinearOperator

class GeneralLinearOperatorTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)   
        
    def testAsLinearOperator(self):
        m = 80 
        n = 100 
        density = 0.5
        
        X = scipy.sparse.rand(m,n,density)
        k = 10 
        
        
        L = GeneralLinearOperator.asLinearOperator(X)
        u = numpy.random.rand(m)
        v = numpy.random.rand(n)        
        U = numpy.random.rand(m, k)
        V = numpy.random.rand(n, k)
        
        nptst.assert_array_almost_equal(X.dot(v), L.matvec(v))
        nptst.assert_array_almost_equal(X.T.dot(u), L.rmatvec(u))
        
        nptst.assert_array_almost_equal(X.dot(V), L.matmat(V))
        nptst.assert_array_almost_equal(X.T.dot(U), L.rmatmat(U))
        
      
if __name__ == '__main__':
    unittest.main()
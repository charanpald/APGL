

import unittest
import numpy
import scipy.sparse 
from exp.sandbox.RandomisedSVD import RandomisedSVD
from apgl.util.Util import Util
import numpy.testing as nptst 


class  RandomisedSVDTest(unittest.TestCase):
    def setUp(self):
        numpy.random.rand(21)
        numpy.set_printoptions(suppress=True, linewidth=200, precision=3)

    def testSvd(self): 
        n = 100 
        A = scipy.sparse.rand(n, n, 0.1)
        
        ks = [10, 20, 30, 40] 
        q = 2 
        
        lastError = numpy.linalg.norm(A.todense())        
        
        for k in ks: 
            U, s, V = RandomisedSVD.svd(A, k, q)
            
            nptst.assert_array_almost_equal(U.T.dot(U), numpy.eye(k))
            nptst.assert_array_almost_equal(V.T.dot(V), numpy.eye(k))
            A2 = (U*s).dot(V.T)
            
            error = numpy.linalg.norm(A - A2)
            self.assertTrue(error <= lastError)
            lastError = error 
            
            #Compare versus exact svd 
            U, s, V = numpy.linalg.svd(numpy.array(A.todense()))
            inds = numpy.flipud(numpy.argsort(s))[0:k*2]
            U, s, V = Util.indSvd(U, s, V, inds)
            
            Ak = (U*s).dot(V.T)
            
            error2 = numpy.linalg.norm(A - Ak)
            self.assertTrue(error2 <= error)

if __name__ == '__main__':
    unittest.main()


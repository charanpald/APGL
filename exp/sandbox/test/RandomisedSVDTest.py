

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
        m = 80
        A = scipy.sparse.rand(m, n, 0.1)
        
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

    def testSvd2(self): 
        """
        We test the situation in which one gives an initial omega matrix 
        for the random projections. 
        """
        numRuns = 10 
        
        for i in range(numRuns): 
            m, n = numpy.random.randint(10, 100), numpy.random.randint(10, 100) 
            X = numpy.random.rand(m, n)
            
            k = numpy.random.randint(5, min(m, n)) 
            U, s, V = RandomisedSVD.svd(X, k)
    
            D = numpy.random.rand(m, n)*0.1
    
            Y = X + D 
            U2, s2, V2 = RandomisedSVD.svd(Y, k, p=0, q=0)
    
            U3, s3, V3 = RandomisedSVD.svd(Y, k, p=0, q=0, omega=V)
            
            error1 = numpy.linalg.norm(Y - (U2*s2).dot(V2.T)) 
            error2 = numpy.linalg.norm(Y - (U3*s3).dot(V3.T))
            
            self.assertTrue(error1 >= error2)


if __name__ == '__main__':
    unittest.main()


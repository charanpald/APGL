
from exp.sandbox.EfficientNystrom import EfficientNystrom
import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 

class ClusterBoundTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)
        #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testEigWeight(self):
        tol = 10**-3

        W = numpy.random.rand(10, 10)
        W = W.dot(W.T)
        w, U = numpy.linalg.eig(W)
        
        W = scipy.sparse.csr_matrix(W)
        
        k = 3 
        m = 5 
        lmbda, V = EfficientNystrom.eigWeight(W, m, k)
        WHat = V.dot(numpy.diag(lmbda)).dot(V.T)
        
        #print(V)
        print(lmbda.shape)
        #self.assertTrue(numpy.linalg.norm(W - WHat) < tol)        

if __name__ == '__main__':
    unittest.main()
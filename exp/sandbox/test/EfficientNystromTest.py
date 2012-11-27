
from exp.sandbox.EfficientNystrom import EfficientNystrom
import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from apgl.graph import GraphUtils 

class ClusterBoundTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)
        #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testEigWeight(self):
        tol = 10**-3
        
        n = 100
        W = numpy.random.rand(n, n)
        W = W.dot(W.T)
        w, U = numpy.linalg.eig(W)
        
        W = scipy.sparse.csr_matrix(W)
        
        k = 4 
        m = 5 
        lmbda, V = EfficientNystrom.eigWeight(W, m, k)
        

        MHat = V.dot(numpy.diag(lmbda)).dot(V.T)
        
        I = scipy.sparse.eye(n, n)
        L = GraphUtils.normalisedLaplacianSym(W) 
        M = I - L 
        
        #print(V)
        numpy.linalg.norm(M.todense() - MHat)
        #print(numpy.linalg.norm(M.todense()))
        #self.assertTrue(numpy.linalg.norm(W - WHat) < tol)
        
        #For fixed k, increasing m should improve approximation but not always 
        lastError = 10        
        
        for m in range(k+1, n+1, 10): 
            lmbda, V = EfficientNystrom.eigWeight(W, m, k)
            #print(V)
            MHat = V.dot(numpy.diag(lmbda)).dot(V.T)
        
            
            error = numpy.linalg.norm(M.todense() - MHat)
            
            self.assertTrue(error <= lastError)
            lastError = error 

    def testOrthogonalise(self): 
        n = 20
        k = 10
        U = numpy.random.rand(n, k)
        lmbda = numpy.random.rand(k)
        
        lmbdaTilde, UTilde = EfficientNystrom.orthogonalise(lmbda, U)
        
        A = (U*lmbda).dot(U.T)
        A2 = (UTilde*lmbdaTilde).dot(UTilde.T)
        
        tol = 10**-6 
        self.assertTrue(numpy.linalg.norm(A - A2) < tol) 
        
        self.assertTrue(numpy.linalg.norm(UTilde.T.dot(UTilde) - numpy.eye(k)) < tol) 
        
        #print(lmbdaTilde)

if __name__ == '__main__':
    unittest.main()
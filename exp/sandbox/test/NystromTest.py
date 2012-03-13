

import unittest
import numpy
import scipy.sparse 
from apgl.sandbox.Nystrom import Nystrom
from apgl.util.SparseUtils import SparseUtils


class  NystromTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.rand(21)
        numpy.set_printoptions(suppress=True, linewidth=200, precision=3)

    def testEigpsd(self):
        tol = 10**-3

        A = numpy.random.rand(10, 10)
        A = A.dot(A.T)
        w, U = numpy.linalg.eig(A)
        A = U.dot(numpy.diag(w+1)).dot(U.T)

        n = 10
        lmbda, V = Nystrom.eigpsd(A, n)
        AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
        self.assertTrue(numpy.linalg.norm(A - AHat) < tol)

        #Approximation should be good when n < 10
        for n in range(2, 11):
            inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
            lmbda, V = Nystrom.eigpsd(A, inds)
            AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
            AHat2 = Nystrom.matrixApprox(A, inds)
            self.assertTrue(numpy.linalg.norm(A - AHat) < numpy.linalg.norm(A))
            self.assertAlmostEquals(numpy.linalg.norm(A - AHat), numpy.linalg.norm(A - AHat2))

        #Now let's test on positive semi-definite
        w[9] = 0
        A = U.dot(numpy.diag(w+1)).dot(U.T)

        n = 10
        lmbda, V = Nystrom.eigpsd(A, n)
        AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
        self.assertTrue(numpy.linalg.norm(A - AHat) < tol)

        #Approximation should be good when n < 10
        for n in range(2, 11):
            inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
            lmbda, V = Nystrom.eigpsd(A, inds)
            AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
            AHat2 = Nystrom.matrixApprox(A, inds)
            self.assertTrue(numpy.linalg.norm(A - AHat) < numpy.linalg.norm(A))
            self.assertAlmostEquals(numpy.linalg.norm(A - AHat), numpy.linalg.norm(A - AHat2))

    def testEigpsd2(self):
        #These tests are on sparse matrices 
        tol = 10**-3

        A = numpy.random.rand(10, 10)
        A = A.dot(A.T)
        w, U = numpy.linalg.eig(A)
        A = U.dot(numpy.diag(w+1)).dot(U.T)
        As = scipy.sparse.csr_matrix(A)

        n = 10
        lmbda, V = Nystrom.eigpsd(As, n)
        AHat = scipy.sparse.csr_matrix(V.dot(numpy.diag(lmbda)).dot(V.T))
        self.assertTrue(numpy.linalg.norm(A - AHat) < tol)

        for n in range(2, 11):
            inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
            lmbda, V = Nystrom.eigpsd(As, inds)
            AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
            AHat2 = Nystrom.matrixApprox(A, inds)
            self.assertTrue(numpy.linalg.norm(A - AHat) < numpy.linalg.norm(A))
            self.assertAlmostEquals(numpy.linalg.norm(A - AHat), numpy.linalg.norm(A - AHat2))

    def testEig(self):
        tol = 10**-3

        #Test with an indeterminate matrix 
        A = numpy.random.rand(10, 10)
        A = A.dot(A.T)
        w, U = numpy.linalg.eig(A)
        A = U.dot(numpy.diag(w-1)).dot(U.T)

        n = 10
        lmbda, V = Nystrom.eig(A, n)
        AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
        self.assertTrue(numpy.linalg.norm(A - AHat) < tol)
        
        #Approximation should be good when n < 10
        for n in range(2, 11):
            inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
            
            lmbda, V = Nystrom.eig(A, inds)
            AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
            AHat2 = Nystrom.matrixApprox(A, inds)
            #self.assertTrue(numpy.linalg.norm(A - AHat) < numpy.linalg.norm(A))
            #print(n)
            #print(numpy.linalg.norm(A - AHat))
            #print(numpy.linalg.norm(A - AHat2))

        #Test with a positive definite matrix 
        A = numpy.random.rand(10, 10)
        A = A.dot(A.T)
        w, U = numpy.linalg.eig(A)
        A = U.dot(numpy.diag(w+1)).dot(U.T)

        #Approximation should be good when n < 10
        for n in range(2, 11):
            inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])

            lmbda, V = Nystrom.eig(A, inds)
            AHat = V.dot(numpy.diag(lmbda)).dot(V.T)
            AHat2 = Nystrom.matrixApprox(A, inds)
            self.assertTrue(numpy.linalg.norm(A - AHat) < numpy.linalg.norm(A))

    def testMatrixApprox(self):
        tol = 10**-6 
        A = numpy.random.rand(10, 10)
        A = A.dot(A.T)

        n = 5
        inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
        AHat = Nystrom.matrixApprox(A, inds)

        n = 10
        AHat2 = Nystrom.matrixApprox(A, n)
        self.assertTrue(numpy.linalg.norm(A - AHat2) < numpy.linalg.norm(A - AHat))
        self.assertTrue(numpy.linalg.norm(A - AHat2) < tol)

        #Test on a sparse matrix
        As = scipy.sparse.csr_matrix(A)
        n = 5
        inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
        AHat = Nystrom.matrixApprox(As, inds)

        n = 10
        AHat2 = Nystrom.matrixApprox(As, n)
        self.assertTrue(SparseUtils.norm(As - AHat2) < SparseUtils.norm(As - AHat))
        self.assertTrue(SparseUtils.norm(As - AHat2) < tol)

        #Compare dense and sparse solutions
        for n in range(1, 9):
            inds = numpy.sort(numpy.random.permutation(A.shape[0])[0:n])
            AHats = Nystrom.matrixApprox(As, inds)
            AHat = Nystrom.matrixApprox(A, inds)

            self.assertTrue(numpy.linalg.norm(AHat - numpy.array(AHats.todense())) < tol)

if __name__ == '__main__':
    unittest.main()


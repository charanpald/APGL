from exp.sandbox.SVDUpdate import SVDUpdate
from apgl.util.Util import Util 
import unittest
import numpy
import scipy
import logging
import sys 
import numpy.testing as nptst 

class SVDUpdateTestCase(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

        # To test functions
        m = 100
        n = 80
        r = 20
        self.k = 10
        p = 0.1  # proportion of coefficients in the sparse matrix

        self.A = numpy.random.rand(m, n)
        U, s, VT = numpy.linalg.svd(self.A)
        V = VT.T
        inds = numpy.flipud(numpy.argsort(s))
        self.U = U[:, inds[0:self.k]]
        self.s = s[inds[0:self.k]]
        self.V = V[:, inds[0:self.k]]

        self.C = numpy.random.rand(r, n)
        
        # Specific to addCols functions
        self.B = numpy.random.rand(m, r)
        self.AB = numpy.c_[self.A, self.B]
        
        UAB, sAB, VABT = numpy.linalg.svd(self.AB, full_matrices=False)
        VAB = VABT.T 
        inds = numpy.flipud(numpy.argsort(sAB))
        UAB = UAB[:, inds[0:self.k]]
        sAB = sAB[inds[0:self.k]]
        VAB = VAB[:, inds[0:self.k]]
        self.ABk = numpy.dot(numpy.dot(UAB, numpy.diag(sAB)), VAB.T)

        # Specific to addSparse functions
        X = numpy.random.rand(m, n)
        X[numpy.random.rand(m, n) < 1-p] = 0
        self.X = scipy.sparse.csc_matrix(X)
        self.AX = self.A + self.X.todense()
        UAX, sAX, VAXT = numpy.linalg.svd(self.AX, full_matrices=False)
        VAX = VAXT.T 
        inds = numpy.flipud(numpy.argsort(sAX))
        UAX = UAX[:, inds[0:self.k]]
        sAX = sAX[inds[0:self.k]]
        VAX = VAX[:, inds[0:self.k]]
        self.AXk = numpy.dot(numpy.dot(UAX, numpy.diag(sAX)), VAX.T)
        UAkXk, sAkXk, VAkXk = SVDUpdate.addSparse(self.U, self.s, self.V, self.X, self.k)
        self.AkXk = numpy.dot(numpy.dot(UAkXk, numpy.diag(sAkXk)), VAkXk.T)

        

    def testAddCols(self):
        Utilde, Stilde, Vtilde = SVDUpdate.addCols(self.U, self.s, self.V, self.B)
        ABkEst = numpy.dot(numpy.dot(Utilde, Stilde), Vtilde.T)

        print(numpy.linalg.norm(self.AB))
        print(numpy.linalg.norm(self.AB - self.ABk))
        print(numpy.linalg.norm(self.AB - ABkEst))
        print(numpy.linalg.norm(self.ABk - ABkEst))

    def testAddCols2(self):
        Utilde, Stilde, Vtilde = SVDUpdate.addCols2(self.U, self.s, self.V, self.B)
        ABkEst = numpy.dot(numpy.dot(Utilde, Stilde), Vtilde.T)

        print(numpy.linalg.norm(self.AB))
        print(numpy.linalg.norm(self.AB - self.ABk))
        print(numpy.linalg.norm(self.AB - ABkEst))
        print(numpy.linalg.norm(self.ABk - ABkEst))
        
        
    def testAddRows(self): 
        #Test if same as add cols 
        #Test case when k = rank 
        Utilde, Stilde, Vtilde = SVDUpdate.addRows(self.U, self.s, self.V, self.C)
        
        nptst.assert_array_almost_equal(Utilde.T.dot(Utilde), numpy.eye(Utilde.shape[1]))
        nptst.assert_array_almost_equal(Vtilde.T.dot(Vtilde), numpy.eye(Vtilde.shape[1]))
        
        self.assertEquals(Stilde.shape[0], self.k)
        
        #Check we get the original solution with full SVD 
        U, s, V = numpy.linalg.svd(self.A)
        inds = numpy.flipud(numpy.argsort(s))
        U, s, V = Util.indSvd(U, s, V, inds)
        
        Utilde, Stilde, Vtilde = SVDUpdate.addRows(U, s, V, self.C)
        D = numpy.r_[self.A, self.C]
        
        nptst.assert_array_almost_equal(D, (Utilde*Stilde).dot(Vtilde.T), 4)
        
        #Check solution for partial rank SVD 
        k = 20 
        U, s, V = numpy.linalg.svd(self.A)
        inds = numpy.flipud(numpy.argsort(s))[0:k]
        U, s, V = Util.indSvd(U, s, V, inds)
        
        Utilde, Stilde, Vtilde = SVDUpdate.addRows(U, s, V, self.C)
        D = numpy.r_[(U*s).dot(V.T), self.C]
        U, s, V = numpy.linalg.svd(D)
        inds = numpy.flipud(numpy.argsort(s))[0:k]
        U, s, V = Util.indSvd(U, s, V, inds)
        
        nptst.assert_array_almost_equal((U*s).dot(V.T), (Utilde*Stilde).dot(Vtilde.T), 4)

    def testAddSparseWrapp(self):
        X = numpy.random.rand(10, 5)
        U, s, V = numpy.linalg.svd(X)
        def myTest(U, s, V, X, k):
            self.assertTrue(X.shape == (10, 5))
            self.assertTrue(U.shape[0] == 10)
            self.assertTrue(V.shape[0] == 5)
            return U, s, V
        SVDUpdate._addSparseWrapp(myTest, U, s, V, X)
        SVDUpdate._addSparseWrapp(myTest, V, s, U, X.T)
        
    def mytestAddSparse(self, f):
        Utilde, stilde, Vtilde = f(self.U, self.s, self.V, self.X, self.k)
        if len(stilde) > self.k:
            inds = numpy.flipud(numpy.argsort(stilde))
            Utilde = Utilde[:, inds[0:self.k]]
            stilde = stilde[inds[0:self.k]]
            Vtilde = Vtilde[:, inds[0:self.k]]
        AkXkEst = (Utilde * stilde).dot(Vtilde.T)
        print("\n", f.__name__)
#        print("||A+X||:", numpy.linalg.norm(self.AX))
#        print("||A+X - (A+X)_k||:", numpy.linalg.norm(self.AX - self.AXk))
#        print("||A+X - (A_k+X)_k||:", numpy.linalg.norm(self.AX - self.AkXk))
#        print("||A+X - (A_k+X)_k^{from alg}||:", numpy.linalg.norm(self.AX - AkXkEst))
        print("||(A_k+X)_k||:", numpy.linalg.norm(self.AkXk))
        print("||(A_k+X)_k - (A_k+X)_k^{from alg}||:", numpy.linalg.norm(self.AkXk - AkXkEst))

    def testAddSparse(self):
        self.mytestAddSparse(SVDUpdate.addSparse)

    def testAddSparseArpack(self):
        self.mytestAddSparse(SVDUpdate.addSparseArpack)

    def testAddSparseProjected(self):
        self.mytestAddSparse(SVDUpdate.addSparseProjected)

    def testAddSparseRSVD(self):
        def f(U, s, V, X, k):
            return SVDUpdate.addSparseRSVD(U, s, V, X, k, kX=2*self.k, kRand=2*self.k, q=1)
        self.mytestAddSparse(f)

if __name__ == '__main__':
    unittest.main()




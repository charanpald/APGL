
from exp.sandbox.EigenUpdater import EigenUpdater
from apgl.util.Util import Util 
import unittest
import numpy
import logging
import sys 

class EigenUpdaterTestCase(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    #@unittest.skip("Demonstrating skipping")
    def testEigenConcat(self):
        tol = 10**-6
        
        for i in range(3): 
            m = numpy.random.randint(10, 20)
            n = numpy.random.randint(5, 10)
            p = numpy.random.randint(5, 10)
#            A = numpy.zeros((m, n), numpy.complex)
#            B = numpy.zeros((m, p), numpy.complex)
#            A.real = numpy.random.randn(m, n)
#            A.imag = numpy.random.randn(m, n)
#            B.real = numpy.random.randn(m, p)
#            B.imag = numpy.random.randn(m, p)
            A = numpy.random.randn(m, n)
            B = numpy.random.randn(m, p)

            logging.debug("m="+str(m)+" n="+str(n)+" p="+str(p))

            AcB = numpy.c_[A, B]
            ABBA = AcB.conj().T.dot(AcB)

            AA = ABBA[0:n, 0:n]
            AB = ABBA[0:n, n:]
            BB = ABBA[n:, n:]

            lastError = 1000
            lastError2 = 1000

            for k in range(1,n):
                #logging.debug("k="+str(k))
                #First compute eigen update estimate
                omega, Q = numpy.linalg.eig(AA)
                pi, V = EigenUpdater.eigenConcat(omega, Q, AB, BB, k)
                ABBAEst = V.dot(numpy.diag(pi)).dot(V.conj().T)

                
                t = min(k, Util.rank(ABBA))
                self.assertTrue(pi.shape[0] == t)
                self.assertTrue(numpy.linalg.norm(V.conj().T.dot(V) - numpy.eye(t)) < tol)

                #Second compute another eigen update estimate
                omega, Q = numpy.linalg.eig(AA)
                pi2, V2, D2, D2UD2 = EigenUpdater.lazyEigenConcatAsUpdate(omega, Q, AB, BB, k, debug=True)
                ABBAEst2 = V2.dot(numpy.diag(pi2)).dot(V2.conj().T)


                U = ABBA.copy()
                U[0:n, 0:n] = 0
                self.assertTrue(numpy.linalg.norm(U - D2.dot(D2UD2).dot(D2.conj().T)) < tol )

                t = min(k, Util.rank(ABBA))
                self.assertTrue(pi2.shape[0] == t)
                self.assertTrue(numpy.linalg.norm(V2.conj().T.dot(V2) - numpy.eye(pi2.shape[0])) < tol)

                #Compute estimate using eigendecomposition of full matrix
                sfull, Vfull = numpy.linalg.eig(ABBA)
                indsfull = numpy.flipud(numpy.argsort(numpy.abs(sfull)))
                Vfull = Vfull[:, indsfull[0:k]]
                sfull = sfull[indsfull[0:k]]
                ABBAEstfull = Vfull.dot(numpy.diag(sfull)).dot(Vfull.conj().T)

                #The errors should reduce
                error = numpy.linalg.norm(ABBAEst - ABBA)
                if Util.rank(ABBA)==k:
                    self.assertTrue(error <= tol)
                lastError = error

                error = numpy.linalg.norm(ABBAEst2 - ABBA)
                self.assertTrue(error <= lastError2+tol)
                lastError2 = error
                

    #@unittest.skip("Demonstrating skipping")
    def testEigenAdd(self):
        for i in range(3):
            numCols = numpy.random.randint(5, 10)
            numXRows = numpy.random.randint(5, 10)
            numYRows = numpy.random.randint(5, 10)

            A = numpy.random.rand(numXRows, numCols)
            Y = numpy.random.rand(numYRows, numCols)

            AA = A.conj().T.dot(A)
            AA = (AA + AA.conj().T)/2
            YY = Y.conj().T.dot(Y)

            lastError = 1000

            for k in range(1, min((numXRows, numCols))):
                #Note using eigh since AA is hermatian 
                omega, Q = numpy.linalg.eigh(AA)
                pi, V = EigenUpdater.eigenAdd(omega, Q, Y, k)
                Pi = numpy.diag(pi)

                tol = 10**-3
                t = min(k, Util.rank(AA+YY))
                self.assertTrue(pi.shape[0] == t)
                self.assertTrue(numpy.linalg.norm(V.conj().T.dot(V) - numpy.eye(t)) < tol)

                inds2 = numpy.flipud(numpy.argsort(numpy.abs(omega)))
                Q = Q[:, inds2[0:k]]
                omega = omega[inds2[0:k]]
                AAk = Q.dot(numpy.diag(omega)).dot(Q.conj().T)
                AAkpYY = AAk + YY
                AApYYEst = V.dot(Pi.dot(V.conj().T))

                error = numpy.linalg.norm(AApYYEst - (AA+YY))
                self.assertTrue(lastError - error >= -tol)
                lastError = error
                
        #Seems to work when k is significant fraction of whole

    def testEigenAdd2(self):
        tol = 10**-6

        for i in range(10):
            m = numpy.random.randint(5, 10)
            n = numpy.random.randint(5, 10)
            p = numpy.random.randint(5, 10)
            A = numpy.random.randn(m, n)
            Y1 = numpy.random.randn(n, p)
            Y2 = numpy.random.randn(n, p)

            AA = A.conj().T.dot(A)
            Y1Y2 = Y1.dot(Y2.conj().T)
            lastError = 100

            omega, Q = numpy.linalg.eigh(AA)
            self.assertTrue(numpy.linalg.norm(AA-(Q*omega).dot(Q.conj().T)) < tol )
            C = AA + Y1Y2 + Y1Y2.conj().T
            for k in range(1,9):
                pi, V, D, DUD = EigenUpdater.eigenAdd2(omega, Q, Y1, Y2, k, debug = True)
                # V is "orthogonal"
                self.assertTrue(numpy.linalg.norm(V.conj().T.dot(V) - numpy.eye(V.shape[1])) < tol  )

                # The approximation converges to the exact decomposition 
                C_k = (V*pi).dot(V.conj().T)
                error = numpy.linalg.norm(C-C_k)
                if Util.rank(C)==k:
                    self.assertTrue(error <= tol)
                lastError = error
                
                # DomegaD corresponds to AA_k
                omega_k, Q_k = Util.indEig(omega, Q, numpy.flipud(numpy.argsort(omega))[0:k])
                DomegakD = (D*numpy.c_[omega_k[numpy.newaxis,:],numpy.zeros((1,max(D.shape[1]-k,0)))]).dot(D.conj().T)
                self.assertTrue(numpy.linalg.norm((Q_k*omega_k).dot(Q_k.conj().T)-DomegakD) < tol )
                
                # DUD is exactly decomposed
                self.assertTrue(numpy.linalg.norm(Y1Y2 + Y1Y2.conj().T - D.dot(DUD).dot(D.conj().T)) < tol )
                
    def testEigenRemove(self):
        tol = 10**-6

        for i in range(10):
            m = numpy.random.randint(5, 10)
            n = numpy.random.randint(5, 10)

            #How many rows/cols to remove 
            p = numpy.random.randint(1, 5)

            A = numpy.random.randn(m, n)
            C = A.conj().T.dot(A)

            lastError = 100

            omega, Q = numpy.linalg.eigh(C)
            self.assertTrue(numpy.linalg.norm(C-(Q*omega).dot(Q.conj().T)) < tol )
            #
            Cprime = C[0:n-p, 0:n-p]
            
            for k in range(1,9):
                pi, V, K, Y1, Y2, omega2 = EigenUpdater.eigenRemove(omega, Q, n-p, k, debug=True)
                # V is "orthogonal"
                self.assertTrue(numpy.linalg.norm(V.conj().T.dot(V) - numpy.eye(V.shape[1])) < tol  )

                # The approximation converges to the exact decomposition 
                C_k = (V*pi).dot(V.conj().T)
                error = numpy.linalg.norm(Cprime-C_k)

                if Util.rank(C)<k:
                    self.assertTrue(error <= tol)
                lastError = error

    def testEigenRemove2(self):
        tol = 10**-6 
        m = 10
        n = 8
        A = numpy.random.randn(m, n)
        C = A.conj().T.dot(A)

        p = 5
        k = 8

        omega, Q = numpy.linalg.eig(C)
        Cprime = C[0:n-p, 0:n-p]

        pi, V = EigenUpdater.eigenRemove(omega, Q, n-p, k, debug=False)

        C_k = (V*pi).dot(V.conj().T)
        error = numpy.linalg.norm(Cprime-C_k)

        self.assertTrue(error <= tol)

if __name__ == '__main__':
    unittest.main()


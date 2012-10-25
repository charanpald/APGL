'''
Created on 31 Jul 2009

@author: charanpal
'''

import unittest
import numpy
import scipy.linalg
import logging
import sys
import numpy.testing as nptst 
 
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.util.PathDefaults import PathDefaults
from apgl.graph.SparseGraph import SparseGraph

#TODO: Test sampleWithoutReplacemnt 
#TODO: Test randNormalInt 

class UtilTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.random.seed(22)
        numpy.set_printoptions(precision=3, suppress=True, linewidth=150)


    def tearDown(self):
        pass


    def testHistogram(self):
        v = numpy.array([0, 0, 1, 5, 0, 2, 2, 2, 5])
        
        (freq, items) = Util.histogram(v)
        self.assertTrue((freq == numpy.array([3, 1, 3, 2])).all())
        self.assertTrue((items == numpy.array([0, 1, 2, 5])).all())

    def testComputeMeanVar(self):
        pass 

    def testMode(self):
        x = numpy.array([1,1,1,2,2,3,3,3,3,3,5,5])
        self.assertEquals(Util.mode(x), 3)

        x = numpy.array([1,1,1,2,2,3,3,3,5,5])
        self.assertEquals(Util.mode(x), 1)

        x = numpy.array([1,2,3,4])
        self.assertEquals(Util.mode(x), 1)

        x = numpy.array([0])
        self.assertEquals(Util.mode(x), 0)

    def testRank(self):
        X = numpy.random.rand(10, 1)
        self.assertEquals(Util.rank(X), 1)

        X = numpy.random.rand(10, 12)
        self.assertEquals(Util.rank(X), 10)

        X = numpy.random.rand(31, 12)
        self.assertEquals(Util.rank(X), 12)

        K = numpy.dot(X, X.T)
        self.assertEquals(Util.rank(X), 12)

    def testPrintIteration(self):
        #Util.printIteration(0, 1, 1)
        #Util.printIteration(0, 1, 10)
        #Util.printIteration(9, 1, 10)
        #Util.printIteration(9, 1, 11)
        #Util.printIteration(1, 1, 7)
        pass

    def testRandomChoice(self):
        v = numpy.array([0.25, 0.25, 0.25])

        tol = 10**-2
        c = numpy.zeros(3)
        numSamples = 500

        for i in range(numSamples):
            j = Util.randomChoice(v)
            #logging.debug(j)
            c[j] += 1

        self.assertTrue((c/numSamples == numpy.array([0.33, 0.33, 0.33])).all() < tol)

        v = v * 20
        c = numpy.zeros(3)

        for i in range(numSamples):
            j = Util.randomChoice(v)
            #logging.debug(j)
            c[j] += 1

        self.assertTrue((c/numSamples == numpy.array([0.33, 0.33, 0.33])).all() < tol)

        #Now try different distribution 
        v = numpy.array([0.2, 0.6, 0.2])

        c = numpy.zeros(3)

        for i in range(numSamples):
            j = Util.randomChoice(v)
            #logging.debug(j)
            c[j] += 1

        self.assertTrue((c/numSamples == v).all() < tol)

        #Test empty vector
        v = numpy.array([])
        self.assertEquals(Util.randomChoice(v), -1)

        #Test case where we want multiple random choices
        n = 1000
        v = numpy.array([0.2, 0.6, 0.2])
        j = Util.randomChoice(v, n)

        self.assertEquals(j.shape[0], n)
        self.assertAlmostEquals(numpy.sum(j==0)/float(n), v[0], places=1)
        self.assertAlmostEquals(numpy.sum(j==1)/float(n), v[1], places=1)

        #Now test the 2D case
        n = 2000
        V = numpy.array([[0.1, 0.3, 0.6], [0.6, 0.3, 0.1]])

        J = Util.randomChoice(V, n)

        self.assertEquals(J.shape[0], V.shape[0])
        self.assertEquals(J.shape[1], n)

        self.assertAlmostEquals(numpy.sum(J[0, :]==0)/float(n), V[0, 0], places=1)
        self.assertAlmostEquals(numpy.sum(J[0, :]==1)/float(n), V[0, 1], places=1)
        self.assertAlmostEquals(numpy.sum(J[0, :]==2)/float(n), V[0, 2], places=1)

        self.assertAlmostEquals(numpy.sum(J[1, :]==0)/float(n), V[1, 0], places=1)
        self.assertAlmostEquals(numpy.sum(J[1, :]==1)/float(n), V[1, 1], places=1)
        self.assertAlmostEquals(numpy.sum(J[1, :]==2)/float(n), V[1, 2], places=1)
        

    def testFitPowerLaw(self):
        alpha = 2.7
        xmin = 1.0
        exponent = (1/(alpha))
        numPoints = 15000
        x = numpy.random.rand(numPoints)**-exponent
        x = x[x>=1]

        alpha2 = Util.fitPowerLaw(x, xmin)
        self.assertAlmostEquals(alpha, alpha2, places=1)

    def testFitDiscretePowerLaw(self):
        #Test with small x
        x = numpy.array([5])
        ks, alpha2, xmin = Util.fitDiscretePowerLaw(x)

        self.assertEquals(ks, -1)
        self.assertEquals(alpha2, -1)

        x = numpy.array([5, 2])
        ks, alpha2, xmin = Util.fitDiscretePowerLaw(x)

        #Test with a large vector x 
        alpha = 2.5
        exponent = (1/(alpha-1))
        numPoints = 15000
        x = 10*numpy.random.rand(numPoints)**-exponent
        x = numpy.array(numpy.round(x), numpy.int)
        x = x[x<=500]
        x = x[x>=1]

        xmins = numpy.arange(1, 15)

        ks, alpha2, xmin = Util.fitDiscretePowerLaw(x, xmins)
        self.assertAlmostEqual(alpha, alpha2, places=1)

    def testFitDiscretePowerLaw2(self):
        try:
            import networkx
        except ImportError:
            logging.debug("Networkx not found, can't run test")
            return

        nxGraph = networkx.barabasi_albert_graph(1000, 2)
        graph = SparseGraph.fromNetworkXGraph(nxGraph)
        degreeSeq = graph.outDegreeSequence()

        output = Util.fitDiscretePowerLaw(degreeSeq)

    def testEntropy(self):
        v = numpy.array([0, 0, 0, 1, 1, 1])

        self.assertEquals(Util.entropy(v), 1)

        v = numpy.array([0, 0, 0])
        self.assertEquals(Util.entropy(v), 0)

        v = numpy.array([1, 1, 1])
        self.assertEquals(Util.entropy(v), 0)


    def testExpandIntArray(self):
        v = numpy.array([1, 3, 2, 4], numpy.int)
        w = Util.expandIntArray(v)

        self.assertTrue((w == numpy.array([0,1,1,1,2,2,3,3,3,3], numpy.int)).all())

        v = numpy.array([], numpy.int)
        w = Util.expandIntArray(v)
        self.assertTrue((w == numpy.array([], numpy.int)).all())


    def testRandom2Choice(self):
        n = 1000
        V = numpy.array([[0.3, 0.7], [0.5, 0.5]])

        J = Util.random2Choice(V, n)
        self.assertAlmostEquals(numpy.sum(J[0, :]==0)/float(n), V[0, 0], places=1)
        self.assertAlmostEquals(numpy.sum(J[0, :]==1)/float(n), V[0, 1], places=1)

        self.assertAlmostEquals(numpy.sum(J[1, :]==0)/float(n), V[1, 0], places=1)
        self.assertAlmostEquals(numpy.sum(J[1, :]==1)/float(n), V[1, 1], places=1)

        #Now use a vector of probabilities
        v = numpy.array([0.3, 0.7])
        j = Util.random2Choice(v, n)
        self.assertAlmostEquals(numpy.sum(j==0)/float(n), v[0], places=1)
        self.assertAlmostEquals(numpy.sum(j==1)/float(n), v[1], places=1)


    def testIncompleteCholesky(self):
        numpy.random.seed(21)
        A = numpy.random.rand(5, 5)
        B = A.T.dot(A)

        k = 4
        R = Util.incompleteCholesky2(B, k)
        R2 = numpy.linalg.cholesky(B)

        #logging.debug(R)
        #logging.debug(R2)

        #logging.debug(B)
        #logging.debug(R.T.dot(R))

    def testSvd(self):
        tol = 10**-6 
        A = numpy.random.rand(10, 3)
        P, s, Q = numpy.linalg.svd(A, full_matrices=False)
        A = P[:, 0:2].dot(numpy.diag(s[0:2]).dot(Q[0:2, :]))
        P2, s2, Q2 = Util.svd(A)

        self.assertTrue(numpy.linalg.norm(P2.dot(numpy.diag(s2)).dot(Q2) -A) < tol )
        self.assertTrue(numpy.linalg.norm(P2.conj().T.dot(P2) - numpy.eye(P2.shape[1])) < tol)
        self.assertTrue(numpy.linalg.norm(Q2.dot(Q2.conj().T) - numpy.eye(Q2.shape[0])) < tol)

    def testPowerLawProbs(self):
        alpha = 3
        zeroVal = 0.1
        maxInt = 100

        p = Util.powerLawProbs(alpha, zeroVal, maxInt)

        self.assertTrue(p.shape[0] == maxInt)

    def testPrintConsiseIteration(self):
        #for i in range(10):
        #    Util.printConciseIteration(i, 1, 10)
        pass


    def testMatrixPower(self):
        A = numpy.random.rand(10, 10)

        tol = 10**-6 
        A2 = A.dot(A)

        lmbda, V = scipy.linalg.eig(A)

        A12 = Util.matrixPower(A, 0.5)

        self.assertTrue(numpy.linalg.norm(A12.dot(A12)  - A) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.linalg.inv(A) - Util.matrixPower(A, -1)) < tol)
        self.assertTrue(numpy.linalg.norm(A - Util.matrixPower(A, 1)) < tol)
        self.assertTrue(numpy.linalg.norm(A2 - Util.matrixPower(A, 2)) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.linalg.inv(A).dot(numpy.linalg.inv(A)) - Util.matrixPower(A, -2)) < tol)        
        
        #Now lets test on a low rank matrix
        lmbda[5:] = 0
        A = V.dot(numpy.diag(lmbda)).dot(numpy.linalg.inv(V))
        A2 = A.dot(A)
        A12 = Util.matrixPower(A, 0.5)
        Am12 = Util.matrixPower(A, -0.5)
        
        
        self.assertTrue(numpy.linalg.norm(numpy.linalg.pinv(A) - Util.matrixPower(A, -1)) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.linalg.pinv(A) - Am12.dot(Am12)) < tol)
        self.assertTrue(numpy.linalg.norm(A12.dot(A12)  - A) < tol)
        self.assertTrue(numpy.linalg.norm(A - Util.matrixPower(A, 1)) < tol)
        self.assertTrue(numpy.linalg.norm(A2 - Util.matrixPower(A, 2)) < tol)

    def testMatrixPowerh(self):
        A = numpy.random.rand(10, 10)
        A = A.T.dot(A)            
            
        tol = 10**-6 
        A2 = A.dot(A)

        lmbda, V = scipy.linalg.eig(A)

        A12 = Util.matrixPowerh(A, 0.5)

        self.assertTrue(numpy.linalg.norm(A12.dot(A12)  - A) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.linalg.inv(A) - Util.matrixPowerh(A, -1)) < tol)
        self.assertTrue(numpy.linalg.norm(A - Util.matrixPowerh(A, 1)) < tol)
        self.assertTrue(numpy.linalg.norm(A2 - Util.matrixPowerh(A, 2)) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.linalg.inv(A).dot(numpy.linalg.inv(A)) - Util.matrixPowerh(A, -2)) < tol)        
        
        #Now lets test on a low rank matrix
        lmbda[5:] = 0
        A = V.dot(numpy.diag(lmbda)).dot(numpy.linalg.inv(V))
        A2 = A.dot(A)
        A12 = Util.matrixPowerh(A, 0.5)
        Am12 = Util.matrixPowerh(A, -0.5)

        
        self.assertTrue(numpy.linalg.norm(numpy.linalg.pinv(A) - Util.matrixPowerh(A, -1)) < tol)
        self.assertTrue(numpy.linalg.norm(numpy.linalg.pinv(A) - Am12.dot(Am12)) < tol)
        self.assertTrue(numpy.linalg.norm(A12.dot(A12)  - A) < tol)
        self.assertTrue(numpy.linalg.norm(A - Util.matrixPowerh(A, 1)) < tol)
        self.assertTrue(numpy.linalg.norm(A2 - Util.matrixPowerh(A, 2)) < tol)
        
    def testDistanceMatrix(self): 
        numExamples1 = 10 
        numExamples2 = 15 
        numFeatures = 2 
        
        U = numpy.random.randn(numExamples1, numFeatures)
        V = numpy.random.randn(numExamples2, numFeatures)
        
        D = Util.distanceMatrix(U, V)
        
        D2 = numpy.zeros((numExamples1, numExamples2))
        
        for i in range(numExamples1): 
            for j in range(numExamples2): 
                D2[i, j] = numpy.sqrt(numpy.sum((U[i, :] - V[j, :])**2))
                
        nptst.assert_almost_equal(D, D2)

    def testCumMin(self): 
        v = numpy.array([5, 6, 4, 5, 1])
        u = Util.cumMin(v)
        nptst.assert_array_equal(u, numpy.array([5, 5, 4, 4, 1]))
        
        v = numpy.array([5, 4, 3, 2, 1])
        u = Util.cumMin(v)
        nptst.assert_array_equal(u, v)
    
        v = numpy.array([1, 2, 3])
        u = Util.cumMin(v)
        nptst.assert_array_equal(u, numpy.ones(3))    
    
    
    def testExtendArray(self): 
        X = numpy.random.rand(5, 5)
        X2 = Util.extendArray(X, (10, 5))
        
        nptst.assert_array_equal(X, X2[0:5, :])
        nptst.assert_array_equal(0, X2[5:, :])          
        
        X2 = Util.extendArray(X, (10, 5), 1.23)
        
        nptst.assert_array_equal(X, X2[0:5, :])
        nptst.assert_array_equal(1.23, X2[5:, :])  
        
        #Now try extending using an array 
        X2 = Util.extendArray(X, (10, 5), numpy.array([1, 2, 3, 4, 5]))
        nptst.assert_array_equal(X, X2[0:5, :])
        
        for i in range(5, 10): 
            nptst.assert_array_equal(numpy.array([1, 2, 3, 4, 5]), X2[i, :])          
        
    
if __name__ == "__main__":
    unittest.main()
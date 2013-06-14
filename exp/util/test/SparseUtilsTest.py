


import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from exp.util.SparseUtils import SparseUtils
from apgl.util.Util import Util 
from exp.util.MCEvaluator import MCEvaluator 
from exp.sandbox.recommendation.SoftImpute import SoftImpute 

class SparseUtilsCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)

    def testGenerateSparseLowRank(self): 
        shape = (5000, 1000)
        r = 5 
        k = 10 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)         
        
        self.assertEquals(U.shape, (shape[0],r))
        self.assertEquals(V.shape, (shape[1], r))
        self.assertTrue(X.nnz <= k)
        
        Y = (U*s).dot(V.T)
        inds = X.nonzero()
        
        for i in range(inds[0].shape[0]):
            self.assertAlmostEquals(X[inds[0][i], inds[1][i]], Y[inds[0][i], inds[1][i]])
 
    def testGenerateLowRank(self): 
        shape = (5000, 1000)
        r = 5  
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        nptst.assert_array_almost_equal(U.T.dot(U), numpy.eye(r))
        nptst.assert_array_almost_equal(V.T.dot(V), numpy.eye(r))
        
        self.assertEquals(U.shape[0], shape[0])
        self.assertEquals(V.shape[0], shape[1])
        self.assertEquals(s.shape[0], r)
        
        #Check the range is not 
        shape = (500, 500)
        r = 100
        U, s, V = SparseUtils.generateLowRank(shape, r)
        X = (U*s).dot(V.T)
        
        self.assertTrue(abs(numpy.max(X) - 1) < 0.5) 
        self.assertTrue(abs(numpy.min(X) + 1) < 0.5) 
       

    def testReconstructLowRank(self): 
        shape = (5000, 1000)
        r = 5
        
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        inds = numpy.array([0])
        X = SparseUtils.reconstructLowRank(U, s, V, inds)
        
        self.assertEquals(X[0, 0], (U[0, :]*s).dot(V[0, :]))
        
    def testSvdSoft(self): 
        A = scipy.sparse.rand(10, 10, 0.2)
        A = A.tocsc()
        
        lmbda = 0.2
        U, s, V = SparseUtils.svdSoft(A, lmbda)
        ATilde = U.dot(numpy.diag(s)).dot(V.T)     
        
        #Now compute the same matrix using numpy
        A = A.todense() 
        
        U2, s2, V2 = numpy.linalg.svd(A)
        inds = numpy.flipud(numpy.argsort(s2))
        inds = inds[s2[inds] > lmbda]
        U2, s2, V2 = Util.indSvd(U2, s2, V2, inds) 
        
        s2 = s2 - lmbda 
        s2 = numpy.clip(s, 0, numpy.max(s2)) 

        ATilde2 = U2.dot(numpy.diag(s2)).dot(V2.T)
        
        nptst.assert_array_almost_equal(s, s)
        nptst.assert_array_almost_equal(ATilde, ATilde2)
        
        #Now run svdSoft with a numpy array 
        U3, s3, V3 = SparseUtils.svdSoft(A, lmbda)
        ATilde3 = U.dot(numpy.diag(s)).dot(V.T)  
        
        nptst.assert_array_almost_equal(s, s3)
        nptst.assert_array_almost_equal(ATilde3, ATilde2)

    def testSvdSparseLowRank(self): 
        numRuns = 10   
        n = 10
        density = 0.2

        for i in range(numRuns):    
            
            A = scipy.sparse.rand(n, n, density) 
            A = A.tocsc()
            
            B = numpy.random.rand(n, n)
            U, s, V = numpy.linalg.svd(B)
            V = V.T         
            
            r = numpy.random.randint(2, n)
            U = U[:, 0:r]
            s = s[0:r]
            V = V[:, 0:r]
            #B is low rank 
            B = (U*s).dot(V.T)
            
            k = numpy.random.randint(1, r)
            U2, s2, V2 = SparseUtils.svdSparseLowRank(A, U, s, V)
            U2 = U2[:, 0:k]
            s2 = s2[0:k]
            V2 = V2[:, 0:k]
                        
            nptst.assert_array_almost_equal(U2.T.dot(U2), numpy.eye(U2.shape[1]))
            nptst.assert_array_almost_equal(V2.T.dot(V2), numpy.eye(V2.shape[1]))
            #self.assertEquals(s2.shape[0], r)
            
            A2 = (U2*s2).dot(V2.T)
            
            #Compute real SVD 
            C = numpy.array(A.todense()) + B
            U3, s3, V3 = numpy.linalg.svd(C)
            V3 = V3.T  
            U3 = U3[:, 0:k]
            s3 = s3[0:k]
            V3 = V3[:, 0:k]
    
            A3 = (U3*s3).dot(V3.T)
            
            #self.assertAlmostEquals(numpy.linalg.norm(A2 - A3), 0)
            nptst.assert_array_almost_equal(s2, s3, 3)
            nptst.assert_array_almost_equal(numpy.abs(U2), numpy.abs(U3), 3)
            nptst.assert_array_almost_equal(numpy.abs(V2), numpy.abs(V3), 3)
            
    def testGenerateSparseLowRank2(self): 
        shape = (2000, 1000)
        r = 5 
        k = 20000 

        X, U, V = SparseUtils.generateSparseLowRank2(shape, r, k, verbose=True)         
        
        self.assertEquals(U.shape, (shape[0],r))
        self.assertEquals(V.shape, (shape[1], r))
        self.assertTrue(X.nnz <= k)
        
        Y = U.dot(V.T)
        inds = X.nonzero()
        
    def testSvdPropack(self): 
        shape = (500, 100)
        r = 5 
        k = 1000 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)                
        
        k2 = 10 
        U, s, V = SparseUtils.svdPropack(X, k2)

        U2, s2, V2 = numpy.linalg.svd(X.todense())
        V2 = V2.T

        nptst.assert_array_almost_equal(s, s2[0:k2])
        nptst.assert_array_almost_equal(numpy.abs(U), numpy.abs(U2[:, 0:k2]), 3)
        nptst.assert_array_almost_equal(numpy.abs(V), numpy.abs(V2[:, 0:k2]), 3)
                
    def testSvdArpack(self): 
        shape = (500, 100)
        r = 5 
        k = 1000 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)                
        
        k2 = 10 
        U, s, V = SparseUtils.svdArpack(X, k2)

        U2, s2, V2 = numpy.linalg.svd(X.todense())
        V2 = V2.T

        nptst.assert_array_almost_equal(s, s2[0:k2])
        nptst.assert_array_almost_equal(numpy.abs(U), numpy.abs(U2[:, 0:k2]), 3)
        nptst.assert_array_almost_equal(numpy.abs(V), numpy.abs(V2[:, 0:k2]), 3)
        
    def testCentreRows(self): 
        shape = (50, 10)
        r = 5 
        k = 100 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
        rowInds, colInds = X.nonzero()
        
        for i in range(rowInds.shape[0]): 
            self.assertEquals(X[rowInds[i], colInds[i]], X.data[i])
        
        mu2 = numpy.array(X.sum(1)).ravel()
        numNnz = numpy.zeros(X.shape[0])
        
        for i in range(X.shape[0]): 
            for j in range(X.shape[1]):     
                if X[i,j]!=0:                 
                    numNnz[i] += 1
                    
        mu2 /= numNnz 
        mu2[numNnz==0] = 0
        
        X, mu = SparseUtils.centerRows(X)      
        nptst.assert_array_almost_equal(numpy.array(X.mean(1)).ravel(), numpy.zeros(X.shape[0]))
        nptst.assert_array_almost_equal(mu, mu2)
        
    def testCentreRows2(self): 
        shape = (50, 10)
        r = 5 
        k = 100 
        
        #Test if centering rows changes the RMSE
        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
 
        Y = X.copy() 
        Y.data = numpy.random.rand(X.nnz)
        
        error = ((X.data - Y.data)**2).sum()
        
        X, mu = SparseUtils.centerRows(X)
        Y, mu = SparseUtils.centerRows(Y, mu)
        
        error2 = ((X.data - Y.data)**2).sum()
        self.assertAlmostEquals(error, error2)
        
        error3 = numpy.linalg.norm(X.todense()- Y.todense())**2
        self.assertAlmostEquals(error2, error3)        
        
        
    def testCentreCols(self): 
        shape = (50, 10)
        r = 5 
        k = 100 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
        rowInds, colInds = X.nonzero()
        
        mu2 = numpy.array(X.sum(0)).ravel()
        numNnz = numpy.zeros(X.shape[1])
        
        for i in range(X.shape[0]): 
            for j in range(X.shape[1]):     
                if X[i,j]!=0:                 
                    numNnz[j] += 1
                    
        mu2 /= numNnz 
        mu2[numNnz==0] = 0
        
        X, mu = SparseUtils.centerCols(X)      
        nptst.assert_array_almost_equal(numpy.array(X.mean(0)).ravel(), numpy.zeros(X.shape[1]))
        nptst.assert_array_almost_equal(mu, mu2)       
        
    def testUncentre(self): 
        shape = (50, 10)
        r = 5 
        k = 100 

        X, U, s, V = SparseUtils.generateSparseLowRank(shape, r, k, verbose=True)   
        rowInds, colInds = X.nonzero()  
        
        Y = X.copy()

        inds = X.nonzero()
        X, mu1 = SparseUtils.centerRows(X)
        X, mu2 = SparseUtils.centerCols(X, inds=inds)   
        
        cX = X.copy()
        
        Y2 = SparseUtils.uncenter(X, mu1, mu2)
        
        nptst.assert_array_almost_equal(Y.todense(), Y2.todense(), 3)
        
        #We try softImpute on a centered matrix and check if the results are the same 
        lmbdas = numpy.array([0.1])
        softImpute = SoftImpute(lmbdas)
        
        Z = softImpute.learnModel(cX, fullMatrices=False)
        Z = softImpute.predict([Z], cX.nonzero())[0]
        
        error1 = MCEvaluator.rootMeanSqError(cX, Z)
        
        X = SparseUtils.uncenter(cX, mu1, mu2)
        Z2 = SparseUtils.uncenter(Z, mu1, mu2)
        
        error2 = MCEvaluator.rootMeanSqError(X, Z2)
        
        self.assertAlmostEquals(error1, error2)
 
    def testNonzeroRowColProbs(self): 
        m = 10 
        n = 5 
        X = scipy.sparse.rand(m, n, 0.5)
        X = X.tocsc()
        
        print(X.todense())
        
        u, v = SparseUtils.nonzeroRowColsProbs(X)
        
        print(u)
        print(v)
       
if __name__ == '__main__':
    unittest.main()
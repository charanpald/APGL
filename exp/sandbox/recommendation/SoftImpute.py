
"""
An implementation of the matrix completion algorithm in "Spectral Regularisation 
Algorithms for learning large incomplete matrices". 
"""

import numpy 
import logging 
import scipy.sparse.linalg 
from sparsesvd import sparsesvd
from apgl.util.SparseUtils import SparseUtils 
from apgl.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 
from apgl.util.Parameter import Parameter 
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter
from exp.util.SparseUtilsCython import SparseUtilsCython

class SoftImpute(AbstractMatrixCompleter): 
    def __init__(self, lmbdas, eps=0.1, k=10):
        """
        Initialise imputing algorithm with given parameters. The lmbdas array 
        is a decreasing set of lmbda values for use with soft thresholded SVD. 
        Eps is the convergence threshold and k is the rank of the SVD. 
        """
        super(SoftImpute, self).__init__()   
        
        self.lmbdas = lmbdas  
        self.eps = eps
        self.k = k        
        
    def setK(self, k):
        Parameter.checkInt(k, 1, float('inf'))
        
        self.k = k 
        
    def getK(self): 
        return self.k
   
    def learnModel(self, X, fullMatrices=True):
        """
        Learn the matrix completion using a sparse matrix X. This is the simple 
        version of the soft impute algorithm in which we store the entire 
        matrices, newZ and oldZ. 
        """
        if not scipy.sparse.isspmatrix_csc(X):
            raise ValueError("Input matrix must be csc_matrix")
            
        (n, m) = X.shape
        oldU = numpy.zeros((n, 1))
        oldS = numpy.zeros(1)
        oldV = numpy.zeros((m, 1))
        omega = X.nonzero()
        tol = 10**-6
        
        rowInds = numpy.array(omega[0], numpy.int)
        colInds = numpy.array(omega[1], numpy.int)
         
        ZList = []
        
        
        for lmbda in self.lmbdas:
            gamma = self.eps + 1
            i = 0
            
            while gamma > self.eps:
                

                ZOmega = SparseUtilsCython.partialReconstruct2((rowInds, colInds), oldU, oldS, oldV)
                Y = X - ZOmega
                Y = Y.tocsc()
                newU, newS, newV = SoftImpute.svdSoft2(Y, oldU, oldS, oldV, lmbda, self.k)
                
                normOldZ = (oldS**2).sum()
                normNewZmOldZ = (oldS**2).sum() + (newS**2).sum() - 2*numpy.trace((oldV.T.dot(newV*newS)).dot(newU.T.dot(oldU*oldS)))
                
                #We can get newZ == oldZ in which case we break
                if normNewZmOldZ < tol: 
                    gamma = 0
                elif abs(normOldZ) < tol:
                    gamma = self.eps + 1 
                else: 
                    gamma = normNewZmOldZ/normOldZ
                
                oldU = newU.copy() 
                oldS = newS.copy() 
                oldV = newV.copy() 
                
                logging.debug("Iteration " + str(i) + " gamma="+str(gamma)) 
                i += 1 
                
            logging.debug("Number of iterations for lambda="+str(lmbda) + ": " + str(i))
            
            if fullMatrices: 
                newZ = scipy.sparse.lil_matrix((newU*newS).dot(newV.T))
                ZList.append(newZ)
            else: 
                ZList.append((newU,newS,newV))
        
        if self.lmbdas.shape[0] != 1:
            return ZList
        else:
            return ZList[0]
     
        
    @staticmethod
    def svdSparseLowRank(X, U, s, V, k=10): 
        """
        Find the SVD of a matrix A = X + U s V.T in which X is sparse and B = 
        U s V.T is a low rank matrix. We do this without explictly computing 
        A, and for the first k singular vectors/values. of X.  
        """
        UX, sX, VX = sparsesvd(X, k)
        UX = UX.T
        VX = VX.T
        
        Y = numpy.c_[UX, U]
        
        Q, R = numpy.linalg.qr(Y)
        
        B = numpy.array(X.T.dot(Q)).T + Q.T.dot(U*s).dot(V.T)
        
        U2, s2, V2 = numpy.linalg.svd(B, full_matrices=False)
        U2 = Q.dot(U2)
        V2 = V2.T
        
        return U2, s2, V2 

    def learnModel2(self, X):
        """
        Learn the matrix completion using a sparse matrix X. This is the simple 
        version of the soft impute algorithm in which we store the entire 
        matrices, newZ and oldZ. 
        """
        if not scipy.sparse.isspmatrix_lil(X):
            raise ValueError("Input matrix must be lil_matrix")
            
        oldZ = scipy.sparse.lil_matrix(X.shape)
        omega = X.nonzero()
        tol = 10**-6
         
        ZList = []
        
        for lmbda in self.lmbdas:
            gamma = self.eps + 1
            while gamma > self.eps:
                newZ = oldZ.copy()
                newZ[omega] = 0
                newZ = X + newZ
                newZ = newZ.tocsc()
                    
                U, s, V = self.svdSoft(newZ, lmbda, self.k)
                #Get an "invalid value encountered in sqrt" warning sometimes
                newZ = scipy.sparse.lil_matrix((U*s).dot(V.T))
                
                oldZ = oldZ.tocsr()
                normOldZ = SparseUtils.norm(oldZ)**2
                
                normNewZmOldZ = SparseUtils.norm(newZ - oldZ)**2               
                
                #We can get newZ == oldZ in which case we break
                if normNewZmOldZ < tol: 
                    gamma = 0
                elif abs(normOldZ) < tol:
                    gamma = self.eps + 1 
                else: 
                    gamma = normNewZmOldZ/normOldZ
                
                oldZ = newZ.copy()
            
            ZList.append(newZ)
        
        if self.lmbdas.shape[0] != 1:
            return ZList
        else:
            return ZList[0]
    
    @staticmethod 
    def svdSoft(X, lmbda, k): 
        """
        Take the soft-thresholded SVD of sparse matrix X under threshold lmbda for the first 
        k singular values and vectors. Returns the left and right singular 
        vectors and the singular values. 
        """
        if not scipy.sparse.issparse(X): 
            raise ValueError("X must be a sparse matrix")
        
        U, s, V = sparsesvd(X, k) 
        U = U.T
        inds = numpy.flipud(numpy.argsort(s))
        U, s, V = Util.indSvd(U, s, V, inds) 
        
        #Soft threshold 
        s = s - lmbda
        s = numpy.clip(s, 0, numpy.max(s))

        return U, s, V 
      
    @staticmethod 
    def svdSoft2(X, U, s, V, lmbda, k): 
        """
        Compute the SVD of a matrix X + UsV^T where X is sparse and U s V^T is 
        low rank. The first k singular values are computed and then soft thresholded 
        with threshold lmbda. 
        """
        U2, s2, V2 = SoftImpute.svdSparseLowRank(X, U, s, V, k)
        
        #Soft threshold 
        s2 = s2 - lmbda
        s2 = numpy.clip(s2, 0, numpy.max(s2))
        
        return U2, s2, V2 
        
    def getMetricMethod(self): 
        return MCEvaluator.meanSqError
        
    def copy(self): 
        """
        Return a new copied version of this object. 
        """
        softImpute = SoftImpute(lmbdas=self.lmbdas, eps=self.eps, k=self.k)

        return softImpute 
        
    def name(self): 
        return "SoftImpute"
        

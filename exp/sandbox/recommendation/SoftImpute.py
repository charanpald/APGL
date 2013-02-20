
"""
An implementation of the matrix completion algorithm in "Spectral Regularisation 
Algorithms for learning large incomplete matrices". 
"""

import numpy 
import scipy.sparse
import scipy.sparse.linalg 
from apgl.util.SparseUtils import SparseUtils 
from apgl.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 

class SoftImpute(object): 
    def __init__(self, lmbdas, eps, k):
        """
        Initialise imputing algorithm with given parameters. The lmbdas array 
        is a decreasing set of lmbda values for use with soft thresholded SVD. 
        Eps is the convergence threshold and k is the rank of the SVD. 
        """
        self.lmbdas = lmbdas  
        self.eps = eps
        self.k = k          
        
    def learnModel(self, X): 
        """
        Learn the matrix completion using a sparse matrix X. 
        """
        
        oldZ = scipy.sparse.csr_matrix(X.shape)
        omega = X.nonzero()
        tol = 10**-6 
         
        ZList = []
        
        for lmbda in self.lmbdas: 
            gamma = self.eps + 1
            
            while gamma > self.eps: 
                newZ = oldZ.copy()
                
                for i in range(omega[0].shape[0]): 
                    newZ[omega[0][i], omega[1][i]] = X[omega[0][i], omega[1][i]]
                    
                U, s, V = self.svdSoft(newZ, lmbda, self.k) 
                newZ = scipy.sparse.csr_matrix((U*s).dot(V.T))
                
                normOldZ = SparseUtils.norm(oldZ)**2
                
                if abs(normOldZ) > tol: 
                    gamma = SparseUtils.norm(newZ - oldZ)**2/normOldZ
                
                if SparseUtils.norm(newZ - oldZ)**2 < tol: 
                    gamma = 0 
                
                oldZ = newZ 
            
            ZList.append(newZ)
            
        return ZList 
        
    
    @staticmethod 
    def svdSoft(A, lmbda, k): 
        """
        Take the soft-thresholded SVD of A under threshold lmbda for the first 
        k singular values and vectors. Returns the left and right singular 
        vectors and the singular values. 
        """
        
        U, s, V = scipy.sparse.linalg.svds(A, k)
        inds = numpy.flipud(numpy.argsort(s))
        U, s, V = Util.indSvd(U, s, V, inds) 
        
        #Soft threshold 
        s = s - lmbda
        s = numpy.clip(s, 0, numpy.max(s))

        return U, s, V 
        
    def getMetricMethod(self): 
        return MCEvaluator.meanSqError()
        
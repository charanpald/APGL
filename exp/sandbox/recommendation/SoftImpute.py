
"""
An implementation of the matrix completion algorithm in "Spectral Regularisation 
Algorithms for learning large incomplete matrices". 
"""

import numpy 
import 
import scipy.sparse.linalg 
from sparsesvd import sparsesvd
from apgl.util.SparseUtils import SparseUtils 
from apgl.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 
from apgl.util.Parameter import Parameter 
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter

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
        
    def learnModel(self, X): 
        """
        Learn the matrix completion using a sparse matrix X. 
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
                print("Adding oldZ entries")
                newZ[omega] = oldZ[omega]  
                print("Done")                  
                newZ = newZ.tocsc()
                    
                U, s, V = self.svdSoft(newZ, lmbda, self.k)
                #Get an "invalid value encountered in sqrt" warning sometimes 
                newZ = scipy.sparse.csc_matrix((U*s).dot(V.T))
                
                oldZ = oldZ.tocsr()
                normOldZ = SparseUtils.norm(oldZ)**2
                
                if abs(normOldZ) > tol: 
                    gamma = SparseUtils.norm(newZ - oldZ)**2/normOldZ
                
                if SparseUtils.norm(newZ - oldZ)**2 < tol: 
                    gamma = 0 
                
                oldZ = newZ 
            
            ZList.append(newZ)
        
        if self.lmbdas.shape[0] != 1: 
            return ZList 
        else: 
            return ZList[0]
        
    
    @staticmethod 
    def svdSoft(A, lmbda, k): 
        """
        Take the soft-thresholded SVD of sparse matrix A under threshold lmbda for the first 
        k singular values and vectors. Returns the left and right singular 
        vectors and the singular values. 
        """
        if not scipy.sparse.issparse(A): 
            raise ValueError("A must be a sparse matrix")
        
        #U, s, V = scipy.sparse.linalg.svds(A, k)
        U, s, V = sparsesvd(A, k) 
        U = U.T
        inds = numpy.flipud(numpy.argsort(s))
        U, s, V = Util.indSvd(U, s, V, inds) 
        
        #Soft threshold 
        s = s - lmbda
        s = numpy.clip(s, 0, numpy.max(s))

        return U, s, V 
        
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
        

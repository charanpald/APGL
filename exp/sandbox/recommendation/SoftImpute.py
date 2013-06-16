
"""
An implementation of the matrix completion algorithm in "Spectral Regularisation 
Algorithms for learning large incomplete matrices". 
"""

import numpy 
import logging 
import scipy.sparse.linalg 
import exp.util.SparseUtils as ExpSU
from apgl.util.SparseUtils import SparseUtils 
from exp.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 
from apgl.util.Parameter import Parameter 
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter
from exp.util.SparseUtilsCython import SparseUtilsCython

class SoftImpute(AbstractMatrixCompleter): 
    def __init__(self, rhos, eps=0.01, k=10):
        """
        Initialise imputing algorithm with given parameters. The rhos array 
        is a decreasing set of rho values for use with soft thresholded SVD. 
        Eps is the convergence threshold and k is the rank of the SVD. 
        """
        super(SoftImpute, self).__init__()   
        
        self.rhos = rhos  
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
        
        for rho in self.rhos:
            gamma = self.eps + 1
            i = 0
            
            Y = scipy.sparse.csc_matrix(X, dtype=numpy.float)
            U, s, V = ExpSU.SparseUtils.svdArpack(Y, 1, kmax=20)
            lmbda = rho*numpy.max(s)
            
            while gamma > self.eps:
                ZOmega = SparseUtilsCython.partialReconstructPQ((rowInds, colInds), oldU*oldS, oldV)
                Y = X - ZOmega
                Y = Y.tocsc()

                newU, newS, newV = ExpSU.SparseUtils.svdSparseLowRank(Y, oldU, oldS, oldV)
        
                #Soft threshold 
                newS = newS - lmbda
                newS = numpy.clip(newS, 0, numpy.max(newS))
                
                
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
                
            logging.debug("Number of iterations for lambda="+str(rho) + ": " + str(i))
            
            if fullMatrices: 
                newZ = scipy.sparse.lil_matrix((newU*newS).dot(newV.T))
                ZList.append(newZ)
            else: 
                ZList.append((newU,newS,newV))
        
        if self.rhos.shape[0] != 1:
            return ZList
        else:
            return ZList[0]
     

    def learnModel2(self, X):
        """
        Learn the matrix completion using a sparse matrix X. This is the simple 
        version of the soft impute algorithm in which we store the entire 
        matrices, newZ and oldZ. 
        """
        #if not scipy.sparse.isspmatrix_lil(X):
        #    raise ValueError("Input matrix must be lil_matrix")
            
        oldZ = scipy.sparse.lil_matrix(X.shape)
        omega = X.nonzero()
        tol = 10**-6
         
        ZList = []
        
        for rho in self.rhos:
            gamma = self.eps + 1
            i = 0
            while gamma > self.eps:
                Y = oldZ.copy()
                Y[omega] = 0
                Y = X + Y
                Y = Y.tocsc()
                U, s, V = ExpSU.SparseUtils.svdSoft(Y, rho)
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
                
                logging.debug("Iteration " + str(i) + " gamma="+str(gamma)) 
                i += 1
            
            logging.debug("Number of iterations for lambda="+str(rho) + ": " + str(i))
            ZList.append(newZ)
        
        if self.rhos.shape[0] != 1:
            return ZList
        else:
            return ZList[0]
    
    def predict(self, ZList, inds): 
        """
        Take a list of Z matrices (given by their decomposition) and reconstruct 
        them for the given indices. 
        """
        predXList = []        
        
        for Z in ZList: 
            U, s, V = Z
            Xhat = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds)
            predXList.append(Xhat)
            
        return predXList 
        
    def getMetricMethod(self): 
        return MCEvaluator.meanSqError
        
    def copy(self): 
        """
        Return a new copied version of this object. 
        """
        softImpute = SoftImpute(rhos=self.rhos, eps=self.eps, k=self.k)

        return softImpute 
        
    def name(self): 
        return "SoftImpute"
        

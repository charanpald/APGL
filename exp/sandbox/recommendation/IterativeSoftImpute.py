import numpy 
import logging 
import scipy.sparse.linalg 
import exp.util.SparseUtils as ExpSU
from apgl.util.SparseUtils import SparseUtils 
from apgl.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 
from apgl.util.Parameter import Parameter 
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter
from exp.util.SparseUtilsCython import SparseUtilsCython
from exp.sandbox.recommendation.SoftImpute import SoftImpute 
from exp.sandbox.SVDUpdate import SVDUpdate 

class IterativeSoftImpute(AbstractMatrixCompleter): 
    """
    Given a set of matrices X_1, ..., X_T find the completed matrices. 
    """
    def __init__(self, lmbda, eps=0.1, k=10, svdAlg="propack", updateAlg="initial", r=10, logStep=10):
        """
        Initialise imputing algorithm with given parameters. The lmbda is a value 
        for use with the soft thresholded SVD. Eps is the convergence threshold and 
        k is the rank of the SVD. 
        
        :param lmbda: The regularisation parameter for soft-impute 
        
        :param eps: The convergence threshold 
        
        :param k: The number of SVs to compute 
        
        :param svdAlg: The algorithm to use for computing a low rank + sparse matrix
        
        :param updateAlg: The algorithm to use for updating an SVD for a new matrix 
        
        :param r: The number of random projections to use for randomised SVD 
        """
        super(AbstractMatrixCompleter, self).__init__()   
        
        self.lmbda = lmbda  
        self.eps = eps
        self.k = k   
        self.svdAlg = svdAlg 
        self.updateAlg = updateAlg
        self.r = r 
        self.q = 2
        self.logStep = logStep
        
    def learnModel(self, matrixIterator):
        """
        Learn the matrix completion using a sparse matrix X. This is the simple 
        version of the soft impute algorithm in which we store the entire 
        matrices, newZ and oldZ. 
        """
        tol = 10**-6    
        ZList = []
        j = 0 
        
        for X in matrixIterator: 
            (n, m) = X.shape
            
            if j == 0: 
                oldU = numpy.zeros((n, 1))
                oldS = numpy.zeros(1)
                oldV = numpy.zeros((m, 1))   
            else: 
                oldN = oldU.shape[0]
                oldM = oldV.shape[0]
                
                if self.updateAlg == "initial":                 
                    if n > oldN: 
                        oldU = Util.extendArray(oldU, (n, oldU.shape[1]))
                    elif n < oldN: 
                        oldU = oldU[0:n, :]
                        
                    if m > oldM: 
                        oldV = Util.extendArray(oldV, (m, oldV.shape[1]))
                    elif m < oldN: 
                        oldV = oldV[0:m, :]
                elif self.updateAlg == "svdUpdate":
                    pass 
                elif self.updateAlg == "zero": 
                    oldU = numpy.zeros((n, 1))
                    oldS = numpy.zeros(1)
                    oldV = numpy.zeros((m, 1))   
                else: 
                    raise ValueError("Unknown SVD update algorithm: " + self.updateAlg)
            
            omega = X.nonzero()
            rowInds = numpy.array(omega[0], numpy.int)
            colInds = numpy.array(omega[1], numpy.int)
             
            gamma = self.eps + 1
            i = 0
            
            while gamma > self.eps:
                ZOmega = SparseUtilsCython.partialReconstruct2((rowInds, colInds), oldU, oldS, oldV)
                Y = X - ZOmega
                Y = Y.tocsc()
                
                if self.svdAlg=="propack": 
                    newU, newS, newV = ExpSU.SparseUtils.svdSparseLowRank(Y, oldU, oldS, oldV, self.k)
                elif self.svdAlg=="svdUpdate": 
                    newU, newS, newV = SVDUpdate.addSparseProjected(oldU, oldS, oldV, Y, self.k)
                elif self.svdAlg=="RSVD": 
                    newU, newS, newV = SVDUpdate.addSparseRSVD(oldU, oldS, oldV, Y, self.k, self.k, kRand=self.r, q=self.q)
                else: 
                    raise ValueError("Unknown SVD algorithm: " + self.svdAlg)
        
                #Soft threshold 
                newS = newS - self.lmbda
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
                
            logging.debug("Number of iterations for lambda="+str(self.lmbda) + ": " + str(i))
            
            ZList.append((newU,newS,newV))
            j += 1 
            
        return ZList

    def predict(self, ZList, indList): 
        """
        Make a set of predictions for a given set of completed matrices and 
        index lists. 
        """ 
        XhatList = []
        
        for i, Z in enumerate(ZList):
            U, s, V = Z 
            Xhat = ExpSU.SparseUtils.reconstructLowRank(U, s, V, indList[i])
            XhatList.append(Xhat)    
        
        return XhatList

    def setK(self, k):
        Parameter.checkInt(k, 1, float('inf'))
        
        self.k = k 
        
    def getK(self): 
        return self.k
        
    def setLambda(self, lmbda):
        Parameter.checkFloat(lmbda, 0.0, float('inf'))
        
        self.lmbda = lmbda 
        
    def getLambda(self): 
        return self.lmbda
        
    def getMetricMethod(self): 
        return MCEvaluator.meanSqError
        
    def copy(self): 
        """
        Return a new copied version of this object. 
        """
        iterativeSoftImpute = IterativeSoftImpute(lmbda=self.lmbda, eps=self.eps, k=self.k)

        return iterativeSoftImpute 
        
    def name(self): 
        return "IterativeSoftImpute"
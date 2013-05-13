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
from exp.sandbox.SVDUpdate import SVDUpdate 
from apgl.util.Sampling import Sampling 
from apgl.util.ProfileUtils import ProfileUtils 

class IterativeSoftImpute(AbstractMatrixCompleter): 
    """
    Given a set of matrices X_1, ..., X_T find the completed matrices. 
    """
    def __init__(self, lmbda=0.1, eps=0.1, k=None, svdAlg="propack", updateAlg="initial", r=10, logStep=10):
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
        if k != None:
            self.kmax = k*5
        else: 
            self.kmax = None 
        self.logStep = logStep
        
    def learnModel(self, XIterator):
        """
        Learn the matrix completion using an iterator which outputs 
        a sequence of sparse matrices X. The output of this method is also 
        an iterator which outputs a sequence of completed matrices. 
        """

        class ZIterator(object): 
            def __init__(self, XIterator, iterativeSoftImpute): 
                self.tol = 10**-6    
                self.j = 0 
                self.XIterator = XIterator
                self.iterativeSoftImpute = iterativeSoftImpute

            def __iter__(self): 
                return self 
                
            def next(self): 
                X = self.XIterator.next() 
                
                if not scipy.sparse.isspmatrix_csc(X): 
                    raise ValueError("X must be a csc_matrix")
                
                (n, m) = X.shape
                
                if self.j == 0: 
                    self.oldU = numpy.zeros((n, 1))
                    self.oldS = numpy.zeros(1)
                    self.oldV = numpy.zeros((m, 1))   
                else: 
                    oldN = self.oldU.shape[0]
                    oldM = self.oldV.shape[0]
                    
                    if self.iterativeSoftImpute.updateAlg == "initial":                 
                        if n > oldN: 
                            self.oldU = Util.extendArray(self.oldU, (n, self.oldU.shape[1]))
                        elif n < oldN: 
                            self.oldU = self.oldU[0:n, :]
                            
                        if m > oldM: 
                            self.oldV = Util.extendArray(self.oldV, (m, self.oldV.shape[1]))
                        elif m < oldN: 
                            self.oldV = self.oldV[0:m, :]
                    elif self.iterativeSoftImpute.updateAlg == "svdUpdate":
                        pass 
                    elif self.iterativeSoftImpute.updateAlg == "zero": 
                        self.oldU = numpy.zeros((n, 1))
                        self.oldS = numpy.zeros(1)
                        self.oldV = numpy.zeros((m, 1))   
                    else: 
                        raise ValueError("Unknown SVD update algorithm: " + self.updateAlg)
                
                omega = X.nonzero()
                rowInds = numpy.array(omega[0], numpy.int)
                colInds = numpy.array(omega[1], numpy.int)
                 
                gamma = self.iterativeSoftImpute.eps + 1
                i = 0
                
                while gamma > self.iterativeSoftImpute.eps:
                    ZOmega = SparseUtilsCython.partialReconstruct2((rowInds, colInds), self.oldU, self.oldS, self.oldV)
                    Y = X - ZOmega
                    Y = Y.tocsc()
                    
                    if self.iterativeSoftImpute.svdAlg=="propack": 
                        newU, newS, newV = ExpSU.SparseUtils.svdSparseLowRank(Y, self.oldU, self.oldS, self.oldV, k=self.iterativeSoftImpute.k, kmax=self.iterativeSoftImpute.kmax)
                    elif self.iterativeSoftImpute.svdAlg=="svdUpdate": 
                        newU, newS, newV = SVDUpdate.addSparseProjected(self.oldU, self.oldS, self.oldV, Y, self.iterativeSoftImpute.k)
                    elif self.iterativeSoftImpute.svdAlg=="RSVD": 
                        newU, newS, newV = SVDUpdate.addSparseRSVD(self.oldU, self.oldS, self.oldV, Y, self.iterativeSoftImpute.k, self.iterativeSoftImpute.k, kRand=self.iterativeSoftImpute.r, q=self.iterativeSoftImpute.q)
                    else: 
                        raise ValueError("Unknown SVD algorithm: " + self.iterativeSoftImpute.svdAlg)
            
                    #Soft threshold 
                    newS = newS - self.iterativeSoftImpute.lmbda
                    newS = numpy.clip(newS, 0, numpy.max(newS))
                    
                    normOldZ = (self.oldS**2).sum()
                    normNewZmOldZ = (self.oldS**2).sum() + (newS**2).sum() - 2*numpy.trace((self.oldV.T.dot(newV*newS)).dot(newU.T.dot(self.oldU*self.oldS)))
                    
                    #We can get newZ == oldZ in which case we break
                    if normNewZmOldZ < self.tol: 
                        gamma = 0
                    elif abs(normOldZ) < self.tol:
                        gamma = self.iterativeSoftImpute.eps + 1 
                    else: 
                        gamma = normNewZmOldZ/normOldZ
                    
                    self.oldU = newU.copy() 
                    self.oldS = newS.copy() 
                    self.oldV = newV.copy() 
                    
                    logging.debug("Iteration " + str(i) + " gamma="+str(gamma)) 
                    i += 1 
                    
                logging.debug("Number of iterations for lambda="+str(self.iterativeSoftImpute.lmbda) + ": " + str(i))
                
                self.j += 1
                return (newU,newS,newV)
                 
        return ZIterator(XIterator, self)

    def predict(self, ZIter, indList): 
        """
        Make a set of predictions for a given iterator of completed matrices and 
        an index list. 
        """         
        class ZTestIter(object):         
            def __init__(self): 
                self.i = 0 
            
            def __iter__(self): 
                return self 
                
            def next(self): 
                U, s, V = ZIter.next()
                Xhat = ExpSU.SparseUtils.reconstructLowRank(U, s, V, indList[self.i])
                self.i += 1 
                
                return Xhat 
  
        return ZTestIter()

    def modelSelect(self, X, lmbdas, cvInds):
        """
        Pick a value of lambda based on a single matrix X. We do cross validation 
        within, and return the best value of lambda (according to the mean 
        squared error). 
        """
        
        Xcoo = X.tocoo() 
        errors = numpy.zeros((lmbdas.shape[0], len(cvInds)))
        
        for i, (trainInds, testInds) in enumerate(cvInds): 
            Util.printIteration(i, 1, len(cvInds), "Fold: ")
            
            trainX = scipy.sparse.coo_matrix(X.shape)
            trainX.data = Xcoo.data[trainInds]
            trainX.row = Xcoo.row[trainInds]
            trainX.col = Xcoo.col[trainInds]
            trainX = trainX.tocsc()
            
            testX = scipy.sparse.coo_matrix(X.shape)
            testX.data = Xcoo.data[testInds]
            testX.row = Xcoo.row[testInds]
            testX.col = Xcoo.col[testInds]
            testX = testX.tocsc()
            
            testInds2 = testX.nonzero()
            
            for j, lmbda in enumerate(lmbdas): 
                Util.printIteration(j, 1, len(lmbdas), "Lambda: ")
                self.setLambda(lmbda)
                ZIter = self.learnModel(iter([trainX]))
                predXIter = self.predict(ZIter, [testInds2])
                
                predX = predXIter.next()
                errors[j, i] = MCEvaluator.meanSqError(testX, predX)
        
        meanErrors = errors.mean(1)

        return meanErrors 

    def setK(self, k):
        #Parameter.checkInt(k, 1, float('inf'))
        
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
# cython: profile=True  

"""
"""

from cython.operator cimport dereference as deref, preincrement as inc 
import cython
import struct
cimport numpy
cdef extern from "math.h":
    double sqrt(double x)
    
import numpy
import numpy.random
import scipy.sparse
import scipy.sparse.linalg 
import scipy.linalg
from exp.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 
import exp.util.SparseUtils as ExpSU
import logging
import copy

# for fast norm2 computation
@cython.boundscheck(False) # turn of bounds-checking for entire function   
cdef inline double norm2Diff(numpy.ndarray[double, ndim=2, mode="c"] M, numpy.ndarray[double, ndim=2, mode="c"] oldM):
    cdef double norm = 0.
    cdef double tmp
    cdef unsigned int i,j
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            tmp = M[i,j]-oldM[i,j]
            norm += tmp*tmp
    return sqrt(norm)

class SGDNorm2Reg(object):
    class ArithmeticError(ArithmeticError):
        def __init__(self):
            ArithmeticError.__init__(self)
        def __str__(self):
            return "infinite or NaN value encountered"
 
    def __init__(self, k, lmbda, eps, tmax, gamma = 1):
        """
        Initialise imputing algorithm with given parameters. lambda is the 
        regularisation parameter, eps is the convergence threshold, tmax is
        the maximum number of allowed iterations, and k is the rank of the
        decomposition.
        """
        self.k = k
        self.lmbda = lmbda  
        self.eps = eps
        self.tmax = tmax
        self.gamma = gamma
        
        # other parameters
        self.t0 = 1
        
    def learnModel(self, X, P=None, Q=None, Z=None, storeAll=True, nTry=3, skipError=True): 
        """
        Learn the matrix completion using a sparse matrix X.
        
        :param storeAll: Store and return a list of intermediate solutions P, Q
        
        When no initial point is given, expect the matrix to be centered
        in rows and columns. 
        """
        for devNull in range(nTry):
            try:
                return self._learnModel(X=X, P=P, Q=Q, Z=Z, storeAll=storeAll, skipError=False)
            except (FloatingPointError, ValueError, SGDNorm2Reg.ArithmeticError):
                pass
        return self._learnModel(X=X, P=P, Q=Q, Z=Z, storeAll=storeAll, skipError=skipError)
            
        

    @cython.boundscheck(False) # turn of bounds-checking for entire function   
    def _learnModel(self, X, P=None, Q=None, Z=None, storeAll=True, skipError=False): 
        """
        core of learnModel
        """
        # usefull
        cdef unsigned int m = X.shape[0]
        cdef unsigned int n = X.shape[1]
        cdef unsigned int k = self.k
        cdef double gamma = self.gamma
        cdef double lmbda = self.lmbda
        cdef double eps = self.eps
        cdef unsigned int tmax = self.tmax
        cdef int t0 = self.t0

        cdef double sX, sP, sQ
        if Z == None:
            if P == None and Q == None:
                sX = X.data.std()
                sP = sQ = sqrt(sX / sqrt(<double>(k)))
                P = numpy.random.randn(X.shape[0], k) * sP 
                Q = numpy.random.randn(X.shape[1], k) * sQ
            else:
                if P == None:
                    sX = X.data.std()
                    sQ = Q.std()
                    sP = sX / sQ / sqrt(<double>(k))
                    P = numpy.random.randn(X.shape[0], k) * sP 
                if Q == None:
                    sX = X.data.std()
                    sP = P.std()
                    sQ = sX / sP / sqrt(<double>(k))
                    Q = numpy.random.randn(X.shape[1], k) * sQ
        else:
            P,Q = Z[-1]
        
        # sanity check (to safely remove boundcheck)
        assert (P.shape[0] == m)
        assert (P.shape[1] == k)
        assert (Q.shape[0] == n)
        assert (Q.shape[1] == k)
        
        cdef numpy.ndarray[double, ndim=2, mode="c"] PP = P.copy()
        cdef numpy.ndarray[double, ndim=2, mode="c"] QQ = Q.copy()
        
        cdef unsigned int nnz = X.nnz
        omega = X.nonzero()
        cdef numpy.ndarray[int, ndim=1] omega0 = omega[0]
        cdef numpy.ndarray[int, ndim=1] omega1 = omega[1]
        cdef numpy.ndarray[double, ndim=1] nonzero = X.data
        cdef unsigned int t = 0
        cdef unsigned int nPass = 0
        
        ZList = []
        
        cdef unsigned int ii, u, i, kk, maxIter
        cdef double error, deltaPNorm, deltaQNorm, ge, gl, tmp
        cdef numpy.ndarray[double, ndim=2, mode="c"] oldP = scipy.zeros((m,k))
        cdef numpy.ndarray[double, ndim=2, mode="c"] oldQ = scipy.zeros((n,k))
        while True:
            if eps > 0:
                oldP[:] = PP[:]
                oldQ[:] = QQ[:]
            
            # do one pass on known values
            logging.debug("epoch " + str(nPass) + " (iteration " + str(t) + ")")
            nPass += 1
            maxIter = min(nnz, tmax-t)
            for ii in range(maxIter):
                u = omega0[ii]
                i = omega1[ii]
                
                error = nonzero[ii]
                for kk in range(k): # vector addition done by hand
                    error -= PP[u,kk] * QQ[i,kk]
                grad_weight = 1.*gamma/<double>(t+t0)
                grad_weight = 1.*gamma/sqrt(<double>(t+t0))
                ge = grad_weight * error
                gl = 1. - grad_weight * lmbda
                for kk in range(k): # vector addition done by hand
                    PP[u,kk], QQ[i,kk] = gl*PP[u,kk] + ge*QQ[i,kk], gl*QQ[i,kk] + ge*PP[u,kk]
                
                t += 1
                    
            if storeAll: 
                ZList.append((PP.copy(), QQ.copy()))
            
            # stop due to no change after a bunch of gradient steps
            if eps > 0:
                deltaPNorm = norm2Diff(PP, oldP)
                deltaQNorm = norm2Diff(QQ, oldQ)
                logging.debug("norm of DeltaP: " + str(deltaPNorm))
                logging.debug("norm of DeltaQ: " + str(deltaQNorm))
                if not scipy.isfinite(deltaPNorm) or not scipy.isfinite(deltaQNorm):
                    if skipError:
                        PP[:] = oldP[:]
                        QQ[:] = oldQ[:]
                        break
                    else:
                        raise SGDNorm2Reg.ArithmeticError()
                if deltaPNorm < eps and deltaQNorm < eps:
                    break
            
            # stop due to limited time budget
            if t >= tmax:
                break
                
        logging.debug("nb grad: " + str(t))

        if storeAll: 
            return ZList 
        else: 
            return [(PP.copy(), QQ.copy())] 

    def predict(self, ZList, inds, i=-1):
        """
        From i-th matrix returned by learnModel, predict the values of indices
        contained in inds.
        """
        U, V = ZList[i]
        Xhat = ExpSU.SparseUtils.reconstructLowRankPQ(U, V, inds)
        return Xhat

    def predictAll(self, ZList, inds):
        """
        Make a set of predictions for a given iterator of completed matrices and
        an index list.
        """
        predXList = []
        
        for i in range(len(ZList)): 
            predXList.append(self.predict(ZList, inds, i))
            
        return predXList 

    def getMetricMethod(self): 
        return MCEvaluator.meanSqError

    def copy(self): 
        """
        Return a new copied version of this object. 
        """

        return copy.copy(self) 
        
        

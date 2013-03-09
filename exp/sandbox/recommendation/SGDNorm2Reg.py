
"""
An implementation of the matrix completion algorithm based on stochastic gradient descent minimizing the objective function:
||M-PQ||_{F on known values}^2 + \lambda (||P||_F^2 + ||Q||_F^2)
"""

import numpy
import numpy.random
import scipy.sparse
import scipy.sparse.linalg 
import scipy.linalg
from apgl.util.MCEvaluator import MCEvaluator 
from apgl.util.Util import Util 
import logging

class SGDNorm2Reg(object): 
    def __init__(self, k, lmbda, eps, tmax):
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
        
        # other parameters
        self.t0 = 1
        self.gamma = 1
        
        
    def learnModel(self, X, P=None, Q=None): 
        """
        Learn the matrix completion using a sparse matrix X. 
        """
        
        if P == None:
            P = numpy.random.randn(X.shape[0], self.k)
        if Q == None:
            Q = numpy.random.randn(X.shape[1], self.k)
        omega = X.nonzero()
#        tol = 10**-6
        t = 1
        
        ZList = []
        
        while True: 
            oldP = P.copy()
            oldQ = Q.copy()
            
            # do one pass on known values
            logging.debug("one pass on the training matrix")
            for u,i in zip(omega[0], omega[1]):
                error = X[u,i] - P[u,:].dot(Q[i,:])
                if error > self.eps:
                    logging.debug(str(u) + " " + str(i) + ": " + str(error))
                grad_weight = self.gamma/(t+self.t0)
#                grad_weight = self.gamma/scipy.sqrt(t+self.t0)
                oldProw = P[u,:].copy()
                P[u,:] += grad_weight * (error*Q[i,:]-self.lmbda*P[u,:])
                Q[i,:] += grad_weight * (error*oldProw-self.lmbda*Q[i,:])
                
                # stop due to limited time budget
                if t >= self.tmax:
                    break;
                t += 1
                    
            ZList.append(scipy.sparse.csr_matrix(P).dot(scipy.sparse.csr_matrix(Q).T))
            
            # stop due to no change after a bunch of gradient steps
            logging.debug("norm of DeltaP: " + str(scipy.linalg.norm(P - oldP)))
            logging.debug("norm of DeltaQ: " + str(scipy.linalg.norm(Q - oldQ)))
            if scipy.linalg.norm(P - oldP) < self.eps and scipy.linalg.norm(Q - oldQ) < self.eps:
                break;
            
            # stop due to limited time budget
            if t >= self.tmax:
                break;
                
        if __debug__:
            logging.info("nb grad: " + str(t))

        return ZList 
        
    def getMetricMethod(self): 
        return MCEvaluator.meanSqError()
        

"""
A wrapper for the matrix factorisation in Nimfa. 
"""

import nimfa
from apgl.util.Parameter import Parameter 
from apgl.util.MCEvaluator import MCEvaluator 
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter

class NimfaFactorise(AbstractMatrixCompleter): 
    def __init__(self, method, rank=10, maxIter=10): 
        """
        Intialise the matrix factorisation with a given algorithm, rank and 
        max number of iterations. The rank can be a 1d array in which case 
        we use warm restarts to compute the full regularisation path. 
        """
        super(NimfaFactorise, self).__init__() 
        self.method = method  
        self.rank = rank 
        self.maxIter = maxIter
    
    def setRank(self, rank): 
        Parameter.checkInt(rank, 1, float("inf"))   
        rank = self.rank 
        
    def getRank(self): 
        return self.rank 
        
    def setMaxIter(self, maxIter): 
        Parameter.checkInt(maxIter, 1, float("inf"))
        maxIter = maxIter 
        
    def getMaxIter(self): 
        return self.maxIter 
    
    def learnModel(self, X):
        """
        Learn X using a matrix factorisation method. If self.rank is an integer 
        then we factorise with that rank. If it is an array then we compute the 
        complete regularisation path and return a list of matrices. 
        """
        if isinstance(self.rank, int): 
            model = nimfa.mf(X, method=self.method, max_iter=self.maxIter, rank=self.rank)
            fit = nimfa.mf_run(model)
            W = fit.basis()
            H = fit.coef()
            
            predX = W.dot(H)
            return predX 
        else: 
            predXList = []

            model = nimfa.mf(X, method=self.method, max_iter=self.maxIter, rank=self.rank[0])
            fit = nimfa.mf_run(model)
            W = fit.basis()
            H = fit.coef()
            predXList.append(W.dot(H))
            
            for i in range(1, self.rank.shape[0]): 
                model = nimfa.mf(X, method=self.method, max_iter=self.maxIter, rank=self.rank[i], W=W, H=H)
                fit = nimfa.mf_run(model)
                W = fit.basis()
                H = fit.coef()
                predXList.append(W.dot(H))

            return predXList

    def getMetricMethod(self): 
        return MCEvaluator.meanSqError
        
    def copy(self): 
        """
        Return a new copied version of this object. 
        """
        nimfaFactorise = NimfaFactorise(method=self.method, rank=self.rank, maxIter=self.maxIter)

        return nimfaFactorise 
        
    def name(self): 
        return "NimfaFactorise"
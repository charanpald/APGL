import numpy
import logging
import scipy.sparse.linalg
import exp.util.SparseUtils as ExpSU
from exp.util.MCEvaluator import MCEvaluator
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from exp.sandbox.recommendation.AbstractMatrixCompleter import AbstractMatrixCompleter


class IterativeMeanRating(AbstractMatrixCompleter):
    """
    Given a set of matrices X_1, ..., X_T find the completed matrices.
    """
    def __init__(self):
        """
        We just compute the mean rating. 
        """
        super(AbstractMatrixCompleter, self).__init__()


    def learnModel(self, XIterator):
        """
        Learn the matrix completion using an iterator which outputs
        a sequence of sparse matrices X. The output of this method is also
        an iterator which outputs a sequence of completed matrices in factorised 
        form. 
        
        :param XIterator: An iterator which emits scipy.sparse.csc_matrix objects 
        
        """
        class ZIterator(object):
            def __init__(self, XIterator):
                self.XIterator = XIterator  

            def __iter__(self):
                return self

            def next(self):
                X = self.XIterator.next()
                logging.debug("Learning on matrix with shape: " + str(X.shape) + " and " + str(X.nnz) + " non-zeros")                
                
                return X.shape, X.mean()

        return ZIterator(XIterator)

    def predict(self, ZIter, indList):
        """
        Make a set of predictions for a given iterator of completed matrices and
        an index list.
        """
        class ZTestIter(object):
            def __init__(self, iterativeSoftImpute):
                self.i = 0
                self.iterativeSoftImpute = iterativeSoftImpute

            def __iter__(self):
                return self

            def next(self):    
                Xhat = self.iterativeSoftImpute.predictOne(ZIter.next(), indList[self.i])  
                self.i += 1
                return Xhat 

        return ZTestIter(self)

    def predictOne(self, Z, inds): 
        shape, mean = Z

        Xhat = scipy.sparse.coo_matrix((numpy.ones(inds[0].shape[0])*mean, inds), shape=shape)
        Xhat = Xhat.tocsc()
    
        return Xhat

    def getMetricMethod(self):
        return MCEvaluator.meanSqError

    def copy(self):
        """
        Return a new copied version of this object.
        """
        iterativeMeanRating = IterativeMeanRating(lmbda=self.lmbda, eps=self.eps, k=self.k)

        return iterativeMeanRating

    def name(self):
        return "IterativeMeanRating"
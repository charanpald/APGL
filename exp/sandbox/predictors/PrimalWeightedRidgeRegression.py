
import numpy
import logging
from apgl.util import * 
from apgl.predictors.PrimalRidgeRegression import PrimalRidgeRegression

"""
Solve the least squares problem given by min (Xu - y)'W(Xu - y) + lmbda ||u||^2.
For the moment we just deal with single labels, not matrix ones so that the
weight matrix is diagonal. We compute W_ii = exp(y_i alpha). 
"""

class PrimalWeightedRidgeRegression(PrimalRidgeRegression):
    def __init__(self, lmbda=1.0, alpha=1.0):
        Parameter.checkFloat(lmbda, 0.0, float('inf'))
        Parameter.checkFloat(alpha, 0.0, float('inf'))
        self.lmbda = lmbda
        self.alpha = alpha

    def learnModel(self, X, Y):
        """
        Learn the weight matrix which matches X and Y with corresponding diagonal
        matrix W which weights cost of mispredictions on ith example. 
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkInt(X.shape[0], 1, float('inf'))
        Parameter.checkInt(X.shape[1], 1, float('inf'))

        if Y.ndim != 1:
            raise ValueError("Can only work on scalar labels")

        numExamples = X.shape[0]
        numFeatures = X.shape[1]

        logging.debug("Training with " + str(numExamples) + " examples and " + str(numFeatures) + " features")

        I = numpy.eye(numFeatures)
        W = self.getWeightMatrix(Y)
        XWX = X.T.dot(W).dot(X)
        XWY = X.T.dot(W).dot(Y)
        invXWX = numpy.linalg.inv(XWX + self.lmbda*I)

        self.U = numpy.dot(invXWX, XWY)

        return self.U

    def getWeightMatrix(self, Y):
        return numpy.diag(numpy.exp(Y*self.alpha))

    def __str__(self):
        return "PrimalWeightedRidgeRegression: lambda = " + str(self.lmbda) + " alpha = " + str(self.alpha)
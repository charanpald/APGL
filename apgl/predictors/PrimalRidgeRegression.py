
import numpy
import logging
from apgl.util.Parameter import Parameter
from apgl.predictors.AbstractPredictor import AbstractPredictor

"""
Solve the least squares problem given by min || XU - Y || + lmbda ||U|| 
"""

class PrimalRidgeRegression(AbstractPredictor):
    def __init__(self, lmbda=1.0):
        Parameter.checkFloat(lmbda, 0.0, float('inf'))
        self.lmbda = lmbda

    def setLambda(self, lmbda):
        Parameter.checkFloat(lmbda, 0.0, float('inf'))
        self.lmbda = lmbda

    def learnModel(self, X, Y):
        """
        Learn the weight matrix which matches X and Y.
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkInt(X.shape[0], 1, float('inf'))
        Parameter.checkInt(X.shape[1], 1, float('inf'))

        numExamples = X.shape[0]
        numFeatures = X.shape[1]

        logging.debug("Training with " + str(numExamples) + " examples and " + str(numFeatures) + " features")
        
        I = numpy.eye(numFeatures)
        XX = numpy.dot(X.T, X)
        XY = numpy.dot(X.T, Y)
        
        invXX = numpy.linalg.inv(XX + self.lmbda*I)

        self.U = numpy.dot(invXX, XY)
        logging.debug("Trace of X'X " + str(numpy.trace(XX)))
        logging.debug("Error " + str(numpy.linalg.norm(numpy.dot(X, self.U) - Y))) 

        return self.U

    def getWeights(self):
        return self.U 

    def predict(self, X):
        return numpy.dot(X, self.U)

    def classify(self, testX):
        """
        Classify a set of examples into {-1. +1} labels. Outputs the labels and
        decision values.
        """
        yPred = self.predict(testX)
        yClass = numpy.sign(yPred)
        yClass = yClass + numpy.array(yClass==0, numpy.int32)

        return yClass, yPred

    def __str__(self):
        return "PrimalRidgeRegression: lambda = " + str(self.lmbda)
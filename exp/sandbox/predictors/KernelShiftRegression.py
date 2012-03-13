
from apgl.graph import *
from apgl.util import *
from apgl.kernel import *
from apgl.predictors.AbstractKernelPredictor import AbstractKernelPredictor
import numpy

"""
We solve the follow optimisation min ||KA -jb' - Y|| + lmbda tr(B'KB), which is
quite similar to kernel ridge regression but with an additional shift vector a.
"""

class KernelShiftRegression(AbstractKernelPredictor):
    def __init__(self, kernel, lmbd=1.0):
        Parameter.checkFloat(lmbd, 0.0, float('inf'))
        Parameter.checkClass(kernel, AbstractKernel)

        self.lmbd = lmbd
        self.kernel = kernel

    def setLambda(self, lmbd):
        Parameter.checkFloat(lmbd, 0.0, float('inf'))
        self.lmbd = lmbd

    def setKernel(self, kernel):
        Parameter.checkClass(kernel, AbstractKernel)
        self.kernel = kernel

    def learnModel(self, trainX, trainY):
        numExamples = trainX.shape[0]
        K = self.kernel.evaluate(trainX, trainX)
        a = numpy.trace(K)/K.shape[0] * 10**-4
        #a = 10**-8
        K = K + numpy.eye(numExamples)*a
        K = (K + K) /2

        j = numpy.ones(numExamples)
        Kj = numpy.sum(K, 0)

        KK = numpy.dot(K, K)

        self.trainX = trainX
        self.alpha = numpy.linalg.inv(KK + self.lmbd * K - (float(1)/numExamples) * numpy.outer(Kj, Kj))
        self.alpha = numpy.dot(self.alpha, (float(1)/numExamples) * numpy.outer(Kj, j) + K)
        self.alpha = numpy.dot(self.alpha, trainY) 

        self.b = (float(1)/numExamples)*( numpy.dot(trainY.T, j) + numpy.dot(self.alpha.T, Kj) )

        return self.alpha, self.b


    def predict(self, testX):
        testTrainK = self.kernel.evaluate(testX, self.trainX)

        return numpy.dot(testTrainK, self.alpha) + self.b

    def classify(self, testX):
        """
        Classify a set of examples into {-1. +1} labels. Outputs the labels and
        decision values.
        """
        yPred = self.predict(testX)
        yClass = numpy.sign(yPred)
        yClass = yClass + numpy.array(yClass==0, numpy.int32)

        return yClass, yPred

    def getWeights(self):
        return self.alpha


    def __str__(self):
        return "KernelShiftRegression: lambda = " + str(self.lmbd) + ", kernel = " + str(self.kernel)
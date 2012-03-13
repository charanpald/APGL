
from apgl.graph import *
from apgl.util import *
from apgl.kernel import * 
from apgl.predictors.AbstractKernelPredictor import AbstractKernelPredictor
import numpy

class KernelRidgeRegression(AbstractKernelPredictor):
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
        a = numpy.trace(K)/K.shape[0] * 10**-6
        K = K + numpy.eye(numExamples)*a
        K = (K + K) /2

        KK = numpy.dot(K, K)

        self.trainX = trainX
        self.alpha = numpy.dot(numpy.linalg.inv(KK + self.lmbd * K), numpy.dot(K, trainY)) 
        return self.alpha


    def predict(self, testX):
        testTrainK = self.kernel.evaluate(testX, self.trainX)

        return numpy.dot(testTrainK, self.alpha)

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
        return "KernelRidgeRegression: lambda = " + str(self.lmbd) + ", kernel = " + str(self.kernel)
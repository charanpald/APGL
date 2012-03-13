"""

Predict a variable using a primal-dual CCA method.

"""
import numpy 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.features.PrimalDualCCA import PrimalDualCCA
from apgl.kernel import *
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter

class PrimalDualCCARegression(AbstractPredictor):
    def __init__(self, kernel, tau1, tau2):
        Parameter.checkFloat(tau1, 0.0, float('inf'))
        Parameter.checkFloat(tau2, 0.0, float('inf'))
        Parameter.checkClass(kernel, AbstractKernel)
        self.tau1 = tau1
        self.tau2 = tau2
        self.kernel = kernel

    def setTau1(self, tau1):
        Parameter.checkFloat(tau1, 0.0, float('inf'))
        self.tau1 = tau1

    def getTau1(self):
        return self.tau1

    def learnModel(self, X, Y):
        """
        Learn the weight matrix which matches X and Y.
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkInt(X.shape[0], 1, float('inf'))
        Parameter.checkInt(X.shape[1], 1, float('inf'))

        self.pdcca = PrimalDualCCA(self.kernel, self.tau1, self.tau2)
        alpha, V, lmbdas = self.pdcca.learnModel(X, Y)

        a = 10**-5
        I = numpy.eye(V.shape[0])
        VV = numpy.dot(V, V.T) + a*I

        self.A = Util.mdot(alpha, V.T, numpy.linalg.inv(VV))
        self.X = X

        return self.A

    def getWeights(self):
        return self.A

    def predict(self, X):
        return numpy.dot(self.kernel.evaluate(X, self.X), self.A)


    def __str__(self):
        return "PrimalDualCCARegressor: tau1 = " + str(self.tau1)
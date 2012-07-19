
import logging
import sys
import numpy
import unittest
import scipy.stats 
from apgl.predictors.ABCSMC import ABCSMC

class NormalModel(object):
    def __init__(self):
        self.mu = 1
        self.sigma = 1

    def setMu(self, mu):
        self.mu = mu

    def setSigma(self, sigma):
        self.sigma = sigma

    def simulate(self):
        return numpy.random.randn(100)*self.sigma + self.mu
        
    def setParams(self, paramsArray): 
        self.mu = paramsArray[0]
        self.sigma = paramsArray[1]


class ABCMetrics(object): 
    def distance(self, x, y):
        return numpy.linalg.norm(x-y)

    def summary(self, D):
        return numpy.array([D.mean(), D.std()])


class ABCParameters(object):
    def __init__(self):
        pass 

    def priorDensity(self, params):
        """
        This is the probability density of a particular theta
        """
        if params[0] > 0 and params[0] < 1 and params[1]>0 and params[1]<1:
            return 1
        else:
            return 0
    
    
    def sampleParams(self):
        mu = numpy.random.rand()
        sigma = numpy.random.rand()

        params = numpy.array([mu, sigma])
        return params
    

    def purtubationKernel(self, theta):
        """
        Take a theta and perturb it a bit
        """
        newTheta = theta.copy()
        variance = 0.02
        newTheta[0] = numpy.random.randn()*variance + theta[0]
        newTheta[1] = numpy.random.randn()*variance + theta[1]
        return newTheta

    def purtubationKernelDensity(self, theta, newTheta):
        variance = 0.02
        p = scipy.stats.norm.pdf(newTheta[0], loc=theta[0], scale=variance)
        p *= scipy.stats.norm.pdf(newTheta[1], loc=theta[1], scale=variance)
        return p


def createNormalModel(t):
    model = NormalModel()
    return model

class ABCSMCTest(unittest.TestCase):
    def setUp(self):
        FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)

    def testEstimate(self):
        #Lets set up a simple model based on normal dist
        abcParams = ABCParameters()
        createModelFunc = createNormalModel
        epsilonArray = numpy.array([0.5, 0.2, 0.1])
        posteriorSampleSize = 20

        #Lets get an empirical estimate of Sprime
        theta = [0.7, 0.5]
        model = NormalModel()
        model.setMu(theta[0])
        model.setSigma(theta[1])

        abcMetrics = ABCMetrics()
        
        Sprime = abcMetrics.summary(model.simulate()) 
        logging.debug(("Real summary statistic: " + str(Sprime)))

        abcSMC = ABCSMC(epsilonArray, Sprime, createModelFunc, abcParams, abcMetrics)
        abcSMC.setPosteriorSampleSize(posteriorSampleSize)
        thetasArray = abcSMC.run()
        
        thetasArray = numpy.array(thetasArray)

        meanTheta = numpy.mean(thetasArray, 0)
        logging.debug((thetasArray.shape))
        logging.debug(thetasArray)
        logging.debug(meanTheta)

        #Note only mean needs to be similar
        self.assertTrue(thetasArray.shape[0] == posteriorSampleSize)
        self.assertEquals(thetasArray.shape[1], 2)
        self.assertTrue(numpy.linalg.norm(theta[0] - meanTheta[0]) < 0.2)

if __name__ == "__main__":
    unittest.main()

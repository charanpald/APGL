
import logging
import sys
import numpy
import unittest
import multiprocessing
import scipy.stats 
from apgl.predictors.ABCSMC import ABCSMC

class ABCSMCTest(unittest.TestCase):
    def setUp(self):
        FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)

    def testEstimate(self):
        #Lets set up a simple model based on normal dist
        class NormalModel(object):
            def __init__(self):
                self.mu = 1
                self.sigma = 1

            def setMu(self, mu):
                self.mu = mu

            def setSigma(self, sigma):
                self.sigma = sigma

            def simulate(self):
                return numpy.random.randn(1)*self.sigma + self.mu
                
            def setParams(self, paramsArray): 
                self.mu = paramsArray[0]
                self.sigma = paramsArray[1]


        class ABCMetrics(object): 
            def distance(self, x, y):
                return numpy.abs(x-y)

            def summary(self, D):
                return D


        class ABCParameters(object):
            def __init__(self):
                pass 

            def priorDensity(self, params):
                """
                This is the probability density of a particular theta
                """
                if params[0] > 0 and params[0] < 2 and params[1]>0 and params[1]<0.2:
                    return 2.5
                else:
                    return 0

            def sampleParams(self):
                mu = numpy.random.rand()*2
                sigma = numpy.random.rand()*0.2

                params = [mu, sigma]
                return params


            def purtubationKernel(self, theta):
                """
                Take a theta and perturb it a bit
                """
                newTheta = theta
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
        
        abcParams = ABCParameters()
        createModelFunc = createNormalModel
        epsilonArray = numpy.array([0.2, 0.1, 0.05])
        posteriorSampleSize = 20

        #Lets get an empirical estimate of Sprime
        theta = [0.7, 0.5]
        model = NormalModel()
        model.setMu(theta[0])
        model.setSigma(theta[1])

        summaryArray = numpy.zeros(posteriorSampleSize)

        for i in range(posteriorSampleSize):
            summaryArray[i] = model.simulate()

        Sprime = numpy.mean(summaryArray)
        logging.debug(("Real summary statistic: " + str(Sprime)))

        #Create shared variables
        thetaQueue = multiprocessing.Queue()
        distQueue = multiprocessing.Queue()
        summaryQueue = multiprocessing.Queue()
        args = (thetaQueue, distQueue, summaryQueue)

        numProcesses = 2
        abcList = []
        
        abcMetrics = ABCMetrics()

        for i in range(numProcesses):
            abcList.append(ABCSMC(args, epsilonArray, Sprime, createModelFunc, abcParams, abcMetrics))
            abcList[i].setPosteriorSampleSize(posteriorSampleSize)
            abcList[i].start()

        for i in range(numProcesses):
            abcList[i].join()

        logging.debug(("Queue size = " + str(thetaQueue.qsize())))
        thetasArray = numpy.zeros((thetaQueue.qsize(), 2))

        for i in range(thetaQueue.qsize()):
            thetasArray[i, :] = numpy.array(thetaQueue.get())

        meanTheta = numpy.mean(thetasArray, 0)
        logging.debug((thetasArray.shape))
        logging.debug(thetasArray)
        logging.debug(meanTheta)

        #Note only mean needs to be similar
        self.assertTrue(thetasArray.shape[0] >= posteriorSampleSize)
        self.assertEquals(thetasArray.shape[1], 2)
        self.assertTrue(numpy.abs(theta[0] - meanTheta[0]) < 0.2)

if __name__ == "__main__":
    unittest.main()

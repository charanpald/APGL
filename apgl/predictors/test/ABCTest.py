import abc
import logging
import sys 
import numpy
import unittest
import multiprocessing 
from apgl.predictors.ABC import ABC

class ABCTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3)
        FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)


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
                #logging.debug("mu="+str(self.mu) + " sigma="+str(self.sigma))
                return numpy.random.randn(1)*self.sigma + self.mu

        class ABCParameters(object):
            def __init__(self, model):
                self.model = model
                
            def getParamFuncs(self):
                return [self.model.setMu, self.model.setSigma]

            def sampleParams(self):
                mu = numpy.random.rand()*1
                sigma = numpy.random.rand()*0.2

                params = [mu, sigma]
                return params

            @staticmethod 
            def distance(x, y):
                return numpy.abs(x-y)

            @staticmethod 
            def summary(D):
                return D

        def createNormalModel():
            model = NormalModel()
            abcParams = ABCParameters(model)
            return model, abcParams
            
        createModelFunc = createNormalModel
        epsilon = 0.01
        posteriorSampleSize = 100

        #Lets get an empirical estimate of Sprime
        theta = [0.5, 0.5]
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

        #Test the multiprocessing functionality 
        numProcesses = 2
        abcList = []

        for i in range(numProcesses):
            abcList.append(ABC(args, epsilon, Sprime, createModelFunc))
            abcList[i].setPosteriorSampleSize(posteriorSampleSize)
            abcList[i].start()

        for i in range(numProcesses):
            abcList[i].join()

        logging.debug(("Queue size = " + str(thetaQueue.qsize())))
        thetasArray = numpy.zeros((thetaQueue.qsize(), 2))

        for i in range(thetaQueue.qsize()):
            thetasArray[i, :] = numpy.array(thetaQueue.get())
            #logging.debug(distQueue.get())
            #logging.debug(summaryQueue.get())

        meanTheta = numpy.mean(thetasArray, 0)
        logging.debug(thetasArray)
        logging.debug(meanTheta)

        #Note only mean needs to be similar
        tol = 0.1
        self.assertTrue(numpy.abs(theta[0] - meanTheta[0]) < tol)
    

if __name__ == "__main__":
    unittest.main()

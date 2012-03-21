"""
A class to perform Approximate Bayesian Computation Sequential Monte Carlo which simulates observations
from a posterior distribution without the use of liklihoods.
"""
import os
import logging
import numpy
import multiprocessing
from datetime import datetime
from apgl.util.Util import Util 
from apgl.util.Parameter import Parameter 

class ABCSMC(multiprocessing.Process):
    def __init__(self, args, epsilonArray, Sprime, createModel, paramsObj, metrics):
        """
        Create a multiprocessing SMCABC object with the given arguments. The aim
        is to estimate a posterior pi(theta| x) propto f(x|theta) pi(theta) without
        requiring an explicit form of the likelihood. Here, theta is a set of
        parameters and x is a data observation.The algorithm can be run in a
        multiprocessing system.

        :param args: a tuple containing (theta, distance, summary) queues.
        
        :param epsilonArray: an array of successively smaller minimum distances
        :type epsilonArray: `numpy.ndarray` 
        
        :param Sprime: the summary statistics on real data
   
        :param createModel: A function to create a new stochastic model

        :param paramsObj: An object which stores information about the parameters of the model 
        
        :param metrics: An object to compute summary statistics and distances 
        """
        super(ABCSMC, self).__init__(args=args)

        dt = datetime.now()
        numpy.random.seed(dt.microsecond)
        self.args = args
        self.epsilonArray = epsilonArray
        self.Sprime = Sprime
        self.createModel = createModel
        self.abcParams = paramsObj 
        self.metrics = metrics

        #Number of particles
        self.T = epsilonArray.shape[0]
        #Size of population
        self.N = 10

    def setPosteriorSampleSize(self, posteriorSampleSize):
        """
        Set the sample size of the posterior distribution (population size).
        
        :param posteriorSampleSize: The size of the population 
        :type posteriorSampleSize: `int`
        """
        Parameter.checkInt(posteriorSampleSize, 0, numpy.float('inf'))
        self.N = posteriorSampleSize

    def getNumAcceptedTheta(self):
        """
        Returns the number of theta values accepted so far.
        """
        return self.args[0].qsize()

    def appendResults(self, theta, dist, summary):
        """
        Add new results in terms of theta, distance and the summary statistics.
        """
        self.args[0].put(theta)
        self.args[1].put(dist)
        self.args[2].put(summary)

    def run(self):
        """
        Make the estimation for a set of parameters theta close to the summary
        statistics S for a real dataset. 
        """
        iter = 0
        logging.debug("Parent PID: " + str(os.getppid()) + " Child PID: " + str(os.getpid()))
        currentWeights = numpy.zeros(self.N)
        currentTheta = []
        t = 0

        maxDist = 10**10

        while t < self.T and self.getNumAcceptedTheta() < self.N:
            i = 0
            lastTheta = currentTheta
            lastWeights = currentWeights
            currentTheta = []
            currentWeights = numpy.zeros(self.N)

            while i < self.N and self.getNumAcceptedTheta() < self.N:
                model = self.createModel(t)
                theta = self.abcParams.sampleParams()
                dist = self.epsilonArray[t] + 1

                while self.abcParams.priorDensity(theta) == 0 or dist > self.epsilonArray[t]:
                    if t == 0:
                        theta = self.abcParams.sampleParams()
                    else:
                        theta = lastTheta[Util.randomChoice(lastWeights)]
                        theta = self.abcParams.purtubationKernel(theta)

                    model = self.createModel(t)

                    #Can't simulate the model with theta if its density is zero
                    if self.abcParams.priorDensity(theta) != 0:
                        model.setParams(theta)
                        D = model.simulate()
                        S = self.metrics.summary(D)
                        dist = self.metrics.distance(S, self.Sprime)

                        if dist <= maxDist:
                            logging.debug("Best distance so far: theta=" + str(numpy.array(theta)) + " dist=" + str(dist))
                            maxDist = dist 
                    iter += 1

                logging.debug("Accepting particle " + str(i) + " at population " + str(t) + " " + "theta=" + str(numpy.array(theta))  + " dist=" + str(dist))
                currentTheta.append(theta)
                
                if t == self.T-1: 
                    self.appendResults(theta, dist, S)

                if t == 0:
                    currentWeights[i] = 1
                else:
                    normalisation = 0
                    for j in range(self.N):
                        normalisation += lastWeights[j]*self.abcParams.purtubationKernelDensity(lastTheta[j], theta)

                    currentWeights[i] = self.abcParams.priorDensity(theta)/normalisation
                i += 1

            currentWeights = currentWeights/numpy.sum(currentWeights)
            t += 1
        
        logging.debug("Finished ABC procedure, total iterations " + str(iter))

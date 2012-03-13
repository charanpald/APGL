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

class ABCSMC(multiprocessing.Process):
    def __init__(self, args, epsilonArray, Sprime, createModelFunc):
        """
        Create a multiprocessing SMCABC object with the given arguments. The aim
        is to estimate a posterior pi(theta| x) propto f(x|theta) pi(theta) without
        requiring an explicit form of the likelihood. Here, theta is a set of
        parameters and x is a data observation.The algorithm can be run in a
        multiprocessing system.

        :param args: a tuple containing (theta, distance, summary) queues.
        :param epsilonArray: an array of successively smaller minimum distances
        :param Sprime: the summary statistics on real data
        :param createModelFunc: A function to create and reset the stochastic model and parameters
        """
        super(ABCSMC, self).__init__(args=args)

        dt = datetime.now()
        numpy.random.seed(dt.microsecond)
        self.args = args
        self.epsilonArray = epsilonArray
        self.Sprime = Sprime
        self.createModelFunc = createModelFunc

        #Number of particles
        self.T = epsilonArray.shape[0]
        #Size of population
        self.N = 10

        #numpy.set_printoptions(suppress=True, precision=6)

    def setPosteriorSampleSize(self, posteriorSampleSize):
        """
        Set the sample size of the posterior distribution (population size).
        """
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
                model, abcParams = self.createModelFunc(t)
                theta = abcParams.sampleParams()
                dist = self.epsilonArray[t] + 1

                while abcParams.priorDensity(theta) == 0 or dist > self.epsilonArray[t]:
                    if t == 0:
                        theta = abcParams.sampleParams()
                    else:
                        theta = lastTheta[Util.randomChoice(lastWeights)]
                        theta = abcParams.purtubationKernel(theta)

                    model, abcParams = self.createModelFunc(t)
                    paramFuncs = abcParams.getParamFuncs()

                    #Can't simulate the model with theta if its density is zero
                    if abcParams.priorDensity(theta) != 0:
                        for j in range(len(theta)):
                            paramFuncs[j](theta[j])
                        D = model.simulate()
                        S = abcParams.summary(D)
                        dist = abcParams.distance(S, self.Sprime)

                        if dist <= maxDist:
                            logging.debug("Best distance so far: theta=" + str(numpy.array(theta)) + " dist=" + str(dist))
                            maxDist = dist 
                    iter += 1

                logging.debug("Accepting particle " + str(i) + " at population " + str(t))
                logging.debug("theta=" + str(numpy.array(theta))  + " dist=" + str(dist))
                currentTheta.append(theta)
                
                if t == self.T-1: 
                    self.appendResults(theta, dist, S)

                if t == 0:
                    currentWeights[i] = 1
                else:
                    normalisation = 0
                    for j in range(self.N):
                        normalisation += lastWeights[j]*abcParams.purtubationKernelDensity(lastTheta[j], theta)

                    currentWeights[i] = abcParams.priorDensity(theta)/normalisation
                i += 1

            currentWeights = currentWeights/numpy.sum(currentWeights)
            t += 1
        
        logging.debug("Finished ABC procedure, total iterations " + str(iter))

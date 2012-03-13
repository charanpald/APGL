"""
A class to perform Approximate Bayesian Computation which simulates observations
from a posterior distribution without the use of liklihoods. 
"""
import os
import logging
import numpy 
import multiprocessing
from datetime import datetime

class ABC(multiprocessing.Process):
    def __init__(self, args, epsilon, Sprime, createModelFunc):
        """
        Create a multiprocessing ABC object with the given multiprocessing
        arguments. The aim is to estimate a posterior
        pi(theta| x) propto f(x|theta) pi(theta) without requiring an explicit
        form of the likelihood. Here, theta is a set of parameters and x is a
        data observation.The algorithm can be run in a multiprocessing system.

        :param args: a tuple containing (theta, distance, summary) queues.
        :param epsilon: the minimum distance to accept thetas
        :param Sprime: the summary statistics on real data
        :param createModelFunc: A function to create a reset the stochastic model
        """
        super(ABC, self).__init__(args=args)

        dt = datetime.now()
        numpy.random.seed(dt.microsecond)
        self.args = args
        self.epsilon = epsilon
        self.Sprime = Sprime
        self.createModelFunc = createModelFunc 

        #The number of theta values to accept  
        self.posteriorSampleSize = 10

    def setPosteriorSampleSize(self, posteriorSampleSize):
        """
        Set the sample size of the posterior distribution. 
        """
        self.posteriorSampleSize = posteriorSampleSize
        
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
        statistics S for a real dataset. To get the resulting values, one can
        make a call to appendResults. 
        """
        iter = 0 
        logging.debug("Parent PID: " + str(os.getppid()) + " Child PID: " + str(os.getpid()))
        
        while self.getNumAcceptedTheta() < self.posteriorSampleSize:
            model, abcParams = self.createModelFunc()
            theta = abcParams.sampleParams()
            logging.debug("theta=" + str(theta))
            paramFuncs = abcParams.getParamFuncs()
            
            for j in range(len(theta)):
                paramFuncs[j](theta[j])

            D = model.simulate()
            S = abcParams.summary(D)
            dist = abcParams.distance(S, self.Sprime)
            logging.debug("dist=" + str(dist))

            if dist < self.epsilon:
                self.appendResults(theta, dist, S)
                logging.debug("Accepting " + str(self.getNumAcceptedTheta()) + "th theta = " + str(theta) + " minDist = " + str(dist) + " minS = " + str(S))

            iter += 1 

        logging.debug("Finished ABC procedure, total iterations " + str(iter))

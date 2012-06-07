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

def runModel(args):
    theta, createModel, metrics, Sprime, t = args 
    model = createModel(t)
    model.setParams(theta)
    D = model.simulate()
    del model 
    S = metrics.summary(D)
    del D
    dist = metrics.distance(S, Sprime) 
    return dist      

class ABCSMC(object):
    def __init__(self, epsilonArray, Sprime, createModel, paramsObj, metrics):
        """
        Create a multiprocessing SMCABC object with the given arguments. The aim
        is to estimate a posterior pi(theta| x) propto f(x|theta) pi(theta) without
        requiring an explicit form of the likelihood. Here, theta is a set of
        parameters and x is a data observation. The algorithm can be run in a
        multiprocessing system.
        
        :param epsilonArray: an array of successively smaller minimum distances
        :type epsilonArray: `numpy.ndarray` 
        
        :param Sprime: the summary statistics on real data
   
        :param createModel: A function to create a new stochastic model

        :param paramsObj: An object which stores information about the parameters of the model 
        
        :param metrics: An object to compute summary statistics and distances 
        """
        dt = datetime.now()
        numpy.random.seed(dt.microsecond)
        self.epsilonArray = epsilonArray
        self.Sprime = Sprime
        self.createModel = createModel
        self.abcParams = paramsObj 
        self.metrics = metrics

        #Number of particles
        self.T = epsilonArray.shape[0]
        #Size of population
        self.N = 10
        self.numProcesses = 8 

    def setPosteriorSampleSize(self, posteriorSampleSize):
        """
        Set the sample size of the posterior distribution (population size).
        
        :param posteriorSampleSize: The size of the population 
        :type posteriorSampleSize: `int`
        """
        Parameter.checkInt(posteriorSampleSize, 0, numpy.float('inf'))
        self.N = posteriorSampleSize
        
    def findTheta(self, lastTheta, lastWeights, t): 
        """
        Find a theta to accept. 
        """
        minDist = numpy.float("inf")
        tempTheta = self.abcParams.sampleParams()
        
        while minDist > self.epsilonArray[t]:
            thetaList = []   
            
            for i in range(self.numProcesses):             
                if t == 0:
                    tempTheta = self.abcParams.sampleParams()
                    thetaList.append((tempTheta.copy(), self.createModel, self.metrics, self.Sprime, t))
                else:  
                    while True: 
                        tempTheta = lastTheta[Util.randomChoice(lastWeights)]
                        tempTheta = self.abcParams.purtubationKernel(tempTheta)
                        if self.abcParams.priorDensity(tempTheta) != 0: 
                            break 
                    thetaList.append((tempTheta.copy(), self.createModel, self.metrics, self.Sprime, t))

            pool = multiprocessing.Pool(processes=self.numProcesses)               
            resultIterator = pool.map(runModel, thetaList)     
    
            i = 0 
            for dist in resultIterator: 
                if dist <= minDist:
                    logging.debug("Best distance so far: theta=" + str(numpy.array(thetaList[i][0])) + " dist=" + str(dist))
                    minDist = dist
                    bestTheta = thetaList[i][0]
                i += 1 
            pool.terminate()
            
        return bestTheta, minDist

    def run(self):
        """
        Make the estimation for a set of parameters theta close to the summary
        statistics S for a real dataset. 
        """
        logging.debug("Parent PID: " + str(os.getppid()) + " Child PID: " + str(os.getpid()))
        currentWeights = numpy.zeros(self.N)
        currentTheta = []

        for t in range(self.T):
            lastTheta = currentTheta
            lastWeights = currentWeights
            currentTheta = []
            currentWeights = numpy.zeros(self.N)

            for i in range(self.N):
                theta, minDist = self.findTheta(lastTheta, lastWeights, t)
                logging.debug("Accepting particle " + str(i) + " at population " + str(t) + " " + "theta=" + str(numpy.array(theta))  + " dist=" + str(minDist))
                currentTheta.append(theta)
                
                if t == 0:
                    currentWeights[i] = 1
                else:
                    normalisation = 0
                    for j in range(self.N):
                        normalisation += lastWeights[j]*self.abcParams.purtubationKernelDensity(lastTheta[j], theta)

                    currentWeights[i] = self.abcParams.priorDensity(theta)/normalisation

            currentWeights = currentWeights/numpy.sum(currentWeights)
        
        logging.debug("Finished ABC procedure") 
        
        return currentTheta 

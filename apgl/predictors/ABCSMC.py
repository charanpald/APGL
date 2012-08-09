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
    theta, createModel, t = args 
    
    logging.debug("Using theta value : " + str(theta)) 
    model = createModel(t)
    model.setParams(theta)
    model.simulate()
    dist = model.distance() 
    del model 
    return dist

class ABCSMC(object):
    def __init__(self, epsilonArray, createModel, paramsObj, thetaDir):
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
        self.createModel = createModel
        self.abcParams = paramsObj 
        self.thetaDir = thetaDir 

        #Number of particles
        self.T = epsilonArray.shape[0]
        #Size of population
        self.N = 10
        self.numProcesses = multiprocessing.cpu_count() 
        self.batchSize = self.numProcesses*2

    def setPosteriorSampleSize(self, posteriorSampleSize):
        """
        Set the sample size of the posterior distribution (population size).
        
        :param posteriorSampleSize: The size of the population 
        :type posteriorSampleSize: `int`
        """
        Parameter.checkInt(posteriorSampleSize, 0, numpy.float('inf'))
        self.N = posteriorSampleSize

    def loadThetas(self, t): 
        """
        Load all thetas saved for particle t. 
        """
        currentThetas = [] 
            
        for i in range(self.N): 
            fileName = self.thetaDir + "theta_t="+str(t)+"_"+str(i)+".npy"
            if os.path.exists(fileName): 
                currentThetas.append(numpy.load(fileName))
                
        return currentThetas 

    def findThetas(self, lastTheta, lastWeights, t): 
        """
        Find a theta to accept. 
        """
        tempTheta = self.abcParams.sampleParams()
        currentTheta = []
        
        while len(currentTheta) != self.N:
            thetaList = []   
            
            for i in range(self.batchSize):             
                if t == 0:
                    tempTheta = self.abcParams.sampleParams()
                    thetaList.append((tempTheta.copy(), self.createModel, t))
                else:  
                    while True: 
                        tempTheta = lastTheta[Util.randomChoice(lastWeights)]
                        tempTheta = self.abcParams.purtubationKernel(tempTheta)
                        if self.abcParams.priorDensity(tempTheta) != 0: 
                            break 
                    thetaList.append((tempTheta.copy(), self.createModel, t))

            pool = multiprocessing.Pool(processes=self.numProcesses)               
            resultIterator = pool.map(runModel, thetaList)     
            #resultIterator = map(runModel, thetaList)     
    
            i = 0 
            for dist in resultIterator: 
                if dist <= self.epsilonArray[t] and len(currentTheta) !=self.N:
                    logging.debug("Accepting particle " + str(len(currentTheta)) + " at population " + str(t) + " " + "theta=" + str(thetaList[i][0])  + " dist=" + str(dist))
                    currentTheta = self.loadThetas(t)                                        
                    fileName = self.thetaDir + "theta_t="+str(t)+"_"+str(len(currentTheta))
                    numpy.save(fileName, thetaList[i][0])
                    currentTheta.append(thetaList[i][0])
                i += 1 
            pool.terminate()
            
        return currentTheta

    def run(self):
        """
        Make the estimation for a set of parameters theta close to the summary
        statistics S for a real dataset. 
        """
        logging.debug("Parent PID: " + str(os.getppid()) + " Child PID: " + str(os.getpid()))
        currentTheta = []
        currentWeights = numpy.zeros(self.N)

        for t in range(self.T):
            lastTheta = currentTheta
            lastWeights = currentWeights
            currentWeights = numpy.zeros(self.N)

            currentTheta = self.findThetas(lastTheta, lastWeights, t)
                   
            for i in range(self.N):
                theta = currentTheta[i]                
                
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

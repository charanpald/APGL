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

def loadThetaArray(N, thetaDir, t): 
    """
    Load the thetas from a particular directory. 
    """
    currentThetas = [] 
    dists = []
        
    for i in range(N): 
        fileName = thetaDir + "theta_t="+str(t)+"_"+str(i)+".npz"
        if os.path.exists(fileName): 
            data = numpy.load(fileName)
            currentThetas.append(data["arr_0"])
            dists.append(data["arr_1"])
            
    return numpy.array(currentThetas), numpy.array(dists)

def runModel(args):
    theta, createModel, t, epsilon, N, thetaDir = args     
    currentTheta = loadThetaArray(N, thetaDir, t)[0].tolist()
    
    if len(currentTheta) < N:     
        logging.debug("Using theta value : " + str(theta)) 
        model = createModel(t)
        model.setParams(theta)
        model.simulate()
        dist = model.distance() 
        del model 
        
        currentTheta = loadThetaArray(N, thetaDir, t)[0].tolist()                
        
        if dist <= epsilon and len(currentTheta) < N:    
            logging.debug("Accepting " + str(len(currentTheta)) + " pop. " + str(t) + " " + str(theta)  + " dist=" + str(dist))
            fileName = thetaDir + "theta_t="+str(t)+"_"+str(len(currentTheta))
            
            distArray = numpy.array([dist])            
            numpy.savez(fileName, theta, distArray)
            currentTheta.append(theta)
            
        return 1 
    return 0 
            
class ABCSMC(object):
    def __init__(self, epsilonArray, createModel, paramsObj, thetaDir, autoEpsilon=False):
        """
        Create a multiprocessing SMCABC object with the given arguments. The aim
        is to estimate a posterior pi(theta| x) propto f(x|theta) pi(theta) without
        requiring an explicit form of the likelihood. Here, theta is a set of
        parameters and x is a data observation. The algorithm can be run in a
        multiprocessing system.
        
        :param epsilonArray: an array of successively smaller minimum distances
        :type epsilonArray: `numpy.ndarray` 
   
        :param createModel: A function to create a new stochastic model. The model must have a distance function with returns the distance to the target theta. 

        :param paramsObj: An object which stores information about the parameters of the model 
        
        :param thetaDir: The directory to store theta values 
        
        :param autoEpsilon: If autoEpsilon is true then the first value in epsilonArray is used as epsilon, and epsilonArray[t+1] is computed as the min dist for particles at t   
        """
        dt = datetime.now()
        numpy.random.seed(dt.microsecond)
        self.epsilonArray = epsilonArray
        self.createModel = createModel
        self.abcParams = paramsObj 
        self.thetaDir = thetaDir 
        self.autoEpsilon = autoEpsilon

        #Number of particles
        self.T = epsilonArray.shape[0]
        #Size of population
        self.N = 10
        self.numProcesses = multiprocessing.cpu_count() 
        self.batchSize = self.numProcesses*2
        self.numRuns = numpy.zeros(self.T) 
        self.maxRuns = 500

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
        return loadThetaArray(self.N, self.thetaDir, t)
        
    def findThetas(self, lastTheta, lastWeights, t): 
        """
        Find a theta to accept. 
        """
        tempTheta = self.abcParams.sampleParams()
        currentTheta = self.loadThetas(t)
        
        while len(currentTheta) < self.N:
            paramList = []   
            
            for i in range(self.batchSize):             
                if t == 0:
                    tempTheta = self.abcParams.sampleParams()
                    paramList.append((tempTheta.copy(), self.createModel, t, self.epsilonArray[t], self.N, self.thetaDir))
                else:  
                    while True: 
                        tempTheta = lastTheta[Util.randomChoice(lastWeights)[0], :]
                        tempTheta = self.abcParams.perturbationKernel(tempTheta)
                        if self.abcParams.priorDensity(tempTheta) != 0: 
                            break 
                    paramList.append((tempTheta.copy(), self.createModel, t, self.epsilonArray[t], self.N, self.thetaDir))

            pool = multiprocessing.Pool(processes=self.numProcesses)               
            resultsIterator = pool.map(runModel, paramList)     
            #resultsIterator = map(runModel, paramList)     

            for result in resultsIterator: 
                self.numRuns[t] += result
            
            if self.numRuns[t] >= self.maxRuns:
                logging.debug("Maximum number of runs exceeded.")
                break 
            
            currentTheta, dists = self.loadThetas(t)                 
            pool.terminate()
            
        if self.autoEpsilon and t!=self.T-1:
            self.epsilonArray[t+1] = numpy.mean(dists)
            logging.debug("Found new epsilon: " + str(self.epsilonArray))
              
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
            logging.debug("Particle number : " + str(t))
            lastTheta = currentTheta
            lastWeights = currentWeights
            currentWeights = numpy.zeros(self.N)

            currentTheta = self.findThetas(lastTheta, lastWeights, t)
            
            if len(currentTheta) != self.N: 
                break 
                   
            for i in range(self.N):
                theta = currentTheta[i]                
                
                if t == 0:
                    currentWeights[i] = 1
                else:
                    normalisation = 0
                    for j in range(self.N):
                        normalisation += lastWeights[j]*self.abcParams.perturbationKernelDensity(lastTheta[j], theta)
                        
                    currentWeights[i] = self.abcParams.priorDensity(theta)/normalisation
            
            currentWeights = currentWeights/numpy.sum(currentWeights)
        
        logging.debug("Finished ABC procedure") 
        
        return currentTheta 
        
    def setNumProcesses(self, numProcesses): 
        self.numProcesses = numProcesses 

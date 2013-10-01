import sys 
import apgl
import numpy 
import unittest
import logging
import pickle 
import scipy.integrate
from apgl.util.Util import Util 

from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVGraph import HIVGraph


@apgl.skipIf(not apgl.checkImport('sppy'), 'No module sppy')
class  HIVABCParametersTest(unittest.TestCase):
    def setUp(self):
        numpy.seterr(invalid='raise')
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(suppress=True, precision=4, linewidth=100)
        numpy.random.seed(21)

        M = 1000
        undirected = True

        graph = HIVGraph(M, undirected)
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)

        self.numParams = 6
        self.graph = graph
        self.meanTheta = numpy.array([100, 0.9, 0.05, 0.001, 0.1, 0.005])
        self.hivAbcParams = HIVABCParameters(self.meanTheta, self.meanTheta/2)
        

    def testGetParamFuncs(self):
        self.hivAbcParams.getParamFuncs()

    def testSampleParams(self):
        repetitions = 2000
        thetas = numpy.zeros((repetitions, self.numParams))

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.sampleParams()            
                
        #The deviation is half the mean       
        self.assertTrue(numpy.linalg.norm(numpy.mean(thetas, 0)/self.meanTheta - numpy.ones(self.numParams)) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(thetas, 0)/self.meanTheta) < 2)

        self.hivAbcParams = HIVABCParameters(self.meanTheta, self.meanTheta/10, 0.1)

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.sampleParams()            
        
        #The deviation is half the mean         
        self.assertTrue(numpy.linalg.norm(numpy.mean(thetas, 0)/self.meanTheta - numpy.ones(self.numParams)) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(thetas, 0)/self.meanTheta) < 2)

    def testPriorDensity(self):
        repetitions = 1000
        thetas = numpy.zeros((repetitions, self.numParams))
        densities = numpy.zeros((repetitions, self.numParams))

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.sampleParams()      
            densities[i, :] = self.hivAbcParams.priorDensity(thetas[i, :], True)
        
        for i in range(self.numParams): 
            inds = numpy.argsort(thetas[:, i])
            self.assertAlmostEquals(scipy.integrate.trapz(densities[inds, i], thetas[inds, i]), 1.0, 1)
            
        #Total density must be correct because they are independent 

    def testPurtubationKernel(self):
        repetitions = 1000
        theta = self.hivAbcParams.sampleParams()    
        newThetas = numpy.zeros((repetitions, self.numParams))
        
        for i in range(repetitions): 
            newThetas[i, :] = self.hivAbcParams.perturbationKernel(theta)

        self.assertTrue(numpy.linalg.norm(numpy.mean(newThetas, 0) - theta) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(newThetas, 0) - self.meanTheta/10) < 1)
         
        self.hivAbcParams = HIVABCParameters(theta, theta/20, purtScale=0.1)

        for i in range(repetitions): 
            newThetas[i, :] = self.hivAbcParams.perturbationKernel(theta)

        #print(numpy.std(newThetas, 0), self.meanTheta/10)

        self.assertTrue(numpy.linalg.norm(numpy.mean(newThetas, 0) - theta) < 2)
        #self.assertTrue(numpy.linalg.norm(numpy.std(newThetas, 0) - self.meanTheta/10) < 1)

    def testPurtubationKernelDensity(self):
        theta = self.hivAbcParams.sampleParams()
        repetitions = 1000
        thetas = numpy.zeros((repetitions, self.numParams))
        densities = numpy.zeros((repetitions, self.numParams))

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.perturbationKernel(theta)      
            densities[i, :] = self.hivAbcParams.perturbationKernelDensity(theta, thetas[i, :], True)
        
        for i in range(self.numParams): 
            inds = numpy.argsort(thetas[:, i])
            self.assertAlmostEquals(scipy.integrate.trapz(densities[inds, i], thetas[inds, i]), 1.0, 1)

    def testCreateTruncNormParamm(self):
        sigma = 0.5
        mu = 0.2
        numSamples = 2000
        sampleArray = numpy.zeros(numSamples)

        for i in range(numSamples):
            priorDist, priorDensity = self.hivAbcParams.createTruncNormParam(sigma, mu)
            sampleArray[i] = priorDist()
            self.assertTrue(sampleArray[i] <= 1.0 and sampleArray[i] >= 0)

        #A crude way of checking the mode
        hist = numpy.histogram(sampleArray, numpy.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]))

        #self.assertTrue(numpy.argmax(hist[0]) == 1)
        
    def testCreateDiscTruncNormParam(self): 
        sigma = 5.0
        mu = 10.0
        numSamples = 2000
        sampleArray = numpy.zeros(numSamples)
        upper = 100 

        for i in range(numSamples):
            priorDist, priorDensity = self.hivAbcParams.createDiscTruncNormParam(sigma, mu, upper)
            sampleArray[i] = priorDist()
            self.assertTrue(sampleArray[i] <= 100 and sampleArray[i] >= 0)
            
        self.assertAlmostEquals(numpy.min(sampleArray), 0, 1)
        self.assertTrue((numpy.mean(sampleArray) - mu)<1)
        #print(numpy.max(sampleArray))

    def testPickle(self):         
        output = pickle.dumps(self.hivAbcParams)
        newParams = pickle.loads(output)
             
        thetas = numpy.zeros((100, self.numParams))
        for i in range(thetas.shape[0]): 
            thetas[i, :] = newParams.sampleParams()
            
        meanTheta = thetas.mean(0)
        self.assertTrue(numpy.linalg.norm(meanTheta - self.hivAbcParams.meanTheta) < 1)
         
    def testCreateGammaParam(self): 
        sigma = 0.1
        mu = 0.2
        numSamples = 2000
        sampleArray = numpy.zeros(numSamples)

        for i in range(numSamples):
            priorDist, priorDensity = self.hivAbcParams.createGammaParam(sigma, mu)
            sampleArray[i] = priorDist()
        
        self.assertTrue(numpy.linalg.norm(mu - numpy.mean(sampleArray)) < 0.5)
        
        
        sigma = 0.0001
        mu = 0.2
        numSamples = 2000
        sampleArray = numpy.zeros(numSamples)

        for i in range(numSamples):
            priorDist, priorDensity = self.hivAbcParams.createGammaParam(sigma, mu)
            sampleArray[i] = priorDist()
            
        print(numpy.mean(sampleArray))


if __name__ == '__main__':
    unittest.main()


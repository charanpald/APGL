import sys 
import apgl
import numpy 
import unittest
import logging
import scipy.integrate
from apgl.util.Util import Util 

from exp.viroscopy.model.HIVABCParameters import HIVABCParameters
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVGraph import HIVGraph


@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
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

        self.numParams = 10
        self.graph = graph
        self.meanTheta = numpy.array([50, 1.0, 0.5, 1.0/800, 0.01, 0.05, 0.1, 38.0/1000, 30.0/1000, 170.0/1000])
        self.hivAbcParams = HIVABCParameters(self.meanTheta)
        

    def testGetParamFuncs(self):
        self.hivAbcParams.getParamFuncs()

    def testSampleParams(self):
        repetitions = 1000
        thetas = numpy.zeros((repetitions, 10))

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.sampleParams()            
        
        #The deviation is half the mean       
        self.assertTrue(numpy.linalg.norm(numpy.mean(thetas, 0) - self.meanTheta) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(thetas, 0) - self.meanTheta/2) < 0.7)

        self.hivAbcParams = HIVABCParameters(self.meanTheta, 0.1)

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.sampleParams()            
        
        #The deviation is half the mean         
        self.assertTrue(numpy.linalg.norm(numpy.mean(thetas, 0) - self.meanTheta) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(thetas, 0) - self.meanTheta/10) < 0.7)

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
            newThetas[i, :] = self.hivAbcParams.purtubationKernel(theta)

        self.assertTrue(numpy.linalg.norm(numpy.mean(newThetas, 0) - theta) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(newThetas, 0) - self.meanTheta/10) < 1)
         
        self.hivAbcParams = HIVABCParameters(theta, purtScale=0.1)

        for i in range(repetitions): 
            newThetas[i, :] = self.hivAbcParams.purtubationKernel(theta)

        self.assertTrue(numpy.linalg.norm(numpy.mean(newThetas, 0) - theta) < 2)
        self.assertTrue(numpy.linalg.norm(numpy.std(newThetas, 0) - self.meanTheta/20) < 1)

    def testPurtubationKernelDensity(self):
        theta = self.hivAbcParams.sampleParams()
        repetitions = 1000
        thetas = numpy.zeros((repetitions, self.numParams))
        densities = numpy.zeros((repetitions, self.numParams))

        for i in range(repetitions): 
            thetas[i, :] = self.hivAbcParams.purtubationKernel(theta)      
            densities[i, :] = self.hivAbcParams.purtubationKernelDensity(theta, thetas[i, :], True)
        
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

if __name__ == '__main__':
    unittest.main()


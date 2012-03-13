
import apgl
import numpy 
import unittest
import logging
from apgl.util.Util import Util 

from apgl.viroscopy.model.HIVABCParameters import HIVABCParameters
try:
    from apgl.viroscopy.model.HIVRates import HIVRates
    from apgl.viroscopy.model.HIVGraph import HIVGraph
except ImportError:
    pass

@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
class  HIVABCParametersTest(unittest.TestCase):
    def setUp(self):
        numpy.seterr(invalid='raise')

        M = 1000
        undirected = True

        graph = HIVGraph(M, undirected)
        logging.debug("Created graph: " + str(graph))
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)

        self.graph = graph
        self.meanTheta = [50, 1.0, 0.5, 1.0/800, 0.01, 0.05, 0.1, 38.0/1000, 30.0/1000, 170.0/1000]
        self.hivAbcParams = HIVABCParameters(graph,rates, self.meanTheta)

    def testGetParamFuncs(self):
        self.hivAbcParams.getParamFuncs()

    def testSampleParams(self):
        theta = self.hivAbcParams.sampleParams()
        logging.debug(theta)

    def testPriorDensity(self):
        theta = self.hivAbcParams.sampleParams()
        logging.debug("theta=" + str(theta))

        density = self.hivAbcParams.priorDensity(theta)
        logging.debug(("density=" + str(density)))

    def testPurtubationKernel(self):
        theta = self.hivAbcParams.sampleParams()
        newTheta = self.hivAbcParams.purtubationKernel(theta)

        logging.debug(theta)
        logging.debug(newTheta)

    def testPurtubationKernelDensity(self):
        theta = self.hivAbcParams.sampleParams()
        newTheta = self.hivAbcParams.purtubationKernel(theta)

        density = self.hivAbcParams.purtubationKernelDensity(theta, newTheta)
        logging.debug(("density=" + str(density)))

    def testDistance(self):
        u = numpy.array([[1, 2, 3], [4, 5, 6]]).T
        u2 = numpy.array([[2, 3, 4], [6, 7, 8]]).T
        u3 = numpy.array([[2, 3, 5], [6, 7, 8]]).T
        uPrime = numpy.array([[1, 2, 3], [4, 5, 6]]).T

        self.assertEquals(HIVABCParameters.distance(u, uPrime), 0)

        self.assertEquals(HIVABCParameters.distance(u2, uPrime), 0)

        self.assertEquals(HIVABCParameters.distance(u3, uPrime), 1)

        u = u[0:2, :]
        self.assertEquals(HIVABCParameters.distance(u, uPrime), 0)

        u = numpy.array([1, 2, 3, 4])
        self.assertRaises(ValueError, HIVABCParameters.distance, u, uPrime)

    def testCreateGammaParam(self):
        sigma = 0.5
        mu = 10.0

        numSamples = 1000
        sampleArray = numpy.zeros(numSamples)

        for i in range(numSamples):
            priorDist, priorDensity = self.hivAbcParams.createGammaParam(sigma, mu)
            sampleArray[i] = priorDist()

        self.assertAlmostEquals(numpy.mean(sampleArray), mu, 1)
        self.assertAlmostEquals(numpy.std(sampleArray), sigma, 1)

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
        logging.debug(hist)

        #self.assertTrue(numpy.argmax(hist[0]) == 1)


    def testSummary(self):
        times = [0, 5, 10]
        infectedIndices = [range(5), range(10), range(15)]
        removedIndices = [range(5), range(10), range(15)]

        D = (times, infectedIndices, removedIndices, self.graph)

        #It's correct
        self.hivAbcParams.summary(D)

        #Now test in empty case
        infectedIndices = [[], range(10), range(15)]
        removedIndices = [[], range(10), range(15)]

        D = (times, infectedIndices, removedIndices, self.graph)
        self.hivAbcParams.summary(D)

if __name__ == '__main__':
    unittest.main()


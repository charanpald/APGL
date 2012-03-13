import unittest
import numpy
import logging
from apgl.sandbox.TemporalKMeans import TemporalKMeans
from apgl.graph import *


class TemporalKMeansTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=False, precision=3)
        numpy.random.seed(21)

    def testUnNormSpectralClusterer(self):
        k = 2
        tau = 0.1

        numExamples = 10
        clusterSize = int(numpy.floor(numExamples/2))
        numFeatures = 2

        Clstr1 = numpy.random.randn(clusterSize, numFeatures) + numpy.array([1, 2])
        Clstr2 = numpy.random.randn(clusterSize, numFeatures) + numpy.array([-1, -2])
        X1 = numpy.r_[Clstr1, Clstr2]

        Clstr1 = numpy.random.randn(clusterSize, numFeatures) + numpy.array([3, 1])
        Clstr2 = numpy.random.randn(clusterSize, numFeatures) + numpy.array([-2, -4])
        X2 = numpy.r_[Clstr1, Clstr2]

        XList = [X1, X2]
        temporalKMeans = TemporalKMeans()
        C, muList = temporalKMeans.cluster(XList, k, tau)

        logging.debug(C)
        logging.debug(numpy.sum(C[:, 0] == C[:, 1]))

        for i in range(len(muList)):
            logging.debug(muList[i])


if __name__ == '__main__':
    unittest.main()


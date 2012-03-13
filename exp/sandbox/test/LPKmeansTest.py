

import unittest
import numpy
import apgl
import logging
from apgl.graph import *
try:
    from apgl.clustering.LPKMeans import LPKMeans
except ImportError:
    pass

@apgl.skipIf(not apgl.checkImport('cvxopt'), 'No module cvxopt')
class LPKMeansTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3)
        numpy.random.seed(21)
        numpy.set_printoptions(threshold=numpy.nan, linewidth=100)

        pass

    def testCluster(self):
        numFeatures = 2

        X1 = numpy.random.randn(30, numFeatures) + numpy.array([1, 1])
        X2 = numpy.random.randn(30, numFeatures) + numpy.array([-1, -1])
        X = numpy.r_[X1, X2]

        logging.debug(X.shape)

        k = 2

        lpkmeans = LPKMeans()

        sol = lpkmeans.cluster(X, k)
        z = sol['x']

        #Z = numpy.reshape(z, (5, 4))
        #logging.debug(Z)


if __name__ == '__main__':
    unittest.main()

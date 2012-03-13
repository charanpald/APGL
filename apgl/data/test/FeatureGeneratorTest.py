# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
import numpy
import logging
from apgl.data.FeatureGenerator import FeatureGenerator 

class  FeatureGeneratorTest(unittest.TestCase):
    def setUp(self):
        pass

    def testCategoricalToIndicator(self):
        X = numpy.zeros((5,5))
        X[:, 0] = numpy.array([1, 1, 2, 4, 6])
        X[:, 1] = numpy.array([2, 1, 2, 4, 6])
        X[:, 2] = numpy.array([1, 1, 2, 4, 2])
        X[:, 3] = numpy.array([1, 2, 3, 4, 2])
        X[:, 4] = numpy.array([1.1, 2.1, 4.5, 6.2, 1.1])

        logging.debug(X)

        generator = FeatureGenerator()
        inds = [0, 1]
        X2 = generator.categoricalToIndicator(X, inds)

        X3 = numpy.zeros((5, 11))
        X3[0, :] = numpy.array([[ 1,   0,   0,   0,   0,   1,   0,   0,   1,   1,   1.1]])
        X3[1, :] = numpy.array([[ 1,   0,   0,   0,   1,   0,   0,   0,   1,   2,   2.1]])
        X3[2, :] = numpy.array([[ 0,   1,   0,   0,   0,   1,   0,   0,   2,   3,   4.5]])
        X3[3, :] = numpy.array([[ 0,   0,   1,   0,   0,   0,   1,   0,   4,   4,   6.2]])
        X3[4, :] = numpy.array([[ 0,   0,   0,   1,   0,   0,   0,   1,   2,   2,   1.1]])

        self.assertTrue(numpy.linalg.norm(X3-X2) < 10**-6)

        #Test case where no indices given
        inds = []
        X2 = generator.categoricalToIndicator(X, inds)

        self.assertTrue(numpy.linalg.norm(X-X2) < 10**-6)

if __name__ == '__main__':
    unittest.main()

